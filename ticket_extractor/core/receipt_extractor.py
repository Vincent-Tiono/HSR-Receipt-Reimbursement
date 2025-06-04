from openai import OpenAI
import base64
import json
import numpy as np
import io
from PIL import Image
import torch
from torchvision import transforms
from transformers import Mask2FormerForUniversalSegmentation
import os
from dotenv import load_dotenv
from ..models.receipt import ReceiptLLM
from .ocr import detect_text
import cv2

# Load environment variables from .env file
load_dotenv()

# Set device and precision
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Initialize the model
try:
    print("Attempting to load image segmentation model...")
    # Using the correct class for Mask2Former model
    birefnet = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic",
        trust_remote_code=True
    )
    birefnet.to(DEVICE)
    print("Image segmentation model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to basic image processing without segmentation")
    birefnet = None

def process(image_array: np.ndarray) -> np.ndarray:
    """Process an image to remove the background."""
    if birefnet is None:
        # Fallback to basic processing without segmentation
        # Just convert to RGBA with full opacity
        height, width = image_array.shape[:2]
        alpha = np.ones((height, width, 1), dtype=np.uint8) * 255
        return np.concatenate([image_array, alpha], axis=2)
    
    # Convert numpy array to tensor
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
    image_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image_tensor)
    input_images = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    mask = pred.numpy()
    
    # Resize mask to match input image dimensions
    height, width = image_array.shape[:2]
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Add alpha channel
    rgba = np.concatenate([image_array, (mask * 255).astype(np.uint8)[..., None]], axis=2)
    return rgba

def crop_transparent(img_array: np.ndarray, padding=5) -> np.ndarray:
    """Crop image to remove transparent areas, with optional padding."""
    alpha = img_array[:, :, 3]
    non_transparent_pixels = np.where(alpha > 0)
    
    if len(non_transparent_pixels[0]) == 0:
        return img_array
    
    min_y, max_y = np.min(non_transparent_pixels[0]), np.max(non_transparent_pixels[0])
    min_x, max_x = np.min(non_transparent_pixels[1]), np.max(non_transparent_pixels[1])
    
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(img_array.shape[1], max_x + padding)
    max_y = min(img_array.shape[0], max_y + padding)
    
    return img_array[min_y:max_y, min_x:max_x]

def process_image(img_b64: str, padding=10) -> str:
    """Process the image in memory without saving to disk.
    
    Args:
        img_b64: Base64 encoded image string
        padding: Padding to add around the cropped image
    Returns:
        Base64 encoded processed image string
    """
    # Convert base64 to numpy array
    img_bytes = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_bytes))
    img_array = np.array(img.convert("RGB"))
    
    # Process image
    transparent = process(img_array)
    cropped = crop_transparent(transparent, padding=padding)
    
    # Convert directly to base64
    # First convert to bytes using cv2
    # Convert RGBA to BGRA for OpenCV
    bgra = cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA)
    success, buffer = cv2.imencode('.png', bgra)
    if not success:
        raise ValueError("Failed to encode image")
    # Then to base64
    processed_img_b64 = base64.b64encode(buffer).decode('utf-8')
    return processed_img_b64

def extract(img_b64: str) -> ReceiptLLM:
    """Send the image to the Gemini model and extract ticket information."""
    try:
        # Process image to remove background and crop
        processed_img_b64 = process_image(img_b64, padding=10)

        #########################################################
        # OCR
        #########################################################
        ocr_text = detect_text(processed_img_b64)
        
        prompt = (
            'Here is an image of a receipt. Extract the following information and return it in JSON format:'
            '- invoice_date (format: yyyy-mm-dd),'
            '- invoice_number (strict format: XX-XXXXXXXX, where XX are two English letters followed by hyphen and 8 digits),'
            '- seller_id (8-digit number following 賣方),'
            '- total_amount (integer number, remove 元),'
            'Return the response in this exact JSON format: {"invoice_date": "yyyy-mm-dd", "invoice_number": "XX-XXXXXXXX", "seller_id": "12345678", "total_amount": "123"}'
            'If you are unable to extract any of the required information, return {"error": "unable to process"}'
        )
        
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{processed_img_b64}"
                            }
                        }
                    ],
                }
            ],
        )
        
        # Get the response content
        response_text = completion.choices[0].message.content
        
        # Remove markdown code block markers if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()
        
        # Try to parse as JSON
        try:
            response = json.loads(response_text)
            
            # Check if response contains error
            if isinstance(response, dict) and "error" in response:
                raise ValueError(response["error"])
            
            validation = {
                "invoice_date": response["invoice_date"] in ocr_text,
                "invoice_number": response["invoice_number"] in ocr_text,
                "seller_id": response["seller_id"] in ocr_text,
                "total_amount": response["total_amount"] in ocr_text
            }
            
            if isinstance(response, dict):
                if "invoice_date" in response:
                    response["invoice_date"] = response["invoice_date"].replace("-", "/").strip()
            
            # Convert to ReceiptLLM model
            return ReceiptLLM(
                invoice_date=response["invoice_date"],
                invoice_number=response["invoice_number"],
                seller_id=response["seller_id"],
                total_amount=response["total_amount"],
                val_invoice_date=validation["invoice_date"],
                val_invoice_number=validation["invoice_number"],
                val_seller_id=validation["seller_id"],
                val_total_amount=validation["total_amount"]
            )
        except json.JSONDecodeError as e:
            # If the response contains error messages about being unable to process
            if "unable to process" in response_text.lower() or "not clear enough" in response_text.lower():
                raise ValueError("unable to process")
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            raise ValueError("unable to process")
    except Exception as e:
        print(f"Error in extract function: {e}")
        raise ValueError("unable to process") 
    
if __name__ == "__main__":
    with open("test.txt", "r") as f:
        img_b64 = f.read()
    print(extract(img_b64))