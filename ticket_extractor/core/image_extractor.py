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
from ..models.ticket import TicketLLM
from .ocr import detect_text
import cv2

# from pydantic import BaseModel
# from typing import Dict, Any

# class TicketLLM(BaseModel):
#     """Base response model for ticket information from Gemini"""
#     date: str
#     price: int
#     departure_station: str
#     arrival_station: str
#     serial_number: str
#     val_date: bool
#     val_price: bool
#     val_departure_station: bool
#     val_arrival_station: bool
#     val_serial_number: bool

# import os
# import requests
# from dotenv import load_dotenv

# load_dotenv()
# api_key = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    
# def detect_text(content):
#     """Detects text in the file using REST API."""

#     url = f'https://vision.googleapis.com/v1/images:annotate?key={api_key}'
    
#     request_data = {
#         'requests': [{
#             'image': {
#                 'content': content
#             },
#             'features': [{
#                 'type': 'TEXT_DETECTION',
#                 'maxResults': 1
#             }]
#         }]
#     }
    
#     # Make the API request
#     response = requests.post(url, json=request_data)
#     response.raise_for_status()  # Raise an exception for bad status codes
    
#     # Process the response
#     result = response.json()
    
#     if 'responses' in result and result['responses']:
#         texts = result['responses'][0].get('textAnnotations', [])
        
#         if texts:
#             # Get only the first text annotation which contains the complete text
#             complete_text = texts[0].get('description', '')
#             print("Text extracted")

#             return complete_text
#         else:
#             print("No text found in the response")
#             return None
#     else:
#         print("No text found in the response")
#         return None
    

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

def extract(img_b64: str) -> TicketLLM:
    """Send the image to the Gemini model and extract ticket information."""
    try:
        # Process image to remove background and crop
        processed_img_b64 = process_image(img_b64, padding=10)

        #########################################################
        # OCR
        #########################################################
        ocr_text = detect_text(processed_img_b64)

        print(ocr_text)

        
        # prompt = (
        #     'Here is an image of an HSR ticket. Extract the following information and return it in JSON format:'
        #     '- date (format: yyyy/mm/dd),'
        #     '- price,'
        #     '- Departure station in English,'
        #     '- Arrival station in English,'
        #     '- 票號 (13-digit strict format: XX-X-XX-X-XXX-XXXX),'
        #     'Return the response in this exact JSON format: {"date": "yyyy/mm/dd", "price": number, "departure_station": "station name", "arrival_station": "station name", "serial_number": "XX-X-XX-X-XXX-XXXX"}'
        #     'If you are unable to extract any of the required information, return {"error": "unable to process"}'
        # )
        

        prompt = (
            'Here is an image of an HSR ticket. Extract the following information and return it in JSON format:'
            '- date,'
            '- price,'
            '- Departure station in English,'
            '- Arrival station in English,'
            '- 票號 (13-digit integer with format XX-X-XX-X-XXX-XXXX or XXXXXXXXXXXXX),'
            'Return the response in JSON format with keys: date, price, departure_station, arrival_station, serial_number'
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
            
            # Validate extracted information against OCR text
            validation = {
                "date": response["date"] in ocr_text,
                "price": str(response["price"]) in ocr_text,
                "departure_station": response["departure_station"] in ocr_text,
                "arrival_station": response["arrival_station"] in ocr_text,
                "serial_number": str(response["serial_number"]) in ocr_text
            }
            
            # print("Validation results:", validation)

            if isinstance(response, dict):
                if "price" in response:
                    price_str = str(response["price"])
                    if "NT$" in price_str:
                        response["price"] = price_str.replace("NT$", "").strip()
                # Extract numeric price
                if "date" in response:
                    response["date"] = response["date"].replace("-", "/").strip()

                if "serial_number" in response:
                    serial_str = str(response["serial_number"])
                    response["serial_number"] = serial_str.replace("-", "").strip()
                
                if "price" in response:
                    price_str = str(response["price"])
                    if "NT$" in price_str:
                        response["price"] = int(price_str.replace("NT$", "").strip())
                
                # Remove "THSR" from station names
                if "departure_station" in response:
                    response["departure_station"] = response["departure_station"].replace("THSR", "").strip()
                    response["departure_station"] = response["departure_station"].replace("HSR", "").strip()
                    response["departure_station"] = response["departure_station"].replace("Station", "").strip()
                if "arrival_station" in response:
                    response["arrival_station"] = response["arrival_station"].replace("THSR", "").strip()
                    response["arrival_station"] = response["arrival_station"].replace("HSR", "").strip()
                    response["arrival_station"] = response["arrival_station"].replace("Station", "").strip()
            
            # Convert to TicketLLM model
            return TicketLLM(
                date=response["date"],
                price=response["price"],
                departure_station=response["departure_station"],
                arrival_station=response["arrival_station"],
                serial_number=response["serial_number"],
                val_date=validation["date"],
                val_price=validation["price"],
                val_departure_station=validation["departure_station"],
                val_arrival_station=validation["arrival_station"],
                val_serial_number=validation["serial_number"]
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
    with open(r"C:\Users\ubiik-ai-vincent\Documents\tickets_0408\new.jpg", "rb") as f:
        img_b64 =  base64.b64encode(f.read()).decode('utf-8')

    # with open("test_2.txt", "r") as f:
    #     img_b64 = f.read()
    print(extract(img_b64))