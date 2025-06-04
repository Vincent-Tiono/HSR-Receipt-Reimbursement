from pydantic import BaseModel
from openai import OpenAI
import base64
import json
import numpy as np
import io
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set device and precision
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the background removal model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(DEVICE)

# Define image transformation
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class HSRTicket(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str
    serial_number: str

def process(image):
    """Process an image to remove the background."""
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(DEVICE)
    
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    
    image.putalpha(mask)
    return image

def crop_transparent(img, padding=5):
    """Crop image to remove transparent areas, with optional padding."""
    if img.mode == 'RGBA':
        img_array = np.array(img)
        alpha = img_array[:, :, 3]
        
        non_transparent_pixels = np.where(alpha > 0)
        if len(non_transparent_pixels[0]) == 0:
            return img
        
        min_y, max_y = np.min(non_transparent_pixels[0]), np.max(non_transparent_pixels[0])
        min_x, max_x = np.min(non_transparent_pixels[1]), np.max(non_transparent_pixels[1])
        
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img.width, max_x + padding)
        max_y = min(img.height, max_y + padding)
        
        return img.crop((min_x, min_y, max_x, max_y))
    return img

def process_image(input_path, padding=10):
    """Process the image in memory without saving to disk."""
    try:
        # Try to use loadimg if available
        from loadimg import load_img
        im = load_img(input_path, output_type="pil")
    except Exception as e:
        # Fallback to PIL
        im = Image.open(input_path)
        im = ImageOps.exif_transpose(im)  # Handle EXIF orientation
    
    # Convert to RGB
    im = im.convert("RGB")
    
    # Remove background
    transparent = process(im)

    # Create output directory if not exists
    output_dir = "fami/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save removed-background image
    removed_bg_path = os.path.join(output_dir, "removed_bg.png")
    transparent.save(removed_bg_path)
    
    # Crop transparent areas
    cropped = crop_transparent(transparent, padding=padding)

    #Save cropped image
    cropped_path = os.path.join(output_dir, "cropped.png")
    cropped.save(cropped_path)
    
    # Convert to bytes in memory
    img_byte_arr = io.BytesIO()
    cropped.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def extract(img_path: str) -> str:
    """Send the image to the Gemini model and extract ticket information."""
    # Process image to remove background and crop
    processed_img_bytes = process_image(img_path, padding=10)
    
    # Convert processed image bytes to base64
    img_b64 = base64.b64encode(processed_img_bytes).decode('utf-8')
    
    prompt = (
        'Here is an image of an HSR ticket. Extract the following information and return it in JSON format:'
        '- date (format: yyyy/mm/dd),'
        '- price,'
        '- Departure station in English,'
        '- Arrival station in English,'
        '- 票號 (13-digit strict format: XX-X-XX-X-XXX-XXXX),'
        'Return the response in this exact JSON format: {"date": "yyyy/mm/dd", "price": number, "dep_station": "station name", "arr_station": "station name", "serial_number": "XX-X-XX-X-XXX-XXXX"}'
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
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    try:
        # Get the response content
        response_text = completion.choices[0].message.content
        
        # Remove markdown code block markers if present
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()
        
        # Try to parse as JSON
        response = json.loads(response_text)
        
        # Ensure serial_number has no hyphens
        if isinstance(response, dict) and "serial_number" in response:
            response["serial_number"] = response["serial_number"].replace("-", "")
        
        # Convert back to JSON string for consistent output format
        return json.dumps(response, ensure_ascii=False)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {response_text}")
        raise

if __name__ == "__main__":
    img_path = os.path.join("images", "5.jpg")
    response = extract(img_path)
    print(response) 