import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import base64
import cv2
import json
import numpy as np
import io
import requests
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
api_key=os.getenv("OPENROUTER_API_KEY")

# Set device and precision
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the segmentation model
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

def process_image(image, padding=5):
    """Removes background from an image without saving to disk."""
    image = load_img(image, output_type="pil").convert("RGB")
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(DEVICE)
    
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred).resize(image_size)
    
    image.putalpha(pred_pil)
    
    # Crop transparent areas
    img_array = np.array(image)
    alpha = img_array[:, :, 3]
    non_transparent_pixels = np.where(alpha > 0)
    
    if len(non_transparent_pixels[0]) > 0:
        min_y, max_y = np.min(non_transparent_pixels[0]), np.max(non_transparent_pixels[0])
        min_x, max_x = np.min(non_transparent_pixels[1]), np.max(non_transparent_pixels[1])
        
        # Apply padding
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.width, max_x + padding)
        max_y = min(image.height, max_y + padding)
        
        image = image.crop((min_x, min_y, max_x, max_y))

    # Show the cropped image
    bg = Image.new('RGB', image.size, (255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    plt.figure(figsize=(10, 8))
    plt.imshow(bg)
    plt.title("Background removed and cropped")
    plt.axis('off')
    plt.show()
    
    # Convert PIL image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr

class DatePrice(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str

class SerialNumber(BaseModel):
    serial_number: str

def b64_to_img(b64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def crop(b64_img: str, x_sr: float = 0.6, y_sr: float = 0.5, y_er: float = 0.8) -> str:
    img = b64_to_img(b64_img)
    h, w = img.shape[:2]
    cropped = img[int(h * y_sr):int(h * y_er), int(w * x_sr):w]
    _, buf = cv2.imencode('.png', cropped)
    return base64.b64encode(buf).decode('utf-8')

def extract(image_input):
    """Extract information from an image provided as bytes or path."""
    # Handle both string paths and bytes
    if isinstance(image_input, str):
        # Input is a path string, process the image
        image_bytes = process_image(image_input, padding=5)
    else:
        # Input is already bytes
        image_bytes = image_input
    
    # Convert image bytes to base64
    img_b64 = base64.b64encode(image_bytes).decode('utf-8')
    
    prompt_1 = (
        'Here is an image of an HSR ticket. Extract the following information:'
        '- date (Year, month and date in format: YYYY/MM/DD),'
        '- price'
        '- Departure station in English (Location: right under the date),'
        '- Arrival station in English'
        '\n\nRespond with ONLY a JSON object with keys: date, price, dep_station, arr_station'
    )
    
    try:
        # First API call using OpenRouter with Claude 3 Haiku (which supports vision)
        print("Sending first request to OpenRouter API with Claude 3 Haiku...")
        res_1 = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app-domain.com", 
                "X-Title": "HSR Ticket Extractor",
            },
            json={
                "model": "anthropic/claude-3-haiku@20240307",  # Use Claude 3 Haiku which supports images
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_1
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
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
        )
        
        # Print raw response for debugging
        print("First API Response Status:", res_1.status_code)
        print("First API Raw Response:", res_1.text[:500] + "..." if len(res_1.text) > 500 else res_1.text)
        
        res_1_json = res_1.json()
        res_1_content = res_1_json.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        print("First extracted content:", res_1_content)
        
        # Crop the image for the second API call
        cropped_b64 = crop(img_b64)
        
        prompt_2 = (
            "Here is an image of a cropped HSR ticket focusing on 票號 (serial number)."
            "Extract the 票號 (Format: XX-X-XX-X-XXX-XXXX)"
            "\n\nRespond with ONLY a JSON object with a single key: serial_number"
        )
        
        # Second API call using OpenRouter with Claude 3 Haiku
        print("Sending second request to OpenRouter API with Claude 3 Haiku...")
        res_2 = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://your-app-domain.com",
                "X-Title": "HSR Ticket Extractor",
            },
            json={
                "model": "anthropic/claude-3-haiku@20240307",  # Use Claude 3 Haiku which supports images
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt_2
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{cropped_b64}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            }
        )
        
        # Print raw response for debugging
        print("Second API Response Status:", res_2.status_code)
        print("Second API Raw Response:", res_2.text[:500] + "..." if len(res_2.text) > 500 else res_2.text)
        
        res_2_json = res_2.json()
        res_2_content = res_2_json.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        print("Second extracted content:", res_2_content)
        
        # Handle JSON responses with better error handling
        try:
            dict_1 = json.loads(res_1_content) if res_1_content and res_1_content != '{}' else {"date": "", "price": 0, "dep_station": "", "arr_station": ""}
        except json.JSONDecodeError as e:
            print(f"Error parsing first JSON response: {e}")
            dict_1 = {"date": "", "price": 0, "dep_station": "", "arr_station": ""}
        
        try:
            dict_2 = json.loads(res_2_content) if res_2_content and res_2_content != '{}' else {"serial_number": ""}
        except json.JSONDecodeError as e:
            print(f"Error parsing second JSON response: {e}")
            dict_2 = {"serial_number": ""}
        
        if isinstance(dict_2, dict) and "serial_number" in dict_2:
            dict_2["serial_number"] = dict_2["serial_number"].replace("-", "")
        
        final_dict = {**dict_1, **dict_2}
        return json.dumps(final_dict, ensure_ascii=False)
    
    except Exception as e:
        print(f"Exception in extract function: {str(e)}")
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    input_path = 'ibon/images/6.jpg'
    
    # Extract information from the image
    response = extract(input_path)
    print("Final extracted information:")
    print(response)