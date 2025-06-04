import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import base64
import cv2
import json
import numpy as np
import io
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
from pydantic import BaseModel
from ollama import Client
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

def process_image(input_path, padding=5):
    """Removes background from an image without saving to disk."""
    # image = Image.open(image_path).convert("RGB")
    # image = ImageOps.exif_transpose(image)
    image = load_img(input_path, output_type="pil").convert("RGB")
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
    client = Client(host='http://192.168.73.29:11434')
    
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
    )
    
    try:
        res_1 = client.chat(
            model='gemma3:27b',
            messages=[{'role': 'user', 'content': prompt_1, 'images': [img_b64]}],
            options={'temperature': 0.0},
            format=DatePrice.model_json_schema()
        )
        
        res_1_content = res_1.message.content
        
        cropped_b64 = crop(img_b64)
        
        prompt_2 = (
            "Here is an image of a cropped HSR ticket focusing on 票號."
            "Extract the 票號 (Format: XX-X-XX-X-XXX-XXXX)"
        )
        
        res_2 = client.chat(
            model='gemma3:27b',
            messages=[{'role': 'user', 'content': prompt_2, 'images': [cropped_b64]}],
            options={'temperature': 0.0},
            format=SerialNumber.model_json_schema()
        )
        
        res_2_content = res_2.message.content
        
        # Handle empty or invalid JSON responses
        dict_1 = json.loads(res_1_content) if res_1_content else {}
        
        try:
            dict_2 = json.loads(res_2_content) if res_2_content else {"serial_number": ""}
        except json.JSONDecodeError:
            dict_2 = {"serial_number": ""}
        
        if isinstance(dict_2, dict) and "serial_number" in dict_2:
            dict_2["serial_number"] = dict_2["serial_number"].replace("-", "")
        
        final_dict = {**dict_1, **dict_2}
        return json.dumps(final_dict, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    input_path = 'ibon/images/4.jpg'
    
    # Process image in memory
    # processed_image_bytes = process_image(input_path, padding=5)
    
    # Extract information from the processed image bytes
    response = extract(input_path)
    print(response)