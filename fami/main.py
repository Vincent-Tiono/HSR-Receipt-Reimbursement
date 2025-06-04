from pydantic import BaseModel
from ollama import Client
import base64
import cv2
import numpy as np
import os
import json
import io
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image

# Set device and precision
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(DEVICE)

# Define image transformation
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

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
        im = load_img(input_path, output_type="pil")
    except Exception as e:
        im = Image.open(input_path)
    
    im = im.convert("RGB")
    transparent = process(im)

    # Create output directory if not exists
    output_dir = "fami/outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Save removed-background image
    removed_bg_path = os.path.join(output_dir, "removed_bg.png")
    transparent.save(removed_bg_path)

    cropped = crop_transparent(transparent, padding=padding)

    # Save cropped image
    cropped_path = os.path.join(output_dir, "cropped.png")
    cropped.save(cropped_path)

    # Convert to bytes in memory
    img_byte_arr = io.BytesIO()
    cropped.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

class DatePrice(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str

class SerialNumber(BaseModel):
    serial_number: str

def crop_for_serial(img_bytes, x_sr: float = 0, y_sr: float = 0.7, x_er: float = 1, y_er: float = 0.88) -> str:
    """Crop an image from bytes and return cropped image as base64"""
    # Convert image bytes to numpy array for OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    x_s, y_s = int(w * x_sr), int(h * y_sr)
    x_e, y_e = int(h * x_er), int(h * y_er)
    cropped = img[y_s:y_e, x_s:x_e]
    
    _, buf = cv2.imencode('.png', cropped)
    return base64.b64encode(buf).decode('utf-8')

def extract(img_path: str) -> str:
    """Process the ticket image and extract relevant information"""
    client = Client(host='http://192.168.73.29:11434')

    # Process image in memory - remove background and crop
    processed_img_bytes = process_image(img_path, padding=10)
    
    # Convert processed image bytes to base64 for API
    img_b64 = base64.b64encode(processed_img_bytes).decode('utf-8')

    # First prompt to extract date and price
    prompt_1 = (
        'Here is an image of an HSR ticket. Extract the following information:'
        '- date (Format: YYYY/MM/DD),'
        '- price'
        '- Departure station in English (Location: right under the date),'
        '- Arrival station in English'
    )

    # First Ollama Vision call for date and price
    res_1 = client.chat(
        model='gemma3:27b',
        messages=[{'role': 'user', 'content': prompt_1, 'images': [img_b64]}],
        options={'temperature': 0.0},
        format=DatePrice.model_json_schema()
    )

    # Process response for date and price
    res_1_content = res_1.message.content

    # Crop image for serial number extraction - directly using processed image bytes
    cropped_b64 = crop_for_serial(processed_img_bytes)

    # Second prompt to extract serial number
    prompt_2 = (
        "Here is an image of a cropped HSR ticket focusing on 票號."
        "Extract the 票號 (Format: XX-X-XX-X-XXX-XXXX)"
    )

    # Second Ollama Vision call for serial number
    res_2 = client.chat(
        model='gemma3:27b',
        messages=[{'role': 'user', 'content': prompt_2, 'images': [cropped_b64]}],
        options={'temperature': 0.0},
        format=SerialNumber.model_json_schema()
    )

    # Process response for serial number
    res_2_content = res_2.message.content

    # Merge responses
    dict_1 = json.loads(res_1_content)
    dict_2 = json.loads(res_2_content)
    
    if isinstance(dict_2, dict) and "serial_number" in dict_2:
        dict_2["serial_number"] = dict_2["serial_number"].replace("-", "")

    final_dict = {**dict_1, **dict_2}

    return json.dumps(final_dict, ensure_ascii=False)

if __name__ == "__main__":
    img_path = 'fami\images\\2.jpg'
    response = extract(img_path)
    print(response)