from pydantic import BaseModel
from ollama import Client
import base64
import cv2
import numpy as np
import json
from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image, ImageOps

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

# def rotate_to_straight(image):
#     """Automatically detect and correct image tilt."""
#     image = ImageOps.exif_transpose(image)  # Correct orientation using EXIF data
#     return image.rotate(-2, expand=True)  # Slight rotation correction

def process(image):
    """Process an image to remove the background."""
    # image = rotate_to_straight(image)
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

def process_file(input_path, output_path):
    """Process a file from input path and save to output path."""
    # print(f"Processing image: {input_path}")
    try:
        im = load_img(input_path, output_type="pil")
    except Exception as e:
        # print(f"Error loading image with load_img: {e}")
        im = Image.open(input_path)
    
    im = im.convert("RGB")
    transparent = process(im)
    cropped = crop_transparent(transparent, padding=10)
    
    # Option 1: Change the output file extension to PNG
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        output_path = output_path.replace('.jpg', '.png').replace('.jpeg', '.png')
    
    # Option 2: Or convert to RGB if you want to keep JPEG
    # cropped = cropped.convert('RGB')
    
    cropped.save(output_path)
    # print(f"Successfully saved cropped image to: {output_path}")
    return output_path

class DatePrice(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str

class SerialNumber(BaseModel):
    serial_number: str

def b64_to_img(b64_str: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def crop(b64_img: str, x_sr: float = 0.6, y_sr: float = 0.5, y_er: float = 0.85) -> str:
    """Crop an image from base64 and return cropped image as base64"""
    img = b64_to_img(b64_img)
    h, w = img.shape[:2]
    x_s, y_s = int(w * x_sr), int(h * y_sr)
    x_e, y_e = w, int(h * y_er)
    cropped = img[y_s:y_e, x_s:x_e]
    
    _, buf = cv2.imencode('.png', cropped)
    return base64.b64encode(buf).decode('utf-8')

def extract(img_path: str) -> str:
    """Process the ticket image and extract relevant information"""
    client = Client(host='http://192.168.73.29:11434')

    processed_img_path = 'ibon\images\\processed.png'
    process_file(img_path, processed_img_path)
    
    # Read and encode the image to base64
    with open(processed_img_path, 'rb') as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode('utf-8')

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

    # Crop image for serial number extraction
    cropped_b64 = crop(img_b64)

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
    img_path = 'ibon\images\\4.jpg'
    response = extract(img_path)
    print(response)

