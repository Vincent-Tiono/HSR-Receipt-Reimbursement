from pydantic import BaseModel
from ollama import Client
import base64
import cv2
import numpy as np
import json
import io
from PIL import Image

class DatePrice(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str

class SerialNumber(BaseModel):
    serial_number: str

def crop_for_serial(img_bytes, x_sr: float = 0, y_sr: float = 0.7, x_er: float = 1, y_er: float = 0.88) -> str:
    """Crop an image from bytes and return the cropped image as a base64-encoded PNG."""
    # Convert image bytes to numpy array for OpenCV
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    h, w = img.shape[:2]
    x_s, y_s = int(w * x_sr), int(h * y_sr)
    x_e, y_e = int(w * x_er), int(h * y_er)
    cropped = img[y_s:y_e, x_s:x_e]
    
    _, buf = cv2.imencode('.png', cropped)
    return base64.b64encode(buf).decode('utf-8')

def extract(image: Image.Image) -> str:
    """Extract information from an already background-removed and cropped PIL Image in RGBA mode."""
    # Ensure image is in RGBA mode
    processed_image = image.convert("RGBA")
    
    # Convert processed image to bytes in PNG format
    img_byte_arr = io.BytesIO()
    processed_image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    
    # Encode image bytes to base64 for the API
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    
    client = Client(host='http://192.168.73.29:11434')

    # First prompt to extract date, price and station information
    prompt_1 = (
        'Here is an image of an HSR ticket. Extract the following information:'
        '- date (Format: YYYY/MM/DD),'
        '- price,'
        '- Departure station in English (Location: right under the date),'
        '- Arrival station in English'
    )

    res_1 = client.chat(
        model='gemma3:27b',
        messages=[{'role': 'user', 'content': prompt_1, 'images': [img_b64]}],
        options={'temperature': 0.0},
        format=DatePrice.model_json_schema()
    )
    res_1_content = res_1.message.content

    # Crop image for serial number extraction
    cropped_b64 = crop_for_serial(img_bytes)

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

    # Merge results from both API calls
    dict_1 = json.loads(res_1_content)
    dict_2 = json.loads(res_2_content)
    if isinstance(dict_2, dict) and "serial_number" in dict_2:
        dict_2["serial_number"] = dict_2["serial_number"].replace("-", "")
    
    final_dict = {**dict_1, **dict_2}
    return json.dumps(final_dict, ensure_ascii=False)

if __name__ == "__main__":
    # Assuming the input image is already background removed and cropped in RGBA mode.
    img_path = 'fami/images/2.png'
    image = Image.open(img_path)
    result = extract(image)
    print(result)