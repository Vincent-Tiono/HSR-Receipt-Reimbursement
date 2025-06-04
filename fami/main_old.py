from pydantic import BaseModel
from ollama import Client
import base64
import cv2
import numpy as np
import json

class DatePrice(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str

class SerialNumber(BaseModel):
    serial_number: str

def b64_to_img(b64_str: str) -> np.ndarray:
    """Convert base64 to image"""
    img_bytes = base64.b64decode(b64_str)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def crop(b64_img: str, x_sr: float = 0, y_sr: float = 0.7, x_er: float = 0.4, y_er: float = 0.88) -> str:
    """Crop base64 image and return as base64"""
    img = b64_to_img(b64_img)
    h, w = img.shape[:2]
    x_s, y_s = int(w * x_sr), int(h * y_sr)
    x_e, y_e = int(w * x_er), int(h * y_er)
    cropped = img[y_s:y_e, x_s:x_e]

    # For debugging
    cv2.imshow('Cropped Image', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    _, buf = cv2.imencode('.png', cropped)
    return base64.b64encode(buf).decode('utf-8')

def extract(image_path: str) -> str:
    """Extract ticket information"""
    client = Client(host='http://192.168.73.29:11434')
    
    # Read image and convert to base64
    with open(image_path, 'rb') as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode('utf-8')

    # First prompt to extract date and price
    prompt_1 = (
        'Here is an image of an HSR ticket. Extract the following information:'
        '- date (Format: yyyy/mm/dd),'
        '- price'
        '- Departure station in English (Location: right under the date),'
        '- Arrival station in English'
    )

    # Call vision model for date and price
    res_1 = client.chat(
        # model='llama3.2-vision:latest',
        model = 'gemma3:12b',
        messages=[{'role': 'user', 'content': prompt_1, 'images': [img_b64]}],
        options={'temperature': 0.0},
        format=DatePrice.model_json_schema()
    )

    res_1_content = res_1.message.content

    # Crop for serial number
    cropped_b64 = crop(img_b64)

    # Second prompt to extract serial number
    prompt_2 = (
        "Here is an image of a cropped HSR ticket focusing on 票號."
        "Extract the 票號 (Format: XX-X-XX-X-XXX-XXXX)"
    )

    # Call vision model for serial number
    res_2 = client.chat(
        model='gemma3:12b',
        messages=[{'role': 'user', 'content': prompt_2, 'images': [cropped_b64]}],
        options={'temperature': 0.0},
        format=SerialNumber.model_json_schema()
    )

    res_2_content = res_2.message.content
    
    # Combine results
    dict_1 = json.loads(res_1_content)
    dict_2 = json.loads(res_2_content)

    if isinstance(dict_2, dict) and "serial_number" in dict_2:
        dict_2["serial_number"] = dict_2["serial_number"].replace("-", "")

    final_dict = {**dict_1, **dict_2}

    return json.dumps(final_dict, ensure_ascii=False)

if __name__ == "__main__":
    img_path = 'fami\images\image_4_fami.png'
    response = extract(img_path)
    print(response)