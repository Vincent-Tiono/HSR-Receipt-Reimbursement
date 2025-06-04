from pydantic import BaseModel
from ollama import Client
import base64
import json
import os
from dotenv import load_dotenv

client = Client(host='http://192.168.73.29:11434')

# Initialize OpenAI client
load_dotenv()
api_key=os.getenv("OPENROUTER_API_KEY")

class HSRTicket(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str
    serial_number: str = Field(
        ..., 
        regex=r"^\d{2}-\d-\d{2}-\d-\d{3}-\d{4}$",
        description="Serial number must follow the format XX-X-XX-X-XXX-XXXX")

def extract(img_path: str) -> str:
    """Send the image to the Ollama client and extract ticket information."""
    with open(img_path, 'rb') as image_file:
        img_b64 = base64.b64encode(image_file.read()).decode('utf-8')
    
    prompt = (
        'Here is an image of an HSR ticket. Extract the following information:'
        '- date (Format: yyyy/mm/dd),'
        '- price (Format: NT$XXX),'
        '- Departure station in English,'
        '- Arrival station in English'
        '- 票號 (ie. 13-digit serial number, Format: XX-X-XX-X-XXX-XXXX)'
    )
    
    result = client.chat(
        model='gemma3:27b',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [img_b64],
            }
        ],
        options={'temperature': 0.0}, 
        format=HSRTicket.model_json_schema()
    )
    
    # Convert JSON string to dictionary
    response = json.loads(result.message.content)

    # Ensure serial_number has no hyphens
    if isinstance(response, dict) and "serial_number" in response:
        response["serial_number"] = response["serial_number"].replace("-", "")

    # Convert back to JSON string for consistent output format
    return json.dumps(response, ensure_ascii=False)


if __name__ == "__main__":
    img_path = "fami\images\image_2_fami.png"
    response = extract(img_path)
    print(response)