import base64
import cv2
from pydantic import BaseModel
from ollama import Client

def crop_ticket_image(image_path: str, x_start_ratio: float = 0.6, y_start_ratio: float = 0.5, y_end_ratio: float = 0.85) -> str:
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image from {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    x_start, y_start = int(width * x_start_ratio), int(height * y_start_ratio)
    x_end, y_end = width, int(height * y_end_ratio)
    
    # Crop the image
    cropped_img = img[y_start:y_end, x_start:x_end]
    
    # Convert to base64
    _, buffer = cv2.imencode('.png', cropped_img)
    cropped_img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return cropped_img_base64

# Define the Pydantic model for the ticket number only
class HSRTicket(BaseModel):
    serial_number: str

# Initialize the Ollama client
ollama_client = Client(host='http://192.168.73.29:11434')

# Image path
image_path = 'extract_0305\ibon\image_10_ibon.PNG'

# Crop the image and get base64
cropped_img_base64 = crop_ticket_image(image_path)

# Define the prompt for the cropped image only
prompt = (
    "Here is an image of a cropped HSR ticket focusing on 票號."
    "Extract the 票號 (Format: XX-X-XX-X-XXX-XXXX)"
)

# Call the Ollama client with the cropped image
result = ollama_client.chat(
    model='llama3.2-vision:latest',
    messages=[
        {
            'role': 'user',
            'content': prompt,
            'images': [cropped_img_base64],
        }
    ],
    options={'temperature': 0.0},
    format=HSRTicket.model_json_schema()
)

# Print the response
response = result# Define the prompt for the cropped image only
prompt = (
    "Here is an image of a cropped HSR ticket focusing on 票號."
    "Extract the 票號 (Format: XX-X-XX-X-XXX-XXXX)"
)

# Call the Ollama client with the cropped image
result = ollama_client.chat(
    model='llama3.2-vision:latest',
    messages=[
        {
            'role': 'user',
            'content': prompt,
            'images': [cropped_img_base64],
        }
    ],
    options={'temperature': 0.0},
    format=HSRTicket.model_json_schema()
)

response = result['message']['content']
print(response)
