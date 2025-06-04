from pydantic import BaseModel
from ollama import Client
import base64

class HSRTicket(BaseModel):
    date: str
    price: int


# Initialize the client
ollama_client = Client(host='http://192.168.73.29:11434')

# Read the image and convert it to base64
with open('extract_0305\image_16_ibon.png', 'rb') as image_file:
    img_base64 = base64.b64encode(image_file.read()).decode('utf-8')


# Define the prompt

prompt = (
    'Here is an image of an HSR ticket. Extract the following information:'
    '- date (format: yyyy/mm/dd),'
    '- price' 
)


result = ollama_client.chat(model='llama3.2-vision:latest',
    messages=[
    {
      'role': 'user',
      'content': prompt,
      'images': [img_base64],
    }],
    options={'temperature': 0.0}, format=HSRTicket.model_json_schema())
response = result.message.content
print(response)