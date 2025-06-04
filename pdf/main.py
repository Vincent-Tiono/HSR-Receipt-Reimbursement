# pip install 'markitdown[all]~=0.1.0a1'
from pydantic import BaseModel
from ollama import Client
import json
from markitdown import MarkItDown

client = Client(
    host='http://192.168.73.29:11434',
)

class HSRTicket(BaseModel):
    date: str
    price: int
    dep_station: str
    arr_station: str
    serial_number: str

def extract(pdf_path: str) -> str:
    """Extract text content from a PDF file, send it to the Ollama client, and extract ticket information."""
    # Extract text content from the PDF
    md_converter = MarkItDown()
    result = md_converter.convert(pdf_path)
    text_content = result.text_content

    # Prepare the prompt for the Ollama client
    prompt = (
        'Here is the text content of an HSR ticket. Extract the following information:'
         f'Text content:\n{text_content}'
        '- date (First year, then month, then date, format: yyyy/mm/dd, travel date not issue date, 乘車日期 not 列印日期),'
        '- price (Format: NT$XXX),'
        '- Departure station and translate it to English,'
        '- Arrival station and translate it to English'
        '- ticket number (票號, 13-digit integer)'
    )
    
    # Send the prompt to the Ollama client
    result = client.chat(
        model='gemma3:12b',
        messages=[
            {
                'role': 'user',
                'content': prompt,
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
    
    if isinstance(response, dict):
        if "dep_station" in response:
            response["dep_station"] = response["dep_station"].replace("Station", "").strip()
        if "arr_station" in response:
            response["arr_station"] = response["arr_station"].replace("Station", "").strip()

    # Convert back to JSON string for consistent output format
    return json.dumps(response, ensure_ascii=False)

if __name__ == "__main__":
    pdf_path = "pdf/docs/pdf_6.pdf"
    response = extract(pdf_path)
    print(response)