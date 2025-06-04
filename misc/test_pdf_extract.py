from pydantic import BaseModel

class TicketLLM(BaseModel):
    """Base response model for ticket information from Gemini"""
    date: str
    price: int
    dep_station: str
    arr_station: str
    serial_number: str
    val_date: bool
    val_price: bool
    val_dep_station: bool
    val_arr_station: bool
    val_serial_number: bool

from openai import OpenAI
import json
import os
from dotenv import load_dotenv
# from ..models.ticket import TicketLLM
import base64
import tempfile

# marker imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Prepare a PdfConverter instance
converter = PdfConverter(
    artifact_dict=create_model_dict(),
)

def extract(pdf_b64: str) -> TicketLLM:
    """Extract text content from a base64-encoded PDF via Marker,
       then send it to the OpenRouter client to parse ticket info."""
    # Decode and write out to a temp file
    pdf_bytes = base64.b64decode(pdf_b64)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(pdf_bytes)

    try:
        # Run Marker's PDF converter
        rendered = converter(tmp_path)
        
        # Pull text and images out of the rendered result
        text_content, _, images = text_from_rendered(rendered)

        # Prepare the prompt for the OpenRouter client
        prompt = (
            'Here is the text content of an HSR ticket. Extract the following information and return it in JSON format:'
            f'Text content:\n{text_content}'
            '- date (First year, then month, then date, format: yyyy/mm/dd, travel date not issue date, 乘車日期 not 列印日期),'
            '- price (Format: NT$XXX),'
            '- Departure station and translate it to English,'
            '- Arrival station and translate it to English'
            '- ticket number (票號, 13-digit integer)'
            'Return the response in JSON format with keys: date, price, dep_station, arr_station, serial_number'
            'If you are unable to extract any of the required information, return {"error": "unable to process"}'
        )
        
        # Send the prompt to the OpenRouter client
        completion = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        # Get the response content
        response_text = completion.choices[0].message.content
        
        # Try to parse as JSON
        try:
            # If the response is wrapped in markdown code blocks, extract the JSON
            if response_text.startswith('```json'):
                json_text = response_text.split('```json')[1].split('```')[0].strip()
                response = json.loads(json_text)
            else:
                response = json.loads(response_text)
            
            # Check if response contains error
            if isinstance(response, dict) and "error" in response:
                raise ValueError(response["error"])
            
            # Post-process the response
            if isinstance(response, dict):
                # Extract numeric price
                if "price" in response:
                    price_str = str(response["price"])
                    if "NT$" in price_str:
                        response["price"] = int(price_str.replace("NT$", "").strip())
                
                # Remove "THSR" from station names
                if "dep_station" in response:
                    response["dep_station"] = response["dep_station"].replace("THSR", "").strip()
                if "arr_station" in response:
                    response["arr_station"] = response["arr_station"].replace("THSR", "").strip()

            # Validate each field against the original text content
            val_date = str(response["date"]) in text_content
            val_price = str(response["price"]) in text_content
            val_dep_station = response["dep_station"] in text_content
            val_arr_station = response["arr_station"] in text_content
            val_serial_number = str(response["serial_number"]) in text_content

            # Convert to TicketLLM model with validation results
            return TicketLLM(
                date=response["date"],
                price=response["price"],
                dep_station=response["dep_station"],
                arr_station=response["arr_station"],
                serial_number=str(response["serial_number"]),
                val_date=val_date,
                val_price=val_price,
                val_dep_station=val_dep_station,
                val_arr_station=val_arr_station,
                val_serial_number=val_serial_number
            )
        except json.JSONDecodeError as e:
            # If the response contains error messages about being unable to process
            if "unable to process" in response_text.lower() or "not clear enough" in response_text.lower():
                raise ValueError("unable to process")
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            raise ValueError("unable to process")
    except Exception as e:
        print(f"Error in extract function: {e}")
        raise ValueError("unable to process")
    
if __name__ == "__main__":
    with open("test.txt", "r") as f:
        tmp = f.read()
    print(extract(tmp))