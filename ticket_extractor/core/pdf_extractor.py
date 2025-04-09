from pydantic import BaseModel
from openai import OpenAI
import json
from markitdown import MarkItDown
import os
from dotenv import load_dotenv
from ..models.ticket import HSRTicket, HSRTicketRaw
import base64
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def extract(pdf_b64: str) -> HSRTicket:
    """Extract text content from a base64-encoded PDF, send it to the OpenRouter client, and extract ticket information."""
    try:
        # Decode base64 to bytes
        pdf_bytes = base64.b64decode(pdf_b64)
        
        # Create BytesIO object for in-memory PDF processing
        pdf_stream = BytesIO(pdf_bytes)
        
        # Extract text content from the PDF
        md_converter = MarkItDown()
        result = md_converter.convert(pdf_stream)
        text_content = result.text_content

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
            response_format=HSRTicketRaw
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
                
                # Ensure serial_number has no hyphens
                if "serial_number" in response:
                    response["serial_number"] = str(response["serial_number"]).replace("-", "")

            # Convert to HSRTicket model
            return HSRTicket(
                file_name=os.path.basename(pdf_b64),
                date=response["date"],
                price=response["price"],
                dep_station=response["dep_station"],
                arr_station=response["arr_station"],
                serial_number=response["serial_number"]
            )
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract information from markdown format
            try:
                # Initialize response dictionary
                response = {}
                
                # Split the response into lines
                lines = response_text.split('\n')
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('*   **'):
                        # Extract key and value
                        key_value = line[6:-2].split(':** ')
                        if len(key_value) == 2:
                            key, value = key_value
                            # Convert key to match expected format
                            key = key.lower().replace(' ', '_')
                            if key == 'ticket_number':
                                key = 'serial_number'
                            elif key == 'departure_station':
                                key = 'dep_station'
                            elif key == 'arrival_station':
                                key = 'arr_station'
                            response[key] = value
                
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
                    
                    # Ensure serial_number has no hyphens
                    if "serial_number" in response:
                        response["serial_number"] = response["serial_number"].replace("-", "")
                
                # Convert to HSRTicket model
                return HSRTicket(
                    file_name=os.path.basename(pdf_b64),
                    date=response["date"],
                    price=response["price"],
                    dep_station=response["dep_station"],
                    arr_station=response["arr_station"],
                    serial_number=response["serial_number"]
                )
            except Exception as e:
                print(f"Error processing markdown response: {e}")
                print(f"Raw response: {response_text}")
                raise
    except Exception as e:
        print(f"Error in extract function: {e}")
        print(f"Raw response: {response_text}")
        raise 