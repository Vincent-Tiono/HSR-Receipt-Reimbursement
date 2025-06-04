from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import io
import tempfile
from PIL import Image
import os
from main import process_image  # Importing the process_image function from main

# Initialize FastAPI app
app = FastAPI()

class ImageRequest(BaseModel):
    image: str  # Expecting a base64-encoded image string

@app.post("/extract-json/")
async def extract_json(data: ImageRequest):
    """API endpoint to extract JSON from a Base64-encoded image."""
    try:
        # Decode base64 string to image bytes
        image_data = base64.b64decode(data.image)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(image_data)
            temp_file_path = temp_file.name
        
        try:
            # Process the image using the temporary file path
            extracted_data = process_image(temp_file_path)
            return JSONResponse(content=extracted_data)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)