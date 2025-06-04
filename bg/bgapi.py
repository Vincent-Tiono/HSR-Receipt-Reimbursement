# from fastapi import FastAPI
# from fastapi.responses import Response
# from pydantic import BaseModel
# import io
# import base64
# from PIL import Image
# from main import process_image_from_pil  # Importing your processing function

# # Initialize FastAPI app
# app = FastAPI()

# class ImageRequest(BaseModel):
#     image_base64: str  # Expecting a base64-encoded string

# @app.post("/remove-bg/")
# async def remove_background(data: ImageRequest):
#     """API endpoint to remove background from an image encoded in Base64."""
#     try:
#         # Decode the base64 string
#         image_data = base64.b64decode(data.image_base64)
#         image = Image.open(io.BytesIO(image_data))

#         # Process image using external function
#         processed_image = process_image_from_pil(image)

#         # Convert processed image to bytes
#         img_byte_arr = io.BytesIO()
#         processed_image.save(img_byte_arr, format="PNG")
#         img_byte_arr = img_byte_arr.getvalue()

#         # Encode back to base64
#         img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

#         return {"image_base64": img_base64}

#     except Exception as e:
#         return {"error": str(e)}
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import io
import base64
from PIL import Image
from main import process_image_from_pil  # Importing your processing function

app = FastAPI()

@app.post("/remove-bg/")
async def remove_background(request: Request):
    """API endpoint to remove background from an image.
    
    Expects a JSON payload with a key "image_base64" containing a Base64-encoded image.
    Processes the image and returns the processed image in JSON format.
    """
    try:
        # Parse the JSON body using FastAPI's request.json() method
        data = await request.json()
        
        # Decode the Base64-encoded image
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data))
        
        # Process image using the external function
        processed_image = process_image_from_pil(image)
        
        # Convert the processed image to bytes and re-encode to Base64
        img_byte_arr = io.BytesIO()
        processed_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()
        img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
        
        # Return the processed image as JSON
        return JSONResponse(content={"image": img_base64})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)})