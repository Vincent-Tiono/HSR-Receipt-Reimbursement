from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from ..core.image_extractor import extract as extract_image
from ..core.pdf_extractor import extract as extract_pdf
from ..models.ticket import HSRTicket

app = FastAPI(
    title="HSR Ticket Extractor API",
    description="API for extracting information from HSR tickets in image or PDF format",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supported file formats
SUPPORTED_IMAGE_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.heic', '.heif'}
SUPPORTED_PDF_FORMATS = {'.pdf'}

class ImageRequest(BaseModel):
    file_name: str
    file_type: str  # File extension with dot (e.g., '.pdf', '.png', '.jpg')
    b64: str  # Base64-encoded image string

    def validate_format(self):
        """Validate that the file type is supported."""
        file_ext = self.file_type.lower()
        
        if file_ext in SUPPORTED_IMAGE_FORMATS:
            return 'image'
        elif file_ext in SUPPORTED_PDF_FORMATS:
            return 'pdf'
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS | SUPPORTED_PDF_FORMATS)}"
            )

@app.post("/extract", response_model=HSRTicket)
async def extract_ticket(data: ImageRequest):
    """
    Extract information from an HSR ticket using base64-encoded image or PDF.
    
    Args:
        data: ImageRequest containing base64-encoded image and file type
        
    Returns:
        HSRTicket: Extracted ticket information
    """
    try:
        # Validate file format and get processing type
        process_type = data.validate_format()
        
        # Process based on file type
        if process_type == 'image':
            result = extract_image(data.b64)
        else:  # pdf
            result = extract_pdf(data.b64)
        
        # Create a new HSRTicket with the file_name
        return HSRTicket(
            file_name=data.file_name,
            date=result.date,
            price=result.price,
            dep_station=result.dep_station,
            arr_station=result.arr_station,
            serial_number=result.serial_number
        )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 