from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables at startup
load_dotenv()

# Verify OpenRouter API key is set
if not os.getenv("OPENROUTER_API_KEY"):
    raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

app = FastAPI(
    title="HSR Ticket Extractor API",
    description="API for extracting information from HSR tickets and receipts",
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

# Error messages
ERROR_MESSAGES = {
    "invalid_format": "Invalid input or unsupported format",
    "invalid_ticket": "The provided file does not appear to be a valid HSR ticket or the ticket format is not recognized",
    "processing_error": "An error occurred while processing the ticket",
    "server_error": "An unexpected error occurred while processing your request. Please try again later",
    "classification_error": "An error occurred while classifying the image"
}

class ImageRequest(BaseModel):
    file_name: str
    file_ext: str  # File extension with dot (e.g., '.pdf', '.png', '.jpg')
    b64: str  # Base64-encoded image string

    def validate_format(self):
        """Validate that the file type is supported."""
        file_ext = self.file_ext.lower()
        
        if file_ext in SUPPORTED_IMAGE_FORMATS:
            return 'image'
        elif file_ext in SUPPORTED_PDF_FORMATS:
            return 'pdf'
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_format",
                    "message": ERROR_MESSAGES["invalid_format"]
                }
            )

class ErrorResponse(BaseModel):
    error: str
    message: str

class ClassificationRequest(BaseModel):
    b64: str  # Base64-encoded image string

class ClassificationResponse(BaseModel):
    classification: str  # "ticket", "receipt", or "invalid image"

# Import and include routers
from .ticket_api import router as ticket_router
from .receipt_api import router as receipt_router
from .trad_receipt_api import router as trad_receipt_router
from .classification_api import router as classification_router

app.include_router(ticket_router)
app.include_router(receipt_router)
app.include_router(trad_receipt_router)
app.include_router(classification_router) 