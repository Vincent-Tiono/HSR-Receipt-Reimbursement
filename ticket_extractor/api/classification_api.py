from fastapi import APIRouter, HTTPException
import sys
import os

# Add the root directory to the path to allow importing classifier module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from ..core.classifier import classify_image

from .main import ClassificationRequest, ClassificationResponse, ErrorResponse, ERROR_MESSAGES

router = APIRouter()

@router.post("/classify", 
    response_model=ClassificationResponse, 
    responses={
        400: {"model": ErrorResponse, "description": ERROR_MESSAGES["invalid_format"]},
        422: {"model": ErrorResponse, "description": "Invalid or unprocessable image"},
        500: {"model": ErrorResponse, "description": ERROR_MESSAGES["server_error"]}
    }
)
async def classify(data: ClassificationRequest):
    """
    Classify an image as ticket, receipt, or invalid image.
    
    Args:
        data: ClassificationRequest containing base64-encoded image
        
    Returns:
        ClassificationResponse: Classification result
        
    Raises:
        HTTPException: With appropriate status code and error message
        - 422 Unprocessable Entity if the image is invalid
        - 500 Internal Server Error for other errors
    """
    try:
        # Call the classify_image function from classifier.py
        result = classify_image(data.b64)
        
        # Return 422 if the image is classified as invalid
        if result == "invalid image":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_image"
                }
            )
        
        # Return the classification result
        return ClassificationResponse(
            classification=result
        )
            
    except HTTPException:
        # Re-raise HTTP exceptions (like our 422)
        raise
    except Exception as e:
        # Use 500 for server errors
        raise HTTPException(
            status_code=500,
            detail={
                "error": "classification_error",
                "message": ERROR_MESSAGES["classification_error"]
            }
        )