from fastapi import APIRouter, HTTPException
from ..core.image_extractor import extract as extract_image
from ..core.pdf_extractor import extract as extract_pdf
from ..models.ticket import TicketOutput
from .main import ImageRequest, ErrorResponse, ERROR_MESSAGES

router = APIRouter()

@router.post("/extract/ticket", 
    response_model=TicketOutput, 
    responses={
        400: {"model": ErrorResponse, "description": ERROR_MESSAGES["invalid_format"]},
        422: {"model": ErrorResponse, "description": ERROR_MESSAGES["invalid_ticket"]},
        500: {"model": ErrorResponse, "description": ERROR_MESSAGES["server_error"]}
    }
)
async def extract_ticket(data: ImageRequest):
    """
    Extract information from an HSR ticket using base64-encoded image or PDF.
    
    Args:
        data: ImageRequest containing base64-encoded image
        
    Returns:
        TicketOutput: Extracted ticket information with file name
        
    Raises:
        HTTPException: With appropriate status code and error message
    """
    return await process_extraction(data)

async def process_extraction(data: ImageRequest):
    """
    Process extraction based on file extension.
    
    Args:
        data: ImageRequest containing base64-encoded image
        
    Returns:
        TicketOutput: Extracted information with file name
        
    Raises:
        HTTPException: With appropriate status code and error message
    """
    try:
        # Validate file format (file extension)
        process_type = data.validate_format()
        
        # Process based on file_type (ticket or receipt) and file_ext (image or pdf)
        if process_type == 'image':
            result = extract_image(data.b64)
        else:  # pdf
            result = extract_pdf(data.b64)
        
        # Create a new TicketOutput by combining the file_name with the TicketLLM
        # return TicketOutput(
        #     file_name=data.file_name,
        #     date=result.date,
        #     price=result.price,
        #     departure_station=result.departure_station,
        #     arrival_station=result.arrival_station,
        #     serial_number=result.serial_number,
        #     val_date=result.val_date,
        #     val_price=result.val_price,
        #     val_departure_station=result.val_departure_station,
        #     val_arrival_station=result.val_arrival_station,
        #     val_serial_number=result.val_serial_number
        # )
        return TicketOutput(
                file_name={"value": data.file_name, "ocr_val": True},
                date={"value": result.date, "ocr_val": result.val_date},
                price={"value": str(result.price), "ocr_val": result.val_price},
                departure_station={"value": result.departure_station, "ocr_val": result.val_departure_station},
                arrival_station={"value": result.arrival_station, "ocr_val": result.val_arrival_station},
                serial_number={"value": result.serial_number, "ocr_val": result.val_serial_number}
        )
            
    except ValueError as e:
        if str(e) == "unable to process":
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_ticket",
                    "message": ERROR_MESSAGES["invalid_ticket"]
                }
            )
        raise HTTPException(
            status_code=422,
            detail={
                "error": "processing_error",
                "message": ERROR_MESSAGES["processing_error"]
            }
        )
    except Exception as e:
        # Only use 500 for actual server errors
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": ERROR_MESSAGES["server_error"]
            }
        ) 