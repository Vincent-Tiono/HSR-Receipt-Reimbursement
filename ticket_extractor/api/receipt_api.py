from fastapi import APIRouter, HTTPException
from ..core.receipt_extractor import extract
from ..models.receipt import ReceiptOutput
from .main import ImageRequest, ErrorResponse, ERROR_MESSAGES

router = APIRouter()

@router.post("/extract/receipt", 
    response_model=ReceiptOutput, 
    responses={
        400: {"model": ErrorResponse, "description": ERROR_MESSAGES["invalid_format"]},
        422: {"model": ErrorResponse, "description": ERROR_MESSAGES["invalid_ticket"]},
        500: {"model": ErrorResponse, "description": ERROR_MESSAGES["server_error"]}
    }
)
async def extract_receipt(data: ImageRequest):
    """
    Extract information from a receipt using base64-encoded image.
    
    Args:
        data: ImageRequest containing base64-encoded image
        
    Returns:
        ReceiptOutput: Extracted receipt information with file name
        
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
        ReceiptOutput: Extracted information with file name
        
    Raises:
        HTTPException: With appropriate status code and error message
    """
    try:
        # Validate file format (file extension)
        # process_type = data.validate_format()
        
        # Process receipt using receipt_extractor
        receipt_result = extract(data.b64)
        
        # Return ReceiptOutput with the receipt data
        # return ReceiptOutput(
        #     file_name=data.file_name,
        #     invoice_date=receipt_result.invoice_date,
        #     invoice_number=receipt_result.invoice_number,
        #     seller_id=receipt_result.seller_id,
        #     total_amount=receipt_result.total_amount,
        #     val_invoice_date=receipt_result.val_invoice_date,
        #     val_invoice_number=receipt_result.val_invoice_number,
        #     val_seller_id=receipt_result.val_seller_id,
        #     val_total_amount=receipt_result.val_total_amount
        # )
        return ReceiptOutput(
            file_name={"value": data.file_name, "ocr_val": True},
            invoice_date={"value": receipt_result.invoice_date, "ocr_val": receipt_result.val_invoice_date},
            invoice_number={"value": receipt_result.invoice_number, "ocr_val": receipt_result.val_invoice_number},
            seller_id={"value": receipt_result.seller_id, "ocr_val": receipt_result.val_seller_id},
            total_amount={"value": str(receipt_result.total_amount), "ocr_val": receipt_result.val_total_amount}
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