from pydantic import BaseModel
from typing import Dict, Any

class ReceiptLLM(BaseModel):
    """Base response model for ticket information from Gemini"""
    invoice_date: str
    invoice_number: str
    seller_id: str
    total_amount: int
    val_invoice_date: bool
    val_invoice_number: bool
    val_seller_id: bool
    val_total_amount: bool

# class ReceiptOutput(BaseModel):
#     """Full response model including file name"""
#     file_name: str
#     invoice_date: str
#     invoice_number: str
#     seller_id: str
#     total_amount: int
#     val_invoice_date: bool
#     val_invoice_number: bool
#     val_seller_id: bool
#     val_total_amount: bool

class ReceiptOutput(BaseModel):
    """Full response model including file name"""
    file_name: Dict[str, Any]
    invoice_date: Dict[str, Any]
    invoice_number: Dict[str, Any]
    seller_id: Dict[str, Any]
    total_amount: Dict[str, Any]
