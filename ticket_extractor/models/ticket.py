from pydantic import BaseModel
from typing import Dict, Any

class TicketLLM(BaseModel):
    """Base response model for ticket information from Gemini"""
    date: str
    price: int
    departure_station: str
    arrival_station: str
    serial_number: str
    val_date: bool
    val_price: bool
    val_departure_station: bool
    val_arrival_station: bool
    val_serial_number: bool

# class TicketOutput(BaseModel):
#     """Full response model including file name"""
#     file_name: str
#     date: str
#     price: int
#     dep_station: str
#     arr_station: str
#     serial_number: str 
#     val_date: bool
#     val_price: bool
#     val_dep_station: bool
#     val_arr_station: bool
#     val_serial_number: bool

class TicketOutput(BaseModel):
    """Full response model including file name"""
    file_name: Dict[str, Any]
    date: Dict[str, Any]
    price: Dict[str, Any]
    departure_station: Dict[str, Any]
    arrival_station: Dict[str, Any]
    serial_number: Dict[str, Any]