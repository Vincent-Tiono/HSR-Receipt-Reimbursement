from pydantic import BaseModel

class HSRTicket(BaseModel):
    file_name: str
    date: str
    price: int
    dep_station: str
    arr_station: str
    serial_number: str 