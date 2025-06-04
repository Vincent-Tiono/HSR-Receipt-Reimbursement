# HSR Ticket Extractor API

- A FastAPI service that extracts information from Taiwan HSR tickets and receipts (發票)
- OCR validation of LLM results to mitigate hallucination issues

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## API Endpoints

Supported Formats
- Images: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`, `.heic`, `.heif`
- Documents: `.pdf`

### 1. Classify Document
`POST /classify`
```json
{
    "file_name": "document.png",
    "file_type": ".png|.pdf|etc",
    "b64": "base64_encoded_string"
}
```
Response: `{"classfication": "ticket|receipt|trad-receipt"}`

### 2. Extract Ticket
`POST /extract-ticket`
```json
{
    "file_name": "ticket.png",
    "file_type": ".png",
    "b64": "base64_encoded_string"
}
```
Response:
```json
{
    "file_name": {"value": "ticket.png", "validated": true},
    "date": {"value": "2024/04/08", "validated": true},
    "price": {"value": 1500, "validated": true},
    "departure_station": {"value": "Taipei", "validated": true},
    "arrival_station": {"value": "Kaohsiung", "validated": true},
    "serial_number": {"value": "1234567890123", "validated": true}
}
```

### 3. Extract Receipt
`POST /extract-receipt`
```json
{
    "file_name": "receipt.png",
    "file_type": ".png",
    "b64": "base64_encoded_string"
}
```
Response:
```json
{
    "file_name": {"value": "1", "validated": true},
    "invoice_date": {"value": "2025-04-18", "validated": true},
    "invoice_number": {"value": "ML-78825797", "validated": true},
    "seller_id": {"value": "00582797", "validated": true},
    "total_amount": {"value": 145, "validated": true}
}
```

### 4. Extract Traditional Receipt
`POST /extract-trad-receipt`
```json
{
    "file_name": "receipt.png",
    "file_type": ".png",
    "b64": "base64_encoded_string"
}
```
Response:
```json
{
    "file_name": {"value": "trad_1", "validated": true},
    "invoice_date": {"value": "2016-02-10", "validated": true},
    "invoice_number": {"value": "BD-52208417", "validated": true},
    "seller_id": {"value": "80333064", "validated": true},
    "total_amount": {"value": 1630, "validated": true}
}
```

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key for Gemini model access
- `GOOGLE_APPLICATION_CREDENTIALS`: Required for OCR validation services

## Model Configuration

- Default model: `google/gemini-2.0-flash-lite-001`.
- Available models: [OpenRouter documentation](https://openrouter.ai/docs)

## Running the API

```
uvicorn main:app --reload
```
Starts the server on port 8000