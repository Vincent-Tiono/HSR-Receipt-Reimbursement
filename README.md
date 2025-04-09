# HSR Ticket Extractor API

A FastAPI service that extracts information from Taiwan HSR (High Speed Rail) tickets in image or PDF format using AI.

## Features

- Extracts ticket information from images and PDFs
- Supports multiple image formats: PNG, JPEG, BMP, TIFF, WEBP, HEIC
- Processes images in memory (no disk storage)
- Uses AI for information extraction
- Configurable AI models (default: Gemini 2.0 Flash Lite)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Send a POST request to `/extract` with:
```json
{
    "file_name": "ticket.png",
    "file_type": ".png",  // or .pdf, .jpg, etc.
    "b64": "base64_encoded_string"
}
```

Response:
```json
{
    "file_name": "ticket.png",
    "date": "2024/04/08",
    "price": 1500,
    "dep_station": "Taipei",
    "arr_station": "Kaohsiung",
    "serial_number": "1234567890123"
}
```

## API Endpoints

- `POST /extract`: Extract ticket information from image/PDF

## Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key for Gemini model access

## Model Configuration

The default model is `google/gemini-2.0-flash-lite-001`. You can change the model by modifying the `model` parameter in the extractor files:
- For images: `ticket_extractor/core/image_extractor.py`
- For PDFs: `ticket_extractor/core/pdf_extractor.py`

Available models can be found in the [OpenRouter documentation](https://openrouter.ai/docs). 