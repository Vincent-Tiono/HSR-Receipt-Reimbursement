# pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2
from transformers import pipeline
import base64
import io
from PIL import Image
from pdf2image import convert_from_bytes
import tempfile
import os

# Initialize the pipeline globally
ckpt = "google/siglip2-so400m-patch14-384"
pipe = pipeline(model=ckpt, task="zero-shot-image-classification")

# Define candidate labels globally
CANDIDATE_LABELS = [
    "a white card-style HSR ticket with orange stripe, showing train details, seat number and departure/arrival stations",
    "a yellow/green FamilyMart printed HSR ticket with THSRC logo and student discount marking",
    "a pink/red iBon printed HSR ticket with iBon logo on sides and circular stamp marking",
    "a simple receipt with with chinese characters 發票, QR codes and barcode for a store purchase, not related to train travel",
    "a white store receipt with chinese characters 發票, multiple barcodes and QR codes showing item purchase details",
    "a transaction record with 台灣高鐵 Taiwan High Speed Rail logo at top, showing travel date, fare amount and ticket number"
]

# Threshold for classification confidence
CLASSIFICATION_THRESHOLD = 0.6

def classify_image(b64_image, save_converted_image=False):
    """
    Classify a base64 encoded image or PDF as ticket, receipt, or invalid image.
    
    Args:
        b64_image: Base64 encoded image or PDF string
        save_converted_image: If True, saves the converted image when processing PDFs
        
    Returns:
        str: Classification result ("ticket", "receipt", or "invalid image")
    """
    try:
        # Decode base64 data
        image_data = base64.b64decode(b64_image)
        
        # Check if the data is a PDF by looking at the first few bytes
        if image_data.startswith(b'%PDF'):
            # print("Detected PDF file, attempting conversion...")
            # Convert PDF to images
            try:
                images = convert_from_bytes(image_data)
                # print(f"Successfully converted PDF to {len(images)} pages")
                if not images:
                    print("No images were extracted from the PDF")
                    return "invalid image"
                
                # Use the first page for classification
                image = images[0]
                # print(f"Using first page of PDF, image size: {image.size}")
                
                # Save converted image if requested
                if save_converted_image:
                    output_jpg = os.path.join(os.getcwd(), "converted_pdf_page.jpg")
                    # print(f"Attempting to save converted image to: {output_jpg}")
                    image.save(output_jpg, "JPEG")
                    # print(f"Successfully saved converted PDF page as: {output_jpg}")
                    # print(f"Absolute path: {os.path.abspath(output_jpg)}")
            except Exception as e:
                print(f"Error during PDF conversion: {str(e)}")
                return "invalid image"
        else:
            # Handle regular image
            image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            image.save(temp_path)
        
        try:
            # Perform zero-shot classification
            output = pipe(temp_path, candidate_labels=CANDIDATE_LABELS)
            
            # Find the label with the highest score
            max_score_result = max(output, key=lambda x: x['score'])
            classified_class = max_score_result['label']
            
            # Check if the highest score meets the threshold
            if max_score_result['score'] < CLASSIFICATION_THRESHOLD:
                return "invalid image"
            else:
                if classified_class in [
                    "a white card-style HSR ticket with orange stripe, showing train details, seat number and departure/arrival stations",
                    "a yellow/green FamilyMart printed HSR ticket with THSRC logo and student discount marking",
                    "a pink/red iBon printed HSR ticket with iBon logo on sides and circular stamp marking",
                    "a transaction record with 台灣高鐵 Taiwan High Speed Rail logo at top, showing travel date, fare amount and ticket number"
                ]:
                    return "ticket"
                elif classified_class in [
                    "a simple receipt with with chinese characters 發票, QR codes and barcode for a store purchase, not related to train travel",
                    "a white store receipt with chinese characters 發票, multiple barcodes and QR codes showing item purchase details"
                ]:
                    return "receipt"
                else:
                    return "invalid image"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    except Exception as e:
        # Return "invalid image" for any processing errors
        return "invalid image"

# Example usage
if __name__ == "__main__":
    # Read base64 content from text file
    with open("tmp2.txt", "r") as f:
        b64_string = f.read().strip()
    
    # Classify the image/PDF
    result = classify_image(b64_string, save_converted_image=True)
    print(f"Classification result: {result}")