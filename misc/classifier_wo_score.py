# pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2
from transformers import pipeline
import base64
import io
from PIL import Image

# Initialize the pipeline globally
ckpt = "google/siglip2-so400m-patch14-384"
pipe = pipeline(model=ckpt, task="zero-shot-image-classification")

# Define candidate labels globally
CANDIDATE_LABELS = [
    "a white card-style HSR ticket with orange stripe, showing train details, seat number and departure/arrival stations",
    "a yellow/green FamilyMart printed HSR ticket with THSRC logo and student discount marking",
    "a pink/red iBon printed HSR ticket with iBon logo on sides and circular stamp marking",
    "a simple receipt with with chinese characters 發票, QR codes and barcode for a store purchase, not related to train travel",
    "a white store receipt with chinese characters 發票, multiple barcodes and QR codes showing item purchase details"
]

# Threshold for classification confidence
CLASSIFICATION_THRESHOLD = 0.6

def classify_image(b64_image):
    """
    Classify a base64 encoded image as ticket, receipt, or invalid image.
    
    Args:
        b64_image: Base64 encoded image string
        
    Returns:
        str: Classification result ("ticket", "receipt", or "invalid image")
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save to temporary file for classification
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
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
                "a pink/red iBon printed HSR ticket with iBon logo on sides and circular stamp marking"
            ]:
                return "ticket"
            elif classified_class in [
                "a simple receipt with with chinese characters 發票, QR codes and barcode for a store purchase, not related to train travel",
                "a white store receipt with chinese characters 發票, multiple barcodes and QR codes showing item purchase details"
            ]:
                return "receipt"
            else:
                return "invalid image"
    
    except Exception as e:
        # Return "invalid image" for any processing errors
        return "invalid image"

# Example usage
if __name__ == "__main__":
    # Example with a local file
    with open("digi_receipt.jpg", "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')
        result = classify_image(b64_string)
        print(f"Classification result: {result}")