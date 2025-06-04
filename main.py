from transformers import pipeline
import json
import logging
import sys
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.models.registry is deprecated, please import via timm.models")
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.models.layers is deprecated, please import via timm.layers")
warnings.filterwarnings("ignore", message=r"Using a slow image processor.*")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Import extract functions from each module
from card.main import extract as extract_card
from fami.main import extract as extract_fami
from ibon.main import extract as extract_ibon
from pdf.main import extract as extract_pdf

def classify(image_path):
    """
    Classifies the HSR ticket image into one of the ticket types.
    
    Args:
        image_path: Path to the image to classify.
    
    Returns:
        The ticket type based on the classification.
    """
    ckpt = "google/siglip2-so400m-patch14-384"
    pipe = pipeline(model=ckpt, task="zero-shot-image-classification")
    texts = [
        "a white card-style HSR ticket with orange stripe",
        "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code",
        "a iBon ticket printed with a pink/red top row and white background in the middle and bottom, featuring a circular stamp and the 'iBon' logo on its sides"
    ]
    outputs = pipe(image_path, candidate_labels=texts)
    for output in outputs:
        print(f"Label: {output['label']}, Score: {output['score']}\n")


    match = max(outputs, key=lambda x: x['score'])
    highest_label = match['label']
    confidence = match['score']
    
    if confidence < 0.5:
        print("Unknown ticket type, please upload again.")
        sys.exit(1)
    
    if highest_label == "a white card-style HSR ticket with orange stripe":
        print("Card")
        return json.loads(extract_card(image_path))
    elif highest_label == "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code":
        print("Fami")
        return json.loads(extract_fami(image_path))
    elif highest_label == "a iBon ticket printed with a pink/red top row and white background in the middle and bottom, featuring a circular stamp and the 'iBon' logo on its sides":
        print("iBon")
        return json.loads(extract_ibon(image_path))


def process_image(image_path):
    """
    Process multiple images and return a list of extracted JSON results.
    
    Args:
        image_paths: List of paths to images to process.
        
    Returns:
        A list of dictionaries containing the extracted JSON data.
    """
    lower_path = image_path.lower()
    if lower_path.endswith('.pdf'):
        response = json.loads(extract_pdf(image_path))
    elif lower_path.endswith(('.png', '.jpg', '.jpeg')):
        response = classify(image_path)
    else:
        print("File format not supported")
        sys.exit(1)
    return response

if __name__ == "__main__":
    img_path = r"ibon/images/7.jpg"
    extracted_data = process_image(img_path)
    print(extracted_data)
    # print(json.dumps(extracted_data, indent=4, ensure_ascii=False))