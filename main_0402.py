from transformers import pipeline, AutoModelForImageSegmentation
import json
import logging
import sys
import warnings
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import tempfile
import os

# Suppress warnings and logs
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.models.registry is deprecated, please import via timm.models")
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.models.layers is deprecated, please import via timm.layers")
warnings.filterwarnings("ignore", message=r"Using a slow image processor.*")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Import extract functions from each module
from card.main import extract as extract_card
from fami.main_extract import extract as extract_fami
from ibon.main import extract as extract_ibon
from pdf.main import extract as extract_pdf

# Set device and precision for background removal
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the background removal model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(DEVICE)

# Define image transformation for background removal
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def process(image):
    """Process an image to remove the background."""
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    return image

def crop_transparent(img, padding=5):
    """Crop image to remove transparent areas, with optional padding."""
    if img.mode == 'RGBA':
        img_array = np.array(img)
        alpha = img_array[:, :, 3]
        non_transparent_pixels = np.where(alpha > 0)
        if len(non_transparent_pixels[0]) == 0:
            return img
        min_y, max_y = np.min(non_transparent_pixels[0]), np.max(non_transparent_pixels[0])
        min_x, max_x = np.min(non_transparent_pixels[1]), np.max(non_transparent_pixels[1])
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(img.width, max_x + padding)
        max_y = min(img.height, max_y + padding)
        return img.crop((min_x, min_y, max_x, max_y))
    return img

def process_image_from_pil(im, padding=10):
    """Take a PIL Image, remove its background, crop transparent edges, and return the cropped image."""
    transparent = process(im)
    cropped = crop_transparent(transparent, padding=padding)
    return cropped

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
        "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides"
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
    elif highest_label == "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides":
        print("iBon")
        return json.loads(extract_ibon(image_path))

def process_multiple_images(image_path):
    """
    Process multiple images and return the extracted JSON result.
    For PDF inputs, directly calls extract_pdf. For image inputs,
    performs background removal and cropping before classification.
    
    Args:
        image_path: Path to the file to process.
        
    Returns:
        A dictionary containing the extracted JSON data.
    """
    if image_path.lower().endswith('.pdf'):
        response = json.loads(extract_pdf(image_path))
    else:
        # Open the image with PIL
        im = Image.open(image_path).convert("RGB")
        # Process the image: remove background & crop transparent areas
        processed_im = process_image_from_pil(im)
        # Save the processed image to a temporary file (PNG supports transparency)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            processed_im.save(temp_path)
        # Classify and extract based on the processed image
        response = classify(temp_path)
        os.remove(temp_path)
    return response

if __name__ == "__main__":
    img_paths = r"C:\Users\ubiik-ai-vincent\Documents\0402 ticket\automated-travel-expense-reimbursement\fami\images\2.jpg"
    extracted_data = process_multiple_images(img_paths)
    print(extracted_data)
    # print(json.dumps(extracted_data, indent=4, ensure_ascii=False))