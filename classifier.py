# pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2
from transformers import pipeline

ckpt = "google/siglip2-so400m-patch14-384"
pipe = pipeline(model=ckpt, task="zero-shot-image-classification")

# Define the input images and candidate labels
inputs = {
    "images": [
        "card\images\image_5.png",  # Replace with the actual path or URL of the first image
        "fami\images\image_3_fami.png",  # Replace with the actual path or URL of the second image
        "card\images\image_2.png",  # Replace with the actual path or URL of the third image
    ],
    "texts": [
        "a white card-style HSR ticket with orange stripe",
        "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code",
        "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides"
    ],
}

# Perform zero-shot classification
outputs = pipe(inputs["images"], candidate_labels=inputs["texts"])

# Process and print the results in the desired format
for image_path, output in zip(inputs["images"], outputs):
    # Find the label with the highest score
    classified_class = max(output, key=lambda x: x['score'])['label']
    if classified_class == "a white card-style HSR ticket with orange stripe":
        classified_class = "card"
    elif classified_class == "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code":
        classified_class = "fami"
    elif classified_class == "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides":
        classified_class = "ibon"
    print(f"{image_path}: {classified_class}\n")