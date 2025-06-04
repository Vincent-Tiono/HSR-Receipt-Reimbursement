from transformers import pipeline
import json
import requests

# Import extract functions from each module
from card.main_old import extract as extract_card
from fami.main_old import extract as extract_fami
from ibon.main_old import extract as extract_ibon

def extract(image_path):
    """
    Classifies an HSR ticket image and processes it with the appropriate module.
    
    Args:
        image_path: Path to the image to classify and process.
    
    Returns:
        The formatted JSON response from the appropriate extraction function.
    """
    
    # Load the SigLIP model for zero-shot image classification
    ckpt = "google/siglip2-so400m-patch14-384"
    pipe = pipeline(model=ckpt, task="zero-shot-image-classification")
    
    # Define the candidate labels for classification
    texts = [
        "a white card-style HSR ticket with orange stripe",
        "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code",
        "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides"
    ]
    
    # Classify the image
    outputs = pipe(image_path, candidate_labels=texts)
    
    # Find the label with the highest score
    match = max(outputs, key=lambda x: x['score'])
    highest_label = match['label']
    
    # Map the classification result to ticket type
    if highest_label == "a white card-style HSR ticket with orange stripe":
        type = "card"
    elif highest_label == "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code":
        type = "fami"
    elif highest_label == "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides":
        type = "ibon"
    else:
        type = "unknown"
    
    # Process the image using the appropriate extraction function
    response = None
    if type == "card":
        response = extract_card(image_path)
    elif type == "fami":
        response = extract_fami(image_path)
    elif type == "ibon":
        response = extract_ibon(image_path)

    response = json.loads(response)
    
    # Modify and format the response
    date = response.get("date", "")
    price = response.get("price", 0)
    serial_number = response.get("serial_number", "")
    
    formatted_response = {
        "fid": 155,
        "apikey": "y2S2Va@&Pr5m",
        "fowner": "u109080",
        "TagComments": "台北 - 新竹往返",
        "TagDocDate": "2025/03/12",
        "TagU_PC_BSTY1_Code": 2,
        "TagU_PC_BSDAT_Code": date,
        "TagU_PC_BSINV_Code": serial_number,
        "TagU_PC_BSNOT": "16446274",
        "TagU_PC_BSAMN_Code": price,
        "TagU_PC_BSTAX_Code": "0",
        "TagU_PC_BSAMT_Code": price,
        "TagU_PC_BSTY5_Code": 0,
        "AR0001": [
            {
                "列日期": date,
                "TagReasonName": "高鐵費",
                "TagAcctCode": "611303",
                "TagQuantity": 1,
                "TagPriceBefDi": price,
                "UntaxedAmount": price,
                "TagVatName": "進項稅額0%",
                "TagVatGroup": "P0",
                "TagVatRate": 0,
                "TagPrice": price,
                "TagHP_Description": "台北新竹往返-2月"
            }
        ],
        "金額總計": price,
        "AR0002": [
            {
                "TagU_PC_BSTY1": "二聯式收銀機∕載有稅額之其他憑證-22",
                "TagU_PC_BSDAT": date,
                "TagU_PC_BSINV": serial_number,
                "TagU_PC_BSNOT": "16446274",
                "TagU_PC_BSAMN": price,
                "TagU_PC_BSTAX": 0,
                "TagU_PC_BSAMT": price,
                "TagU_PC_BSTY5": "是"
            }
        ],
        "TagDpmVat": price,
        "TagVatSum": 0,
        "TagDocTotal": price
    }

    return formatted_response

def process_multiple_images(image_paths):
    """
    Process multiple images and collect their results.
    
    Args:
        image_paths: List of paths to images to process.
        
    Returns:
        Dictionary mapping image paths to their formatted processing results.
    """
    results = {}
    for image_path in image_paths:
        result = extract(image_path)
        results[image_path] = result
    return results

if __name__ == "__main__":
    # Example usage
    img_path = "ibon/images/image_10_ibon.png"
    # Process a single image
    result = extract(img_path)
    print(json.dumps(result, indent=4, ensure_ascii=False))

    url = 'https://10.5.1.10/eipplus/formsflow/start.php'
    x = requests.post(url, json=result, verify=False)
    print(x.text)
