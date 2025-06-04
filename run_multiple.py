'''
from transformers import pipeline
import json
import requests

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
        return "card"
    elif highest_label == "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code":
        return "fami"
    elif highest_label == "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides":
        return "ibon"
    else:
        return "unknown"

def extract(image_path, ticket_type):
    """
    Extracts data from the image based on the ticket type.
    
    Args:
        image_path: Path to the image to extract information from.
        ticket_type: The type of ticket (card, fami, or ibon).
    
    Returns:
        The formatted JSON response after extraction.
    """
    # Process the image using the appropriate extraction function based on ticket type
    response = None
    if ticket_type == "card":
        response = extract_card(image_path)
    elif ticket_type == "fami":
        response = extract_fami(image_path)
    elif ticket_type == "ibon":
        response = extract_ibon(image_path)

    response = json.loads(response)
    
    # Extract values for processing
    date = response.get("date", "")
    price = response.get("price", 0)
    serial_number = response.get("serial_number", "")
    
    # Format the response for 'result_up' and 'result_down'
    formatted_response_up = {
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
        "TagHP_Description": ""
    }

    formatted_response_down = {
        "TagU_PC_BSTY1": "二聯式收銀機∕載有稅額之其他憑證-22",
        "TagU_PC_BSDAT": date,
        "TagU_PC_BSINV": serial_number,
        "TagU_PC_BSNOT": "16446274",
        "TagU_PC_BSAMN": price,
        "TagU_PC_BSTAX": 0,
        "TagU_PC_BSAMT": price,
        "TagU_PC_BSTY5": "是"
    }

    return formatted_response_up, formatted_response_down

def process_multiple_images(image_paths):
    """
    Process multiple images and collect their results into two lists: 
    one for `result_up` and one for `result_down`.
    
    Args:
        image_paths: List of paths to images to process.
        
    Returns:
        Two lists: one containing all `result_up` data, the other containing all `result_down` data.
    """
    result_up_total = []
    result_down_total = []
    
    for image_path in image_paths:
        # Classify the image first
        ticket_type = classify(image_path)
        # Then extract based on the classified ticket type
        result_up, result_down = extract(image_path, ticket_type)
        
        # Append the results to the respective lists
        result_up_total.append(result_up)
        result_down_total.append(result_down)
    
    return result_up_total, result_down_total

def format_final_result(result_up_total, result_down_total):
    """
    Format the final result into the required structure and return as a JSON object.
    
    Args:
        result_up_total: The list containing formatted 'result_up' data.
        result_down_total: The list containing formatted 'result_down' data.
    
    Returns:
        A dictionary with the final formatted JSON result.
    """
    # Calculate the price as the sum of TagPrice from result_up_total
    price = sum(item["TagPrice"] for item in result_up_total)
    
    final_result = {
        "fid": 155,
        "apikey": "y2S2Va@&Pr5m",
        "fowner": "u109080",
        "TagComments": "台北 - 新竹往返",
        "TagDocDate": "2025/03/12",
        "TagU_PC_BSTY1_Code": 2,
        "TagU_PC_BSDAT_Code": "",
        "TagU_PC_BSINV_Code": "",
        "TagU_PC_BSNOT": "16446274",
        "TagU_PC_BSAMN_Code": "0",
        "TagU_PC_BSTAX_Code": "0",
        "TagU_PC_BSAMT_Code": "0",
        "TagU_PC_BSTY5_Code": 0,
        "AR0001": result_up_total,
        "金額總計": price,
        "AR0002": result_down_total,
        "TagDpmVat": price,
        "TagVatSum": 0,
        "TagDocTotal": price
    }
    
    return final_result

if __name__ == "__main__":
    # Example usage
    img_paths = ["ibon/images/image_10_ibon.png", "card/images/image_1.png", "fami/images/image_3_fami.png", "pdf\docs\pdf_2.pdf"]
    
    # Process multiple images
    result_up_total, result_down_total = process_multiple_images(img_paths)
    
    # Format the final result into the required structure
    final_result = format_final_result(result_up_total, result_down_total)
    
    # Print the final formatted result as JSON
    print(json.dumps(final_result, indent=4, ensure_ascii=False))

    url = 'https://10.5.1.10/eipplus/formsflow/start.php'
    x = requests.post(url, json=final_result, verify=False)
    print(x.text)
'''
from transformers import pipeline
import json
import requests

# Import extract functions from each module
from card.main_old import extract as extract_card
from fami.main_old import extract as extract_fami
from ibon.main_old import extract as extract_ibon
from pdf.main import extract as extract_pdf

def classify(image_path):
    """
    Classifies the HSR ticket image into one of the ticket types.
    
    Args:
        image_path: Path to the image to classify.
    
    Returns:
        The ticket type based on the classification.
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
        return "card"
    elif highest_label == "a yellow/green receipt-style HSR ticket printed at FamilyMart with barcode and QR code":
        return "fami"
    elif highest_label == "a pink/red receipt-style HSR ticket printed at iBon with circular stamp and 'iBon' logo on sides":
        return "ibon"
    else:
        return "unknown"

def extract(image_path, ticket_type):
    """
    Extracts data from the image based on the ticket type.
    
    Args:
        image_path: Path to the image to extract information from.
        ticket_type: The type of ticket (card, fami, or ibon).
    
    Returns:
        The formatted JSON response after extraction.
    """
    # Process the image using the appropriate extraction function based on ticket type
    response = None
    if ticket_type == "card":
        response = extract_card(image_path)
    elif ticket_type == "fami":
        response = extract_fami(image_path)
    elif ticket_type == "ibon":
        response = extract_ibon(image_path)

    response = json.loads(response)
    
    # Extract values for processing
    date = response.get("date", "")
    price = response.get("price", 0)
    serial_number = response.get("serial_number", "")
    
    # Format the response for 'result_up' and 'result_down'
    formatted_response_up = {
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
        "TagHP_Description": ""
    }

    formatted_response_down = {
        "TagU_PC_BSTY1": "二聯式收銀機∕載有稅額之其他憑證-22",
        "TagU_PC_BSDAT": date,
        "TagU_PC_BSINV": serial_number,
        "TagU_PC_BSNOT": "16446274",
        "TagU_PC_BSAMN": price,
        "TagU_PC_BSTAX": 0,
        "TagU_PC_BSAMT": price,
        "TagU_PC_BSTY5": "是"
    }

    return formatted_response_up, formatted_response_down

def process_multiple_images(image_paths):
    """
    Process multiple images and collect their results into two lists: 
    one for `result_up` and one for `result_down`.
    
    Args:
        image_paths: List of paths to images to process.
        
    Returns:
        Two lists: one containing all `result_up` data, the other containing all `result_down` data.
    """
    result_up_total = []
    result_down_total = []
    
    for image_path in image_paths:
        if image_path.lower().endswith('.pdf'):
            # Directly extract from PDF without classification
            response = extract_pdf(image_path)
            response = json.loads(response)
            date = response.get("date", "")
            price = response.get("price", 0)
            serial_number = response.get("serial_number", "")
        else:
            # Classify the image first
            ticket_type = classify(image_path)
            # Then extract based on the classified ticket type
            result_up, result_down = extract(image_path, ticket_type)
            date = result_up["列日期"]
            price = result_up["TagPrice"]
            serial_number = result_down["TagU_PC_BSINV"]

        # Format the response for 'result_up' and 'result_down'
        formatted_response_up = {
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
            "TagHP_Description": ""
        }

        formatted_response_down = {
            "TagU_PC_BSTY1": "二聯式收銀機∕載有稅額之其他憑證-22",
            "TagU_PC_BSDAT": date,
            "TagU_PC_BSINV": serial_number,
            "TagU_PC_BSNOT": "16446274",
            "TagU_PC_BSAMN": price,
            "TagU_PC_BSTAX": 0,
            "TagU_PC_BSAMT": price,
            "TagU_PC_BSTY5": "是"
        }

        # Append the results to the respective lists
        result_up_total.append(formatted_response_up)
        result_down_total.append(formatted_response_down)
    
    return result_up_total, result_down_total

def format_final_result(result_up_total, result_down_total):
    """
    Format the final result into the required structure and return as a JSON object.
    
    Args:
        result_up_total: The list containing formatted 'result_up' data.
        result_down_total: The list containing formatted 'result_down' data.
    
    Returns:
        A dictionary with the final formatted JSON result.
    """
    # Calculate the price as the sum of TagPrice from result_up_total
    price = sum(item["TagPrice"] for item in result_up_total)
    
    final_result = {
        "fid": 155,
        "apikey": "y2S2Va@&Pr5m",
        "fowner": "u109080",
        "TagComments": "台北 - 新竹往返",
        "TagDocDate": "2025/03/12",
        "TagU_PC_BSTY1_Code": 2,
        "TagU_PC_BSDAT_Code": "",
        "TagU_PC_BSINV_Code": "",
        "TagU_PC_BSNOT": "16446274",
        "TagU_PC_BSAMN_Code": "0",
        "TagU_PC_BSTAX_Code": "0",
        "TagU_PC_BSAMT_Code": "0",
        "TagU_PC_BSTY5_Code": 0,
        "AR0001": result_up_total,
        "金額總計": price,
        "AR0002": result_down_total,
        "TagDpmVat": price,
        "TagVatSum": 0,
        "TagDocTotal": price
    }
    
    return final_result

if __name__ == "__main__":
    # Example usage
    img_paths = ["ibon/images/image_10_ibon.png", "card/images/image_1.png", "fami/images/image_3_fami.png", "pdf/docs/pdf_2.pdf"]
    
    # Process multiple images
    result_up_total, result_down_total = process_multiple_images(img_paths)
    
    # Format the final result into the required structure
    final_result = format_final_result(result_up_total, result_down_total)
    
    # Print the final formatted result as JSON
    print(json.dumps(final_result, indent=4, ensure_ascii=False))

    url = 'https://10.5.1.10/eipplus/formsflow/start.php'
    x = requests.post(url, json=final_result, verify=False)
    print(x.text)