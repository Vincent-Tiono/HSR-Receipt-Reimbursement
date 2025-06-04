import requests
import base64

def classify_pdf():
    # Read base64 content from file
    with open('tmp2.txt', 'r') as f:
        base64_content = f.read().strip()

    # API endpoint
    url = 'http://192.168.73.29:57543/classify'

    # Prepare request data
    data = {
        'b64': base64_content
    }

    try:
        # Make POST request
        response = requests.post(url, json=data)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Print classification result
        print("Classification result:", response.json())
        
    except requests.exceptions.RequestException as e:
        print("Error making request:", e)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    classify_pdf()
