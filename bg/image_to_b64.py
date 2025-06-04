import base64
from io import BytesIO
from PIL import Image

def image_to_base64(image: Image.Image, format="PNG") -> str:
    """Convert a PIL Image to a Base64 string."""
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Encode image bytes to Base64 string
    img_base64 = base64.b64encode(img_byte_arr).decode("utf-8")
    return img_base64

# Load an image
image = Image.open(r"C:\Users\ubiik-ai-vincent\Documents\0402 ticket\automated-travel-expense-reimbursement\card\images\6.jpg")
base64_str = image_to_base64(image)

# Save the Base64 string to a text file
with open(r"C:\Users\ubiik-ai-vincent\Documents\0402 ticket\automated-travel-expense-reimbursement\bg\image_base64.txt", "w") as f:
    f.write(base64_str)