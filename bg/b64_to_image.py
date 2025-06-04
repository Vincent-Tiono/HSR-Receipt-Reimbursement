import base64
from io import BytesIO
from PIL import Image

def base64_to_image(base64_str: str) -> Image.Image:
    """Convert a Base64 string back to a PIL Image."""
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))

# Read the base64 string from the txt file
with open(r"bg\removed_bg_b64.txt", "r") as file:
    base64_str = file.read().strip()

image = base64_to_image(base64_str)
image.show()  # Display the image
# image.save("output.png")  # Save the image