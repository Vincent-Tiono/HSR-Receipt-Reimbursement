from loadimg import load_img
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np

# Set device and precision
torch.set_float32_matmul_precision("high")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
birefnet.to(DEVICE)

# Define image transformation
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# def rotate_to_straight(image):
#     """Automatically detect and correct image tilt."""
#     image = ImageOps.exif_transpose(image)  # Correct orientation using EXIF data
#     return image.rotate(-2, expand=True)  # Slight rotation correction

def process(image):
    """Process an image to remove the background."""
    # image = rotate_to_straight(image)
    image_size = image.size
    input_images = transform_image(image).unsqueeze(0).to(DEVICE)
    
    # Prediction
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

def process_file(input_path, output_path):
    """Process a file from input path and save to output path."""
    print(f"Processing image: {input_path}")
    try:
        im = load_img(input_path, output_type="pil")
    except Exception as e:
        print(f"Error loading image with load_img: {e}")
        im = Image.open(input_path)
    
    im = im.convert("RGB")
    transparent = process(im)
    cropped = crop_transparent(transparent, padding=10)
    
    # Option 1: Change the output file extension to PNG
    if output_path.lower().endswith(('.jpg', '.jpeg')):
        output_path = output_path.replace('.jpg', '.png').replace('.jpeg', '.png')
    
    # Option 2: Or convert to RGB if you want to keep JPEG
    # cropped = cropped.convert('RGB')
    
    cropped.save(output_path)
    print(f"Successfully saved cropped image to: {output_path}")
    return output_path

if __name__ == "__main__":
    input_path = "ibon/images/4.jpg"
    # Change output to PNG since we want to keep transparency
    output_path = "ibon/images/4_removed_bg.png"
    
    process_file(input_path, output_path)