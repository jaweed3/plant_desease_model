from PIL import Image
import numpy as np

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    if image.shape[-1] == 4:  # Check for alpha channel
        image = image[..., :3]  # Remove alpha channel if present
    return np.expand_dims(image, axis=0)  # Add batch dimension

def preprocess_input(image):
    # Additional preprocessing steps can be added here if needed
    return image