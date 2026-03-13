# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
 
import cv2

# Removing the black background from fundus image using contour detection
def remove_background(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        image = image[y:y+h, x:x+w]
    return image

"""
Full preprocessing pipeline:
    1. Reading the image
    2. Removing the black background
    3. Applying CLAHE for histogram equalization
    4. Resizing it to target_size
    5. Converting it to grayscale
"""
def preprocess_image(image_path, target_size=(800, 800)):
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
 
    img = remove_background(img)        # Removing the background
 
    # Applying CLAHE on LAB color space (L channel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
 
    # Resizing the image to target size
    img = cv2.resize(img, target_size)
 
    # Converting the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    return gray
    