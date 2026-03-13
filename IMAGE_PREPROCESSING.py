import os
import random
import cv2

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

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

# ============================================================================
# RUNNING PREPROCESSING
# ============================================================================

# Paths to input and output folders
INPUT_DIR = "./Datasets/aptos2019-blindness-detection/train_images"
OUTPUT_DIR = "./Datasets/aptos2019-blindness-detection/preprocessed_images"

# Set sample size here (use None to process all images)
SAMPLE_SIZE = 5

# Get all image files
all_images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_images = len(all_images)
print(f"Found {total_images} images in '{INPUT_DIR}'.\n")

# Select images
if SAMPLE_SIZE is None or SAMPLE_SIZE >= total_images:
    selected_images = all_images
else:
    selected_images = random.sample(all_images, SAMPLE_SIZE)

print(f"Processing {len(selected_images)} images...\n")

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run preprocessing
success = 0
for i, filename in enumerate(selected_images):
    try:
        preprocessed = preprocess_image(os.path.join(INPUT_DIR, filename))
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), preprocessed)
        success += 1
    except Exception as e:
        print(f"  Failed: {filename} — {e}")

    if (i + 1) % 10 == 0 or (i + 1) == len(selected_images):
        print(f"  Progress: {i + 1}/{len(selected_images)}")

print(f"\nDone! {success}/{len(selected_images)} images saved to '{OUTPUT_DIR}'")