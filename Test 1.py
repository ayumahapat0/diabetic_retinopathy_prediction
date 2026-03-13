import os
import random
import cv2
from IMAGE_PREPROCESSING import preprocess_image

# Paths
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