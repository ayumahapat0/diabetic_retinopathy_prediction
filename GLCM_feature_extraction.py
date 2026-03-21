import os
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

# Two window sizes as per the paper (64x64 and 128x128)
WINDOW_SIZES = [64, 128]

# GLCM distances and angles — standard settings
DISTANCES = [1]
ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]

BASE_DIR = "/Users/tanaymihani/PycharmProjects/PythonProject_CS5100_Diabetic_Rethinopathy"
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
OUTPUT_DIR = os.path.join(BASE_DIR, "features")



def extract_glcm_features(image, window_size):
    # Resizing the image to window size and extract Haralick features from GLCM
    # We average across all angles to get rotation-invariant features
    # 4 angles: 0, 45, 90, 135 degrees

    img_resized = cv2.resize(image, (window_size, window_size))

    # Normalizing to 64 gray levels to reduce GLCM matrix size
    # Full 256 levels would make the matrix huge and slow
    img_normalized = (img_resized / 4).astype(np.uint8)

    glcm = graycomatrix(
        img_normalized,
        distances  = DISTANCES,
        angles     = ANGLES,
        levels     = 64,
        symmetric  = True,
        normed     = True
    )

    # Extracting Haralick features(averaged across angles)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    asm = graycoprops(glcm, 'ASM').mean()

    # Additional Haralick features computed manually from GLCM
    glcm_mean    = glcm.mean()
    glcm_var     = glcm.var()
    entropy      = shannon_entropy(glcm)

    features = [
        contrast, dissimilarity, homogeneity,
        energy, correlation, asm,
        glcm_mean, glcm_var, entropy
    ]

    return np.array(features)


def extract_features_for_split(image_paths):
    # Extracting features for all images at both window sizes
    # Final feature vector per image = features at 64x64 + features at 128x128

    all_features = []
    total = len(image_paths)

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Could not read image: {img_path}, skipping")
            all_features.append(np.zeros(len(WINDOW_SIZES) * 9))
            continue

        # Extracting features at each window size and then concatenate
        feature_vector = []
        for ws in WINDOW_SIZES:
            feats = extract_glcm_features(img, ws)
            feature_vector.extend(feats)

        all_features.append(feature_vector)

        # Progress update every 100 images
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Progress: {i+1}/{total}")

    return np.array(all_features)



def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load splits saved from Phase 1
    for split_name in ["train", "val", "test"]:
        print(f"\nExtracting features for {split_name} set...")

        split = np.load(os.path.join(SPLITS_DIR, f"{split_name}_split.npz"))
        paths = split["paths"]
        labels = split["labels"]

        features = extract_features_for_split(paths)

        # Saving features + labels
        save_path = os.path.join(OUTPUT_DIR, f"{split_name}_features.npz")
        np.savez(save_path, features=features, labels=labels)

        print(f" {split_name} features -> {save_path}")
        print(f" Shape: {features.shape}  (images x features)")


    print(f"Feature vector size per image: {len(WINDOW_SIZES) * 9} ({len(WINDOW_SIZES)} window sizes x 9 features)")


if __name__ == "__main__":
    main()