# Data Splitting

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import Counter


RANDOM_SEED = 42
TEST_SIZE   = 0.20
VAL_SIZE    = 0.125   # 12.5% of remaining 80% = 10% of total
N_FOLDS     = 5

BASE_DIR         = "."
CSV_PATH         = os.path.join(BASE_DIR, "aptos2019-blindness-detection", "train.csv")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "PREPROCESSED_IMAGES")
OUTPUT_DIR       = os.path.join(BASE_DIR, "splits")


# Load the pre-processed images

def load_dataset(csv_path, preprocessed_dir):
    # Read labels from train.csv
    # APTOS has 5 severity classes (0-4), we convert to binary:
    #   0 → 0 (No DR)
    #   1,2,3,4 → 1 (DR)
    # We do this here so everything downstream works with binary labels

    df = pd.read_csv(csv_path)

    print(f"\n[INFO] Total records in CSV: {len(df)}")
    print(f"[INFO] Original distribution: {dict(sorted(df['diagnosis'].value_counts().items()))}")

    df['binary_label'] = df['diagnosis'].apply(lambda x: 0 if x == 0 else 1)

    print(f"[INFO] After binary conversion -> 0: {sum(df['binary_label']==0)}, 1: {sum(df['binary_label']==1)}")

    # Match each image to its preprocessed file
    image_paths = []
    labels      = []
    missing     = 0

    for _, row in df.iterrows():
        img_path = os.path.join(preprocessed_dir, row['id_code'] + ".png")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            labels.append(row['binary_label'])
        else:
            missing += 1

    print(f"[INFO] Images matched: {len(image_paths)}")
    if missing > 0:
        print(f"[WARN] {missing} images in CSV not found in preprocessed folder")

    return image_paths, labels


# Splitting the Dataset

def split_dataset(image_paths, labels):
    # 70% train, 10% val, 20% test
    # Using stratified splits to maintain class balance in each set

    labels_array = np.array(labels)
    paths_array  = np.array(image_paths)

    # Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        paths_array, labels_array,
        test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels_array
    )

    # Tain and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_temp
    )

    total = len(image_paths)
    print("\n" + "="*50)
    print("SPLIT SUMMARY")
    print("="*50)
    print(f"  Total  : {total}")
    print(f"  Train  : {len(X_train)} ({len(X_train)/total*100:.1f}%) | 0: {sum(y_train==0)}  1: {sum(y_train==1)}")
    print(f"  Val    : {len(X_val)}  ({len(X_val)/total*100:.1f}%) | 0: {sum(y_val==0)}  1: {sum(y_val==1)}")
    print(f"  Test   : {len(X_test)} ({len(X_test)/total*100:.1f}%) | 0: {sum(y_test==0)}  1: {sum(y_test==1)}")
    print("="*50)

    return {
        "train": {"paths": X_train, "labels": y_train},
        "val"  : {"paths": X_val,   "labels": y_val},
        "test" : {"paths": X_test,  "labels": y_test},
    }


# Cross Validation

def get_cross_validation_folds(train_paths, train_labels):
    # 5-fold CV on training set only as in the paper
    # Stratified so each fold has same class balance

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    folds = []
    print(f"\n[INFO] {N_FOLDS}-Fold CV on training set")
    print("="*50)

    for fold_num, (train_idx, val_idx) in enumerate(skf.split(train_paths, train_labels), 1):
        folds.append((train_idx, val_idx))
        ft = train_labels[train_idx]
        fv = train_labels[val_idx]
        print(f"  Fold {fold_num} -> Train: {len(train_idx)} (0:{sum(ft==0)} 1:{sum(ft==1)}) | Val: {len(val_idx)} (0:{sum(fv==0)} 1:{sum(fv==1)})")

    print("="*50)
    return skf, folds


# Saving the Splits

def save_splits(splits, output_dir):
    # Saving as .npz for reusability

    os.makedirs(output_dir, exist_ok=True)

    for split_name, data in splits.items():
        save_path = os.path.join(output_dir, f"{split_name}_split.npz")
        np.savez(save_path, paths=data["paths"], labels=data["labels"])
        print(f"[SAVED] {split_name} -> {save_path}")



def main():
    image_paths, labels = load_dataset(CSV_PATH, PREPROCESSED_DIR)
    splits              = split_dataset(image_paths, labels)
    skf, folds          = get_cross_validation_folds(splits["train"]["paths"], splits["train"]["labels"])
    save_splits(splits, OUTPUT_DIR)

    print(f"  Train: {len(splits['train']['paths'])} | Val: {len(splits['val']['paths'])} | Test: {len(splits['test']['paths'])}")
    print(f"  {N_FOLDS} CV folds ready for Phase 2")

    return splits, skf, folds


if __name__ == "__main__":
    splits, skf, folds = main()