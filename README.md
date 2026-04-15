---
title: Diabetic Retinopathy Detection
emoji: 🎨
---

# Diabetic Retinopathy Detection

## 📋 What This Project Does

1. Preprocesses eyes images from the aptos 2019 blindness detection dataset from Kaggle
2. Splits the processed images into a training, validation, and testing split of 70%, 10%, and 20% respectively
3. Extracts GLCM features from the each split
4. Utilizes a Genetic algorithm for feature selection and lightgbm hyperparameter tuning
5. Evaluates and compares the optimized model to a baseline model utilizing all GLCM features and no hyperparameter tuning 
through metrics and plots

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- A virtual environment in order install the libraries 

### Clone the Repository

```bash
git clone https://github.com/ayumahapat0/diabetic_retinopathy_prediction.git
cd diabetic_reintopathy_prediction
```
## Running Pipeline

There are 2 ways to run this pipeline: utilizing the bash script or running each file one at a time

### Option 1: Bash Scipt

The easiest way is to run the bash script

### Run Bash Script
```bash
./run_pipeline.sh
```

If you want to make sure you can use the bash script, just change the execution permissions locally

### Changing Execution Permissions

```bash
chmod u+x run_pipeline.sh
```

### Changing Execution Premissions in Github Repository

```bash
git update-index --chmod=+x run_pipeline.sh
git commit -m "message"
```

Run this line code to see the execution permissions of all the files

064: unexecutable
075: executable

```bash
git ls-files -s
```

### Option 2: Run each file one at a time

### Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install {module 1} {module 2} . . .
```

### Processing

```bash
python3 preprocess_images.py
```

### Data Splitting

```bash
python3 data_splitting.py
```

### GLCM Feature Extraction

```bash
python3 glcm_feature_extractions.py
```

### GA Optimization and Evaluation

```bash
python3 ga_optimization_and_evaluation.py
```

The results will be in the results fold

```bash
cd results
```
---
