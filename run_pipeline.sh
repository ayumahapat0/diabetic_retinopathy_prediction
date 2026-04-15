#!/bin/bash
 
# =============================================================================
# run_pipeline.sh
# Creates a virtual environment, installs dependencies, and runs entire pipeline
# =============================================================================
 
set -e  # Exit immediately if any command fails
 
# --- Configuration -----------------------------------------------------------
 
VENV_DIR="venv"
REQUIREMENTS="requirements.txt"
 
# Pipeline in order: Preprocessing -> data splitting -> feature extraction -> optimization -> evaluation
PYTHON_FILES=(
    "preprocess_images.py"
    "data_splitting.py"
    "glcm_feature_extractions.py"
    "ga_optimization_and_evaluation.py"
)
 
# --- Helpers -----------------------------------------------------------------
 
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
RESET="\033[0m"
 
info()    { echo -e "${GREEN}[INFO]${RESET}  $1"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $1"; }
error()   { echo -e "${RED}[ERROR]${RESET} $1"; }
 
# --- Step 1: Create virtual environment --------------------------------------
 
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    info "Creating virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
    info "Virtual environment created."
fi
 
# --- Step 2: Activate virtual environment ------------------------------------
 
info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
 
# --- Step 3: Install dependencies --------------------------------------------
 
if [ ! -f "$REQUIREMENTS" ]; then
    error "Requirements file '$REQUIREMENTS' not found. Aborting."
    exit 1
fi
 
info "Installing dependencies from '$REQUIREMENTS'..."
pip install --upgrade pip --quiet
pip install -r "$REQUIREMENTS"
info "Dependencies installed."
 
# --- Step 4: Run Python files in order ---------------------------------------
 
info "Starting script execution...\n"
 
for script in "${PYTHON_FILES[@]}"; do
    if [ ! -f "$script" ]; then
        error "Script '$script' not found. Aborting."
        exit 1
    fi
 
    info "Running: $script"
    echo "------------------------------------------------------------"
    python3 "$script"
    echo "------------------------------------------------------------"
    info "Finished: $script\n"
done
 
# --- Done --------------------------------------------------------------------
 
info "All scripts completed successfully."
deactivate