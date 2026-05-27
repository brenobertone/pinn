# --- 1. Clone & Install ---
import os

# Repository configuration
GITHUB_REPO = "https://github.com/brenobertone/pinn.git" 
PROJECT_NAME = "pinn"

if not os.path.exists(PROJECT_NAME):
    print(f"Cloning {PROJECT_NAME}...")
    !git clone {GITHUB_REPO}
    %cd {PROJECT_NAME}
    print("Installing dependencies in editable mode...")
    !pip install -e .
else:
    %cd {PROJECT_NAME}
    print("Updating repository...")
    !git pull

# --- 2. Persist Results to Google Drive ---
from google.colab import drive
import os

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Create results folder in Drive if it doesn't exist
DRIVE_RESULTS_DIR = f"/content/drive/MyDrive/{PROJECT_NAME}_results"
os.makedirs(DRIVE_RESULTS_DIR, exist_ok=True)

# Symlink the local results/ folder to Drive for persistence
# This ensures that even if the Colab session dies, your models and plots are safe.
if os.path.islink('results'):
    !rm results
elif os.path.exists('results'):
    # Move existing local results to Drive to avoid data loss
    !mv results/* {DRIVE_RESULTS_DIR}/ 2>/dev/null || true
    !rm -rf results

!ln -s "{DRIVE_RESULTS_DIR}" results
print(f"Results will be synced to: {DRIVE_RESULTS_DIR}")

# --- 3. Run Training ---
# The script is interactive; look for the input prompts below the cell.
!python train_interactive.py
