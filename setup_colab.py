import os
import sys
from pathlib import Path

# --- Configuration ---
REPO_URL = "https://github.com/brenobertone/pinn.git"
PROJECT_NAME = "pinn"

# 1. Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("Running in Google Colab.")
    if not os.path.exists(PROJECT_NAME):
        print(f"Cloning {PROJECT_NAME}...")
        !git clone {REPO_URL}
        %cd {PROJECT_NAME}
    else:
        %cd {PROJECT_NAME}
        print("Repository already exists.")
else:
    print("Running in local environment (VS Code).")
    # Automatically find the project root
    def find_project_root(current_path, marker="train_interactive.py"):
        current = Path(current_path).resolve()
        for parent in [current] + list(current.parents):
            if (parent / marker).exists():
                return parent
        return current
    
    project_root = find_project_root(os.getcwd())
    os.chdir(project_root)

# 2. Final Path Verification
project_root = Path(os.getcwd()).resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"\nProject root: {project_root}")
print(f"Working directory: {os.getcwd()}")

# Ensure results directory exists
os.makedirs("results", exist_ok=True)
print("Environment ready.")