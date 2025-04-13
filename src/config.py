import os

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Get the repository directory (one level up from src)
REPO_DIR = os.path.dirname(SCRIPT_DIR)

# Get the parent directory of the repository (one level up from the repository)
PARENT_DIR = os.path.dirname(REPO_DIR)

# Path to the data directory (outside the repository)
DATA_DIR = os.path.join(PARENT_DIR, "coverletter_data")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Default LLM model to use for analysis and generation
DEFAULT_LLM_MODEL = "gemma3:12b"
