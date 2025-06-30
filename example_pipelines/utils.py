from pathlib import Path

def get_project_root():
    """Return the absolute path to the project root directory."""
    # Navigate up from utils.py to the project root
    return Path(__file__).parent.parent