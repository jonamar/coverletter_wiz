"""
Configuration settings for the coverletter_wiz package.

This module defines configuration settings and paths used throughout the application,
including data directory locations and default model settings.

The configuration is designed to keep all user data separate from the application code
in an external data directory, aligned with the privacy approach of the project.
"""

import os
from typing import Final

# Get the directory of the current script
SCRIPT_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))

# Get the repository directory (one level up from src)
REPO_DIR: Final[str] = os.path.dirname(SCRIPT_DIR)

# Get the parent directory of the repository (one level up from the repository)
PARENT_DIR: Final[str] = os.path.dirname(REPO_DIR)

# Path to the data directory (outside the repository)
DATA_DIR: Final[str] = os.path.join(PARENT_DIR, "coverletter_data")

# Path to the reports directory within the data directory
REPORTS_DIR: Final[str] = os.path.join(DATA_DIR, "reports")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Ensure the reports directory exists
os.makedirs(REPORTS_DIR, exist_ok=True)

# Default LLM model to use for analysis and generation
DEFAULT_LLM_MODEL: Final[str] = "gemma3:12b"
