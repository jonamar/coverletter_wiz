#!/bin/bash
# Script to activate the virtual environment for coverletter_wiz

# Path to the virtual environment
VENV_PATH="$(dirname "$0")/.venv"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Set any necessary environment variables
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# Print confirmation message
echo "Virtual environment activated for coverletter_wiz"
echo "Project root: $(dirname "$0")"
echo "Python interpreter: $(which python)"
echo "Python version: $(python --version)"

# List installed packages
echo "Installed packages:"
pip list
