"""
Test configuration that overrides the main application configuration.
This ensures tests use temporary directories instead of the actual data directory.
"""

import os
import tempfile

# Create a temporary directory for test data
TEST_DATA_DIR = tempfile.mkdtemp()

# Ensure the test data directory exists
os.makedirs(TEST_DATA_DIR, exist_ok=True)

# Create subdirectories that mimic the structure of the real data directory
os.makedirs(os.path.join(TEST_DATA_DIR, "json"), exist_ok=True)
os.makedirs(os.path.join(TEST_DATA_DIR, "config"), exist_ok=True)
os.makedirs(os.path.join(TEST_DATA_DIR, "text-archive"), exist_ok=True)
