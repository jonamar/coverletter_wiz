#!/usr/bin/env python3
"""
Integration tests for the project structure and main application functionality.

These tests verify that:
1. Modules can be properly imported from the src directory
2. The CLI help command works correctly
"""

import os
import sys
import unittest
import subprocess
from unittest.mock import patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProjectStructure(unittest.TestCase):
    """Test the project structure and main application functionality."""

    def test_import_modules(self):
        """Test that modules can be properly imported from the src directory."""
        try:
            from src.core.content_matcher import ContentMatcher
            from src.core.job_analyzer import JobAnalyzer
            from src.config import DATA_DIR
            
            # Verify DATA_DIR points to external data directory
            self.assertTrue("coverletter_data" in DATA_DIR, 
                           f"DATA_DIR should point to external data directory, got: {DATA_DIR}")
            
            # Test creating instances of key classes with minimal initialization
            # We're not testing functionality, just that the classes can be instantiated
            matcher = ContentMatcher()
            analyzer = JobAnalyzer()
            
            # No assertion needed here - if we got this far without exceptions, the test passes
        except Exception as e:
            self.fail(f"Failed to import modules or create class instances: {e}")

    def test_cli_help(self):
        """Test that the CLI help command works correctly."""
        # Get the path to the main script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "coverletter_wiz.py")
        
        # Run the help command
        result = subprocess.run(
            ["python", script_path, "--help"],
            capture_output=True,
            text=True
        )
        
        # The command might return non-zero due to warnings, but should still show usage info
        help_output = result.stdout + result.stderr
        self.assertTrue("usage:" in help_output, 
                       f"CLI help should show usage information, got:\n{help_output}")
        self.assertTrue("Cover Letter Wizard" in help_output, 
                       "CLI help should include application name")


if __name__ == "__main__":
    unittest.main()
