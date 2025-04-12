#!/usr/bin/env python3
"""
Test runner for Cover Letter Wizard unit tests.

This script discovers and runs all tests in the tests directory.
"""

import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests():
    """Discover and run all tests in the tests directory."""
    # Discover all tests in the current directory
    loader = unittest.TestLoader()
    test_suite = loader.discover(os.path.dirname(os.path.abspath(__file__)))
    
    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    result = test_runner.run(test_suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
