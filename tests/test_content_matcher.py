"""
Unit tests for the ContentMatcher module.

These tests validate the error handling and core functionality
of the ContentMatcher class, focusing on robust error handling,
LLM interactions, and file operations.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import json
import tempfile
from datetime import datetime

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.content_matcher import ContentMatcher


class TestContentMatcher(unittest.TestCase):
    """Test cases for the ContentMatcher class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock content file
        self.content_file = os.path.join(self.test_dir, "content.json")
        self.content_data = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat()
            },
            "file1.md": {
                "content": {
                    "paragraphs": [
                        {
                            "sentences": [
                                {
                                    "text": "I am experienced in Python programming.",
                                    "rating": 8.5,
                                    "tags": ["python", "programming"]
                                },
                                {
                                    "text": "I have worked on many successful projects.",
                                    "rating": 7.0,
                                    "tags": ["experience", "projects"]
                                }
                            ]
                        }
                    ]
                }
            }
        }
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
        
        # Create a mock jobs file
        self.jobs_file = os.path.join(self.test_dir, "jobs.json")
        self.jobs_data = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat()
            },
            "jobs": [
                {
                    "id": 1,
                    "org_name": "TestCorp",
                    "job_title": "Software Engineer",
                    "summary": "A software engineering position",
                    "tags": {
                        "high_priority": ["python", "programming"],
                        "medium_priority": ["experience"],
                        "low_priority": ["projects"]
                    }
                }
            ],
            "sequential_jobs": {
                "jobs": [
                    {
                        "id": 1,
                        "org_name": "TestCorp",
                        "job_title": "Software Engineer",
                        "summary": "A software engineering position",
                        "tags": {
                            "high_priority": ["python", "programming"],
                            "medium_priority": ["experience"],
                            "low_priority": ["projects"]
                        }
                    }
                ]
            }
        }
        with open(self.jobs_file, "w") as f:
            json.dump(self.jobs_data, f)
        
        # Create the matcher
        self.matcher = ContentMatcher(
            jobs_file=self.jobs_file,
            content_file=self.content_file,
            llm_model="test-model"
        )
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove test files and directory
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_job_not_found(self):
        """Test handling of job not found error."""
        with self.assertRaises(ValueError) as context:
            # Attempt to generate a cover letter for a non-existent job
            try:
                self.matcher.generate_cover_letter(999)
            except Exception as e:
                raise ValueError(str(e))
                
        self.assertIn("Job with ID 999 not found", str(context.exception))
    
    @patch("ollama.generate")
    def test_ollama_connection_error(self, mock_generate):
        """Test handling of Ollama connection error in cover letter generation."""
        # Simulate connection refused error
        mock_generate.side_effect = Exception("connection refused")
        
        # Call the function under test
        result = self.matcher.generate_cover_letter(1)
        
        # Check the error message
        self.assertIn("Error generating cover letter", result)
        self.assertIn("Failed to connect to Ollama", result)
        self.assertIn("check that Ollama is downloaded", result)
    
    @patch("ollama.generate")
    def test_ollama_model_not_found(self, mock_generate):
        """Test handling of Ollama model not found error in cover letter generation."""
        # Simulate model not found error
        mock_generate.side_effect = Exception("model not found")
        
        # Call the function under test
        result = self.matcher.generate_cover_letter(1)
        
        # Check the error message
        self.assertIn("Model 'test-model' not found in Ollama", result)
    
    @patch("ollama.generate")
    def test_short_output_warning(self, mock_generate):
        """Test handling of unusually short LLM output."""
        # Mock a very short response
        mock_response = MagicMock()
        mock_response.__iter__.return_value = [("response", "Too short")]
        mock_generate.return_value = mock_response
        
        # Call the function under test
        with patch("builtins.print") as mock_print:
            result = self.matcher.generate_cover_letter(1)
            
            # Check that warning was printed
            mock_print.assert_any_call("Warning: Generated cover letter is unusually short or empty.")
            
        # Check the error message
        self.assertIn("Error: Failed to generate a proper cover letter", result)
        self.assertIn("output was too short", result)
    
    @patch("ollama.generate")
    def test_error_in_response(self, mock_generate):
        """Test handling of error message in LLM response."""
        # Mock an error message response
        mock_response = MagicMock()
        mock_response.__iter__.return_value = [("response", "Error: Unable to generate content")]
        mock_generate.return_value = mock_response
        
        # Call the function under test
        result = self.matcher.generate_cover_letter(1)
        
        # Check that the error was returned in the result string
        # Our implementation detects this first as a short response
        self.assertIn("Error: Failed to generate a proper cover letter", result)
        self.assertIn("The output was too short", result)
    
    def test_report_directory_creation(self):
        """Test automatic creation of reports directory."""
        # Create a matcher with modified REPORTS_DIR constant
        original_reports_dir = "reports"
        test_reports_dir = os.path.join(self.test_dir, "test_reports")
        
        # Patch the REPORTS_DIR constant
        with patch("src.core.content_matcher.REPORTS_DIR", test_reports_dir):
            # Create a new matcher
            matcher = ContentMatcher(
                jobs_file=self.jobs_file,
                content_file=self.content_file
            )
            
            # Mock the generate_markdown_report method to avoid actual processing
            with patch.object(matcher, "generate_markdown_report", return_value="Test Report"):
                # Call save_markdown_report
                matcher.save_markdown_report(1)
                
                # Check that directory was created
                self.assertTrue(os.path.exists(test_reports_dir))


if __name__ == "__main__":
    unittest.main()
