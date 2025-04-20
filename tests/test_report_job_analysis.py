"""
Unit tests for the job analysis functionality in the report module.

These tests validate the job URL fetching and analysis functionality
added to the generate_report module, ensuring it correctly fetches,
analyzes, and stores job data.
"""

import os
import sys
import unittest
import json
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli.generate_report import fetch_and_analyze_job, analyze_job_description


class TestReportJobAnalysis(unittest.TestCase):
    """Test cases for job analysis functionality in the report module."""
    
    def setUp(self) -> None:
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Sample HTML content for tests
        self.sample_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Test Job</title></head>
        <body>
            <div id="job-description">
                <h1>Senior Python Developer</h1>
                <p>TechCorp is looking for a skilled Python developer.</p>
                <h2>Requirements:</h2>
                <ul>
                    <li>5+ years Python experience</li>
                    <li>Strong knowledge of web frameworks</li>
                    <li>Experience with data analysis</li>
                </ul>
            </div>
        </body>
        </html>
        """
    
    def tearDown(self) -> None:
        """Clean up after each test."""
        import shutil
        shutil.rmtree(self.test_dir)
    
    @patch("requests.get")
    def test_fetch_and_analyze_job_success(self, mock_get: MagicMock) -> None:
        """Test successful fetching and analyzing of a job URL."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = self.sample_html
        mock_get.return_value = mock_response
        
        # Mock the analyze_job_description function
        with patch("src.cli.generate_report.analyze_job_description") as mock_analyze:
            mock_analyze.return_value = (
                "Senior Python Developer", 
                "TechCorp", 
                "Python developer position requiring 5+ years experience.",
                "Sample job text",
                {
                    "high_priority": ["python", "web frameworks"],
                    "medium_priority": ["data analysis"],
                    "low_priority": []
                }
            )
            
            # Call function under test
            result = fetch_and_analyze_job("https://example.com/job", "test-model")
            
            # Assertions
            self.assertIsNotNone(result)
            self.assertEqual(result["job_title"], "Senior Python Developer")
            self.assertEqual(result["org_name"], "TechCorp")
            self.assertEqual(result["summary"], "Python developer position requiring 5+ years experience.")
            self.assertEqual(result["url"], "https://example.com/job")
            self.assertEqual(result["raw_text"], "Sample job text")
            self.assertEqual(result["tags"]["high_priority"], ["python", "web frameworks"])
    
    @patch("requests.get")
    def test_fetch_and_analyze_job_request_failure(self, mock_get: MagicMock) -> None:
        """Test handling of request failure in fetch_and_analyze_job."""
        # Setup mock response for a failed request
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Call function under test
        result = fetch_and_analyze_job("https://example.com/nonexistent", "test-model")
        
        # Assertions
        self.assertIsNone(result)
    
    @patch("requests.get")
    def test_fetch_and_analyze_job_exception(self, mock_get: MagicMock) -> None:
        """Test handling of exceptions in fetch_and_analyze_job."""
        # Setup mock to raise an exception
        mock_get.side_effect = Exception("Network error")
        
        # Call function under test
        result = fetch_and_analyze_job("https://example.com/job", "test-model")
        
        # Assertions
        self.assertIsNone(result)
    
    @patch("ollama.generate")
    def test_analyze_job_description_success(self, mock_generate: MagicMock) -> None:
        """Test successful analysis of a job description."""
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.__iter__.return_value = [(
            "response", 
            "ORG: TechCorp\nTITLE: Senior Python Developer\nSUMMARY: A senior developer role focused on Python development."
        )]
        mock_generate.return_value = mock_response
        
        # Mock the prioritize_tags_for_job function
        with patch("src.cli.generate_report.prioritize_tags_for_job") as mock_prioritize:
            mock_prioritize.return_value = {
                "high_priority": ["python", "web frameworks"],
                "medium_priority": ["data analysis"],
                "low_priority": []
            }
            
            # Mock the yaml loading since our test environment won't have the categories file
            with patch("yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"categories": ["python", "web frameworks", "data analysis"]}
                
                # Call function under test
                job_title, org_name, summary, processed_text, tags = analyze_job_description(
                    "Sample job text describing a Python developer position."
                )
                
                # Assertions
                self.assertEqual(job_title, "Senior Python Developer")
                self.assertEqual(org_name, "TechCorp")
                self.assertEqual(summary, "A senior developer role focused on Python development.")
                self.assertEqual(tags["high_priority"], ["python", "web frameworks"])
    
    @patch("ollama.generate")
    def test_analyze_job_description_llm_failure(self, mock_generate: MagicMock) -> None:
        """Test handling of LLM failure in analyze_job_description."""
        # Mock LLM to raise an exception
        mock_generate.side_effect = Exception("LLM error")
        
        # Mock prioritize_tags_for_job to return empty tags
        with patch("src.cli.generate_report.prioritize_tags_for_job") as mock_prioritize:
            mock_prioritize.return_value = {
                "high_priority": [],
                "medium_priority": [],
                "low_priority": []
            }
            
            # Mock the yaml loading
            with patch("yaml.safe_load") as mock_yaml:
                mock_yaml.return_value = {"categories": []}
                
                # Also mock preprocess_job_text to return the input text
                with patch("src.cli.generate_report.preprocess_job_text") as mock_preprocess:
                    mock_preprocess.return_value = "Sample job text."
                    
                    # Call function under test
                    job_title, org_name, summary, processed_text, tags = analyze_job_description(
                        "Sample job text."
                    )
                    
                    # Assertions - should use fallback values when LLM fails
                    self.assertEqual(job_title, "Unknown Position")
                    self.assertEqual(org_name, "Unknown Organization")
                    self.assertEqual(summary, "No summary available")
                    self.assertEqual(processed_text, "Sample job text.")
                    self.assertEqual(tags["high_priority"], [])


if __name__ == "__main__":
    unittest.main()
