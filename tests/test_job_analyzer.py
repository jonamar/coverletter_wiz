"""
Unit tests for the JobAnalyzer module.

These tests validate the error handling and core functionality
of the JobAnalyzer class, including input validation, LLM interaction,
and tag analysis.
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

from src.core.job_analyzer import JobAnalyzer


class TestJobAnalyzer(unittest.TestCase):
    """Test cases for the JobAnalyzer class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a mock categories file
        self.categories_path = os.path.join(self.test_dir, "categories.yaml")
        with open(self.categories_path, "w") as f:
            f.write("""
categories:
  - name: technical
    weight: 1.5
    keywords:
      - python
      - programming
  - name: soft_skills
    weight: 1.2
    keywords:
      - communication
      - teamwork
            """)
            
        # Set up mock jobs data
        self.jobs_data = {
            "jobs": [],
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat()
            }
        }
        
        # Create a jobs file
        self.jobs_file = os.path.join(self.test_dir, "analyzed_jobs.json")
        with open(self.jobs_file, "w") as f:
            json.dump(self.jobs_data, f)
            
        # Create the analyzer with our test files
        # Using _load_categories = lambda x=None: ... to override loading from file
        with patch.object(JobAnalyzer, '_load_categories', return_value={
            'categories': [
                {'name': 'technical', 'weight': 1.5, 'keywords': ['python', 'programming']},
                {'name': 'soft_skills', 'weight': 1.2, 'keywords': ['communication', 'teamwork']}
            ]
        }):
            self.analyzer = JobAnalyzer(
                output_file=self.jobs_file,
                llm_model="test-model"
            )
        
    def tearDown(self):
        """Clean up after each test."""
        # Remove test files and directory
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_empty_job_text(self):
        """Test handling of empty job text."""
        result = self.analyzer.analyze_job_posting("http://test.com", "")
        self.assertIsNone(result)
    
    def test_short_job_text(self):
        """Test handling of job text that is too short."""
        with patch("builtins.print") as mock_print:
            result = self.analyzer.analyze_job_posting("http://test.com", "Short text")
            # Should print a warning but still attempt analysis
            mock_print.assert_any_call("Warning: Job text is very short, analysis may be incomplete.")
            # Result should be None since the ollama call would fail in a real test
            self.assertIsNone(result)
    
    @patch("ollama.generate")
    def test_ollama_connection_error(self, mock_generate):
        """Test handling of Ollama connection error."""
        # Simulate connection refused error
        mock_generate.side_effect = Exception("connection refused")
        
        # Call the function under test and check the result
        with patch("builtins.print") as mock_print:
            result = self.analyzer.analyze_job_posting("http://test.com", "This is a job posting text")
            
            # Check that appropriate error was printed
            mock_print.assert_any_call("Error analyzing job posting: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download")
            
            # Result should be None
            self.assertIsNone(result)
    
    @patch("ollama.generate")
    def test_ollama_model_not_found(self, mock_generate):
        """Test handling of Ollama model not found error."""
        # Simulate model not found error
        mock_generate.side_effect = Exception("model not found")
        
        # Call the function under test and check the result
        with patch("builtins.print") as mock_print:
            result = self.analyzer.analyze_job_posting("http://test.com", "This is a job posting text")
            
            # Check that appropriate error was printed
            mock_print.assert_any_call(f"Error analyzing job posting: Model 'test-model' not found in Ollama. Please check available models with 'ollama list' or download this model with 'ollama pull test-model'.")
            
            # Result should be None
            self.assertIsNone(result)
    
    @patch("ollama.generate")
    def test_url_fallback_extraction(self, mock_generate):
        """Test fallback extraction of organization name from URL."""
        # Mock LLM response with no org/title extraction
        mock_response = MagicMock()
        mock_response.__iter__.return_value = [("response", "Some LLM output without ORG or TITLE")]
        mock_generate.return_value = mock_response
        
        # Mock tag analysis to return empty dict
        with patch.object(self.analyzer, "_analyze_tags", return_value={
            "high_priority": [], "medium_priority": [], "low_priority": []
        }):
            # Test with a lever.co URL
            with patch("builtins.print"):  # Suppress prints
                result = self.analyzer.analyze_job_posting(
                    "https://jobs.lever.co/signal/job-id", 
                    "This is a job posting text"
                )
                
                # Check that org name was extracted from URL
                self.assertEqual(result["org_name"], "Signal")
    
    @patch("ollama.generate")
    def test_empty_tags_warning(self, mock_generate):
        """Test warning when no tags are extracted."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.__iter__.return_value = [("response", "ORG: TestOrg\nTITLE: TestJob\nSUMMARY: Test summary")]
        mock_generate.return_value = mock_response
        
        # Mock tag analysis to return empty dict
        with patch.object(self.analyzer, "_analyze_tags", return_value={
            "high_priority": [], "medium_priority": [], "low_priority": []
        }):
            with patch("builtins.print") as mock_print:
                result = self.analyzer.analyze_job_posting("http://test.com", "This is a job posting text")
                
                # Check that warning was printed
                mock_print.assert_any_call("Warning: No tags were extracted from the job posting.")


if __name__ == "__main__":
    unittest.main()
