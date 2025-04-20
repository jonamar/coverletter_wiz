"""
Tests related to report generation.

Note: URL-based job analysis functionality has been removed from the codebase.
This test file is kept as a placeholder to document the intentional removal of features.
"""

import os
import sys
import unittest
import json
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional, List

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli.generate_report import generate_report


class TestReportIntegration(unittest.TestCase):
    """Test cases for report generation after URL functionality was removed."""
    
    def setUp(self) -> None:
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.reports_dir = os.path.join(self.test_dir, "reports")
        
        # Create reports directory
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Setup patches to avoid real file access outside our test dir
        self.patch_reports_dir = patch("src.cli.generate_report.REPORTS_DIR", self.reports_dir)
        self.patch_reports_dir.start()
        
        self.patch_categories_file = patch("src.cli.generate_report.CATEGORIES_FILE", 
                                          os.path.join(self.test_dir, "categories.yaml"))
        self.patch_categories_file.start()
        
        # Create categories file
        with open(os.path.join(self.test_dir, "categories.yaml"), "w") as f:
            f.write("""
python:
  - django
  - flask
api:
  - rest
  - graphql
            """)
            
        # Create default jobs file path
        self.patch_default_jobs_file = patch("src.cli.generate_report.DEFAULT_JOBS_FILE", 
                                           os.path.join(self.test_dir, "jobs.json"))
        self.patch_default_jobs_file.start()
        
        # Create default content file path
        self.patch_default_content_file = patch("src.cli.generate_report.DEFAULT_CONTENT_FILE", 
                                              os.path.join(self.test_dir, "content.json"))
        self.patch_default_content_file.start()
        
        # Create sample job data
        self.sample_job_data = {
            "jobs": [
                {
                    "id": "1",
                    "job_title": "Existing Python Developer",
                    "org_name": "ExistingCorp",
                    "url": "https://existing.com/job",
                    "summary": "Existing job summary",
                    "raw_text": "Detailed job description for existing job.",
                    "tags": {
                        "high_priority": ["python", "django"],
                        "medium_priority": ["api", "database"],
                        "low_priority": ["agile", "testing"]
                    }
                }
            ]
        }
        
        # Write sample job data to file
        with open(os.path.join(self.test_dir, "jobs.json"), "w") as f:
            json.dump(self.sample_job_data, f)
        
        # Create sample content data
        self.sample_content_data = {
            "document1": {
                "content": {
                    "paragraphs": [
                        {
                            "text": "I have experience with Python and Django development.",
                            "tags": ["python", "django"],
                            "ratings": {"average": 8.5}
                        },
                        {
                            "text": "I built RESTful APIs with Django Rest Framework.",
                            "tags": ["api", "django"],
                            "ratings": {"average": 7.5}
                        }
                    ]
                }
            }
        }
        
        # Write sample content data to file
        with open(os.path.join(self.test_dir, "content.json"), "w") as f:
            json.dump(self.sample_content_data, f)
    
    def tearDown(self) -> None:
        """Clean up after each test."""
        self.patch_reports_dir.stop()
        self.patch_categories_file.stop()
        self.patch_default_jobs_file.stop()
        self.patch_default_content_file.stop()
        
        import shutil
        shutil.rmtree(self.test_dir)
    
    @patch("src.cli.generate_report.ollama.generate")
    def test_generate_report_with_config(self, mock_generate: MagicMock) -> None:
        """Test generating a report with a custom configuration."""
        # Mock ollama.generate to avoid actual LLM calls
        mock_generate.return_value = MagicMock()
        
        # Test configuration
        test_config = {
            "min_rating": 5.0,
            "content_weight": 0.5,
            "weights": [4, 3, 2, 0.2],
            "similarity_threshold": 0.7,
            "include_cover_letter": False,
            "use_semantic_dedup": True
        }
        
        # Run the function with the configuration
        report_path = generate_report(
            job_id="1",  # Use the existing job ID
            config=test_config
        )
        
        # Assertions
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
        
        # Verify that the report was generated
        with open(report_path, "r") as f:
            report_content = f.read()
            self.assertIn("Python Developer", report_content)  # Job title
            self.assertIn("ExistingCorp", report_content)  # Organization


if __name__ == "__main__":
    unittest.main()
