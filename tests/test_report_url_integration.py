"""
Unit tests for the report generation with job URLs.

These tests validate the integration of job URL fetching and analysis
with the report generation functionality.
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


class TestReportUrlIntegration(unittest.TestCase):
    """Test cases for the integration of job URL fetching in report generation."""
    
    def setUp(self) -> None:
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.jobs_file = os.path.join(self.test_dir, "jobs.json")
        self.content_file = os.path.join(self.test_dir, "content.json")
        self.reports_dir = os.path.join(self.test_dir, "reports")
        
        # Create jobs directory
        os.makedirs(os.path.dirname(self.jobs_file), exist_ok=True)
        
        # Create reports directory
        os.makedirs(self.reports_dir, exist_ok=True)
        
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
        with open(self.jobs_file, "w") as f:
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
        with open(self.content_file, "w") as f:
            json.dump(self.sample_content_data, f)
        
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
    
    def tearDown(self) -> None:
        """Clean up after each test."""
        self.patch_reports_dir.stop()
        self.patch_categories_file.stop()
        
        import shutil
        shutil.rmtree(self.test_dir)
    
    @patch("src.cli.generate_report.fetch_and_analyze_job")
    def test_generate_report_with_new_job_url(self, mock_fetch: MagicMock) -> None:
        """Test generating a report with a new job URL that needs to be fetched and analyzed."""
        # Setup the mock to return a sample job
        mock_fetch.return_value = {
            "id": "new_job_id",
            "job_title": "New Python Developer",
            "org_name": "NewCorp",
            "url": "https://new.com/job",
            "summary": "New job summary",
            "raw_text": "Detailed job description for new job.",
            "tags": {
                "high_priority": ["python", "flask"],
                "medium_priority": ["api", "database"],
                "low_priority": ["agile", "docker"]
            }
        }
        
        # Mock other functions to avoid external calls
        with patch("src.cli.generate_report.ollama.generate") as mock_generate:
            mock_generate.return_value = MagicMock()
            
            # Run the function with a new URL
            report_path = generate_report(
                job_url="https://new.com/job",
                include_cover_letter=False,  # Skip cover letter to simplify test
                jobs_file=self.jobs_file,
                content_file=self.content_file
            )
            
            # Assertions
            self.assertIsNotNone(report_path)
            self.assertTrue(os.path.exists(report_path))
            
            # Verify that fetch_and_analyze_job was called with the right URL
            mock_fetch.assert_called_once_with("https://new.com/job", "gemma3:12b")
            
            # Check if the job was saved to the jobs file
            with open(self.jobs_file, "r") as f:
                jobs_data = json.load(f)
                # Should now have 2 jobs (original + new)
                self.assertEqual(len(jobs_data["jobs"]), 2)
                # Check if new job is in the list
                new_job_found = any(j["url"] == "https://new.com/job" for j in jobs_data["jobs"])
                self.assertTrue(new_job_found)
    
    @patch("src.cli.generate_report.fetch_and_analyze_job")
    def test_generate_report_with_existing_job_url(self, mock_fetch: MagicMock) -> None:
        """Test generating a report with an existing job URL that's already in the database."""
        # Run the function with an existing URL
        report_path = generate_report(
            job_url="https://existing.com/job",
            include_cover_letter=False,  # Skip cover letter to simplify test
            jobs_file=self.jobs_file,
            content_file=self.content_file
        )
        
        # Assertions
        self.assertIsNotNone(report_path)
        self.assertTrue(os.path.exists(report_path))
        
        # Verify fetch_and_analyze_job was NOT called since job already exists
        mock_fetch.assert_not_called()
    
    @patch("src.cli.generate_report.fetch_and_analyze_job")
    def test_generate_report_with_failed_fetch(self, mock_fetch: MagicMock) -> None:
        """Test handling of failed fetching during report generation."""
        # Setup mock to return None, simulating a failed fetch
        mock_fetch.return_value = None
        
        # Run the function with a URL that will fail to fetch
        report_path = generate_report(
            job_url="https://fail.com/job",
            include_cover_letter=False,
            jobs_file=self.jobs_file,
            content_file=self.content_file
        )
        
        # Assertions
        self.assertIsNone(report_path)  # Should return None on failure
        
        # Verify fetch_and_analyze_job was called but failed
        mock_fetch.assert_called_once()
        
        # Check that no new job was added to the jobs file
        with open(self.jobs_file, "r") as f:
            jobs_data = json.load(f)
            # Should still have only the original job
            self.assertEqual(len(jobs_data["jobs"]), 1)


if __name__ == "__main__":
    unittest.main()
