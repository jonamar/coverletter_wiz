#!/usr/bin/env python3
"""
Test to ensure data privacy by checking that no JSON files with personal data
are stored in the main application repository.
"""

import os
import unittest
import json
from pathlib import Path

# Get the repository directory
REPO_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = Path(REPO_DIR).parent / "coverletter_data"

class TestDataPrivacy(unittest.TestCase):
    """Tests to ensure data privacy in the repository."""
    
    def test_no_json_data_in_main_repo(self):
        """
        Test that no JSON files containing personal data are stored in the main repository.
        
        This test ensures that all personal data is stored in the data repository,
        not in the main application repository.
        """
        # List of patterns that might indicate personal data in JSON files
        personal_data_patterns = [
            "job", "jobs", "content", "cover", "letter", "report", "rating",
            "analyzed", "processed", "personal", "private", "user", "profile"
        ]
        
        # Directories to exclude from the check
        exclude_dirs = [
            ".git",
            "__pycache__",
            ".venv",
            "tests/fixtures"  # Allow JSON fixtures for tests
        ]
        
        # Find all JSON files in the repository
        json_files = []
        for root, dirs, files in os.walk(REPO_DIR):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file.endswith('.json'):
                    json_path = os.path.join(root, file)
                    json_files.append(json_path)
        
        # Check each JSON file for potential personal data
        privacy_violations = []
        
        for json_file in json_files:
            try:
                # Check if the filename contains any personal data patterns
                filename = os.path.basename(json_file)
                if any(pattern in filename.lower() for pattern in personal_data_patterns):
                    privacy_violations.append(f"Potential personal data in filename: {json_file}")
                    continue
                
                # Check the content of the JSON file
                with open(json_file, 'r') as f:
                    try:
                        data = json.load(f)
                        
                        # Check if the JSON structure might contain personal data
                        if isinstance(data, dict):
                            keys = ' '.join(data.keys()).lower()
                            if any(pattern in keys for pattern in personal_data_patterns):
                                privacy_violations.append(f"Potential personal data in JSON structure: {json_file}")
                                continue
                            
                            # Check for job data
                            if 'jobs' in data and isinstance(data['jobs'], list) and len(data['jobs']) > 0:
                                privacy_violations.append(f"Job data found in: {json_file}")
                                continue
                            
                            # Check for content data
                            if 'content' in data and isinstance(data['content'], dict):
                                privacy_violations.append(f"Content data found in: {json_file}")
                                continue
                    except json.JSONDecodeError:
                        # Not valid JSON, skip
                        pass
            except Exception as e:
                # Skip files that can't be read
                pass
        
        # Assert that there are no privacy violations
        self.assertEqual(len(privacy_violations), 0, 
                         f"Found {len(privacy_violations)} potential privacy violations:\n" + 
                         "\n".join(privacy_violations))
        
        # Print success message
        if not privacy_violations:
            print(f"✅ No personal data JSON files found in the main repository.")

if __name__ == '__main__':
    unittest.main()
