"""
Unit tests for the ContentProcessor module.

These tests validate the error handling and core functionality
of the ContentProcessor class, particularly focusing on file I/O
operations and robust error handling.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import tempfile
from datetime import datetime

# Add parent directory to path to import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.content_processor import ContentProcessor


class TestContentProcessor(unittest.TestCase):
    """Test cases for the ContentProcessor class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create content file path
        self.content_file = os.path.join(self.test_dir, "content.json")
        
        # Sample content data
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
    
    def tearDown(self):
        """Clean up after each test."""
        # Remove test files and directory
        import shutil
        shutil.rmtree(self.test_dir)
    
    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent file."""
        # Use a path that definitely doesn't exist
        nonexistent_file = os.path.join(self.test_dir, "nonexistent.json")
        
        # Create processor with nonexistent file
        with patch("builtins.print") as mock_print:
            processor = ContentProcessor(nonexistent_file)
            
            # Check that appropriate message was printed
            mock_print.assert_any_call(f"Error: Content file {nonexistent_file} does not exist.")
            mock_print.assert_any_call("Creating empty content structure.")
            
            # Check that an empty structure was created
            self.assertTrue("metadata" in processor.data)
            self.assertTrue("version" in processor.data["metadata"])
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON file."""
        # Create a file with invalid JSON
        with open(self.content_file, "w") as f:
            f.write("This is not valid JSON")
        
        # Create processor with invalid JSON file
        with patch("builtins.print") as mock_print:
            processor = ContentProcessor(self.content_file)
            
            # Check that appropriate error was printed
            # The print includes the JSONDecodeError message which varies, so we check for partial match
            for call in mock_print.call_args_list:
                args, _ = call
                if len(args) > 0 and isinstance(args[0], str) and args[0].startswith(f"Error: Content file {self.content_file} contains invalid JSON:"):
                    break
            else:
                self.fail("Expected error message about invalid JSON not found in print calls")
            
            # Check that an empty structure was created
            self.assertTrue("metadata" in processor.data)
            self.assertTrue("error" in processor.data["metadata"])
            self.assertEqual(processor.data["metadata"]["error"], "invalid_json")
    
    def test_permission_denied_reading(self):
        """Test handling of permission denied when reading file."""
        # Create the content file
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
        
        # Mock os.access to simulate permission denied
        with patch("os.access", return_value=False):
            with patch("builtins.print") as mock_print:
                processor = ContentProcessor(self.content_file)
                
                # Check that appropriate error was printed
                mock_print.assert_any_call(f"Error: No read permission for {self.content_file}.")
                
                # Check that an error structure was created
                self.assertEqual(processor.data["metadata"]["error"], "permission_denied")
    
    def test_permission_denied_writing(self):
        """Test handling of permission denied when saving file."""
        # Create the content file
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
        
        # Create processor
        processor = ContentProcessor(self.content_file)
        
        # Mock os.access to simulate permission denied for writing
        with patch("os.access", return_value=False):
            with patch("builtins.print") as mock_print:
                result = processor._save_ratings()
                
                # Check that appropriate error was printed
                mock_print.assert_any_call(f"Error: No write permission for {self.content_file}.")
                
                # Check that save failed
                self.assertFalse(result)
    
    def test_disk_space_error(self):
        """Test handling of disk space error when saving."""
        # Create the content file
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
        
        # Create processor
        processor = ContentProcessor(self.content_file)
        
        # Mock open to raise OSError simulating no disk space
        m = mock_open()
        m.side_effect = OSError("No space left on device")
        
        with patch("builtins.open", m):
            with patch("builtins.print") as mock_print:
                result = processor._save_ratings()
                
                # Check that appropriate error was printed
                mock_print.assert_any_call(f"Error: Not enough disk space to save to {self.content_file}")
                
                # Check that save failed
                self.assertFalse(result)
    
    def test_empty_content_handling(self):
        """Test handling of empty content file."""
        # Create an empty content file
        with open(self.content_file, "w") as f:
            json.dump({}, f)
        
        # Create processor with empty file
        with patch("builtins.print") as mock_print:
            processor = ContentProcessor(self.content_file)
            
            # Check that appropriate message was printed
            mock_print.assert_any_call(f"Warning: Content file {self.content_file} is empty.")
            
            # Check that a valid structure was created
            self.assertTrue("metadata" in processor.data)
            self.assertTrue("version" in processor.data["metadata"])


if __name__ == "__main__":
    unittest.main()
