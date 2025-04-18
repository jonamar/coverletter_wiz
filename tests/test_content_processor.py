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
        nonexistent_file = os.path.join(self.test_dir, "nonexistent.json")
        
        # Patch print to capture output
        with patch("builtins.print") as mock_print:
            processor = ContentProcessor(nonexistent_file)
            
            # Verify the processor was initialized properly despite missing file
            self.assertIsInstance(processor, ContentProcessor)
            
            # Check if the appropriate error message was printed
            mock_print.assert_any_call(f"Notice: Content file {nonexistent_file} does not exist.")
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON file."""
        # Create an invalid JSON file
        with open(self.content_file, "w") as f:
            f.write("{invalid: json")
        
        # Patch print to capture output
        with patch("builtins.print") as mock_print:
            processor = ContentProcessor(self.content_file)
            
            # Verify the processor was initialized despite invalid JSON
            self.assertIsInstance(processor, ContentProcessor)
            
            # Check if the appropriate error message was printed
            for call in mock_print.call_args_list:
                if "invalid JSON" in str(call) and self.content_file in str(call):
                    break
            else:
                self.fail("Expected error message about invalid JSON not found in print calls")
    
    def test_permission_denied_reading(self):
        """Test handling of permission denied when reading file."""
        # Create a file with no read permissions
        with open(self.content_file, "w") as f:
            f.write("{}")
        os.chmod(self.content_file, 0o000)  # No permissions
        
        try:
            # Patch print to capture output
            with patch("builtins.print") as mock_print:
                processor = ContentProcessor(self.content_file)
                
                # Verify the processor was initialized despite permission error
                self.assertIsInstance(processor, ContentProcessor)
                
                # Check if the appropriate error message was printed
                found_error = False
                for call in mock_print.call_args_list:
                    if "permission denied" in str(call).lower() and self.content_file in str(call):
                        found_error = True
                        break
                
                self.assertTrue(found_error, "Expected error message about permission denied not found in print calls")
        finally:
            os.chmod(self.content_file, 0o644)  # Reset permissions
    
    def test_permission_denied_writing(self):
        """Test handling of permission denied when saving file."""
        # Create an empty content processor
        processor = ContentProcessor(self.content_file)
        
        # Make the directory read-only
        os.chmod(os.path.dirname(self.content_file), 0o500)  # Read-only directory
        
        try:
            # Patch print to capture output and open to simulate permission error
            with patch("builtins.print") as mock_print, \
                 patch("builtins.open", side_effect=PermissionError("Permission denied")):
                 
                # Attempt to save ratings
                result = processor._save_ratings()
                
                # Verify the save failed
                self.assertFalse(result)
                
                # Check if the appropriate error message was printed
                found_error = False
                for call in mock_print.call_args_list:
                    if "permission denied" in str(call).lower():
                        found_error = True
                        break
                
                self.assertTrue(found_error, "Expected error message about permission denied not found in print calls")
        finally:
            os.chmod(os.path.dirname(self.content_file), 0o755)  # Reset directory permissions
    
    def test_disk_space_error(self):
        """Test handling of disk space error when saving."""
        # Create an empty content processor
        processor = ContentProcessor(self.content_file)
        
        # Patch print to capture output and open to simulate disk space error
        with patch("builtins.print") as mock_print, \
             patch("builtins.open", side_effect=OSError("No space left on device")):
             
            # Attempt to save ratings
            result = processor._save_ratings()
            
            # Verify the save failed
            self.assertFalse(result)
            
            # Check if the appropriate error message was printed
            found_error = False
            for call in mock_print.call_args_list:
                if "error saving" in str(call).lower():
                    found_error = True
                    break
            
            self.assertTrue(found_error, "Expected error message about disk space not found in print calls")
    
    def test_empty_content_handling(self):
        """Test handling of empty content file."""
        # Create an empty content file
        with open(self.content_file, "w") as f:
            f.write("{}")
        
        # Patch print to capture output
        with patch("builtins.print") as mock_print:
            processor = ContentProcessor(self.content_file)
            
            # Verify the processor was initialized with empty data
            self.assertIsInstance(processor, ContentProcessor)
            
            # Check for warning about empty content
            found_warning = False
            for call in mock_print.call_args_list:
                if "empty" in str(call).lower() and self.content_file in str(call):
                    found_warning = True
                    break
            
            self.assertTrue(found_warning, "Expected warning about empty content not found in print calls")
    
    def test_edit_block_functionality(self):
        """Test the edit block functionality."""
        # Create the content file
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
        
        # Create processor
        processor = ContentProcessor(self.content_file)
        
        # Test the _edit_block method directly
        test_block = {
            "text": "Original text for testing",
            "tags": ["test", "original"],
            "rating": 7.5
        }
        
        # Mock the input function to simulate user editing the block
        with patch("builtins.input", side_effect=["Edited text for testing"]):
            edited_block = processor._edit_block(test_block)
            
            # Check that the edited block has the new text
            self.assertEqual(edited_block["text"], "Edited text for testing")
            
            # Check that tags were preserved
            self.assertEqual(edited_block["tags"], ["test", "original"])
            
            # Check that rating was preserved
            self.assertEqual(edited_block["rating"], 7.5)
            
            # Check that edit metadata was added
            self.assertEqual(edited_block["edited_from"], "Original text for testing")
            self.assertTrue("edit_date" in edited_block)
    
    def test_tournament_edit_functionality(self):
        """Test the edit functionality in tournament mode."""
        # Create a more complex content file with multiple blocks for tournament testing
        tournament_data = {
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
                                    "tags": ["python", "programming", "skills_competencies"]
                                },
                                {
                                    "text": "I have worked on many successful projects.",
                                    "rating": 7.0,
                                    "tags": ["experience", "projects", "skills_competencies"]
                                }
                            ]
                        }
                    ]
                }
            }
        }
        
        # Create the content file
        with open(self.content_file, "w") as f:
            json.dump(tournament_data, f)
        
        # Create processor
        processor = ContentProcessor(self.content_file)
        
        # Test the edit functionality directly
        test_block = {
            "text": "Original tournament text",
            "tags": ["test", "tournament"],
            "rating": 7.0
        }
        
        # Mock the input function to simulate user editing the block
        with patch("builtins.input", return_value="Edited tournament block"):
            edited_block = processor._edit_block(test_block)
            
            # Check that the edited block has the new text
            self.assertEqual(edited_block["text"], "Edited tournament block")
            
            # Check that edit metadata was added
            self.assertEqual(edited_block["edited_from"], "Original tournament text")
            self.assertTrue("edit_date" in edited_block)


if __name__ == "__main__":
    unittest.main()
