#!/usr/bin/env python3
"""
Unit tests for the DataManager class.

These tests validate that the DataManager provides correct functionality
for data access, content operations, and error handling.
"""

import unittest
import os
import json
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, mock_open

from src.core.data_manager import DataManager

class TestDataManager(unittest.TestCase):
    """Test cases for the DataManager class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary test directory
        self.temp_dir = tempfile.mkdtemp()
        self.json_dir = os.path.join(self.temp_dir, "json")
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Create test content file path
        self.content_file = os.path.join(self.json_dir, "test_content.json")
        
        # Sample content data for testing
        self.sample_content = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat()
            },
            "test-file.md": {
                "filename": "test-file.md",
                "last_modified": datetime.now().isoformat(),
                "processed_date": datetime.now().isoformat(),
                "content": {
                    "paragraphs": [
                        {
                            "text": "This is a sample paragraph.",
                            "sentences": [
                                {
                                    "text": "This is a sample sentence.",
                                    "rating": 7.5,
                                    "tags": ["sample", "test"],
                                    "batch_rating": True
                                },
                                {
                                    "text": "This is another sample sentence.",
                                    "rating": 5.0,
                                    "tags": ["sample", "test"],
                                    "batch_rating": False
                                }
                            ]
                        }
                    ]
                }
            }
        }
        
        # Reset DataManager singleton between tests
        DataManager._instance = None
    
    def tearDown(self):
        """Clean up after each test."""
        # Clean up test directory
        shutil.rmtree(self.temp_dir)
        
        # Reset DataManager singleton after tests
        DataManager._instance = None
    
    def test_singleton_pattern(self):
        """Test that DataManager implements the singleton pattern correctly."""
        # Create two instances with the same file
        dm1 = DataManager(content_file=self.content_file)
        dm2 = DataManager(content_file=self.content_file)
        
        # They should be the same object
        self.assertIs(dm1, dm2)
        
        # Even with a different file parameter (since the singleton is created)
        dm3 = DataManager(content_file=os.path.join(self.json_dir, "other_file.json"))
        self.assertIs(dm1, dm3)
        
        # Reset the singleton to test different file initialization
        DataManager._instance = None
        
        # Now we should get a different instance with different file
        dm4 = DataManager(content_file=os.path.join(self.json_dir, "other_file.json"))
        self.assertEqual(dm4.content_file, os.path.join(self.json_dir, "other_file.json"))
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file creates default structure."""
        # The file doesn't exist yet
        self.assertFalse(os.path.exists(self.content_file))
        
        # Create DataManager with non-existent file
        dm = DataManager(content_file=self.content_file)
        
        # It should create a default data structure
        self.assertIsInstance(dm.data, dict)
        self.assertIn("metadata", dm.data)
        self.assertIn("version", dm.data["metadata"])
    
    def test_save_and_load_data(self):
        """Test saving and loading data works correctly."""
        # Create a DataManager and set data
        dm = DataManager(content_file=self.content_file)
        dm.data = self.sample_content
        
        # Save the data
        result = dm.save_data()
        self.assertTrue(result)
        
        # Verify the file was created
        self.assertTrue(os.path.exists(self.content_file))
        
        # Reset the singleton
        DataManager._instance = None
        
        # Create a new DataManager and load the data
        dm2 = DataManager(content_file=self.content_file)
        
        # Verify the data was loaded correctly
        self.assertEqual(dm2.data["metadata"]["version"], "1.0")
        self.assertIn("test-file.md", dm2.data)
    
    def test_get_content_blocks(self):
        """Test extracting content blocks from the data."""
        # Create a DataManager with sample data
        dm = DataManager(content_file=self.content_file)
        dm.data = self.sample_content
        
        # Get content blocks
        blocks = dm.get_content_blocks()
        
        # Verify blocks were extracted correctly
        self.assertEqual(len(blocks), 2)
        
        # Check first block
        self.assertEqual(blocks[0]["text"], "This is a sample sentence.")
        self.assertEqual(blocks[0]["rating"], 7.5)
        self.assertIn("sample", blocks[0]["tags"])
        self.assertEqual(blocks[0]["batch_rating"], True)
        
        # Check second block
        self.assertEqual(blocks[1]["text"], "This is another sample sentence.")
        self.assertEqual(blocks[1]["rating"], 5.0)
    
    def test_update_ratings(self):
        """Test updating ratings in the data."""
        # Create a DataManager with sample data
        dm = DataManager(content_file=self.content_file)
        dm.data = self.sample_content
        
        # Create updated blocks
        updated_blocks = [
            {
                "text": "This is a sample sentence.",
                "rating": 9.0,  # Changed from 7.5
                "batch_rating": False,  # Changed from True
                "tags": ["sample", "test", "updated"]  # Added "updated"
            }
        ]
        
        # Update ratings
        result = dm.update_ratings(updated_blocks)
        self.assertTrue(result)
        
        # Verify ratings were updated
        updated_sentence = dm.data["test-file.md"]["content"]["paragraphs"][0]["sentences"][0]
        self.assertEqual(updated_sentence["rating"], 9.0)
        self.assertEqual(updated_sentence["batch_rating"], False)
        self.assertIn("updated", updated_sentence["tags"])
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON files."""
        # Create an invalid JSON file
        with open(self.content_file, "w") as f:
            f.write("{invalid: json")
        
        # Create DataManager with the invalid file
        with patch("builtins.print") as mock_print:
            dm = DataManager(content_file=self.content_file)
            
            # Verify error handling
            self.assertIn("metadata", dm.data)
            self.assertIn("error", dm.data["metadata"])
            self.assertEqual(dm.data["metadata"]["error"], "invalid_json")
    
    def test_permission_denied_handling(self):
        """Test handling of permission denied errors."""
        # Create a DataManager
        dm = DataManager(content_file=self.content_file)
        
        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Permission denied")), \
             patch("builtins.print") as mock_print:
            # Try to save data
            result = dm.save_data()
            
            # Verify save failed
            self.assertFalse(result)
    
    def test_add_content_from_processed_file(self):
        """Test adding content from a processed file."""
        # Create a DataManager
        dm = DataManager(content_file=self.content_file)
        
        # Create a processed file with new content
        processed_file = os.path.join(self.json_dir, "processed_text_files.json")
        processed_content = {
            "metadata": {
                "version": "1.0"
            },
            "new-file.md": {
                "filename": "new-file.md",
                "last_modified": datetime.now().isoformat(),
                "processed_date": datetime.now().isoformat(),
                "content": {
                    "paragraphs": [
                        {
                            "text": "This is a new paragraph.",
                            "sentences": [
                                {
                                    "text": "This is a new sentence.",
                                    "rating": 0,
                                    "tags": ["new", "test"]
                                }
                            ]
                        }
                    ]
                }
            }
        }
        
        with open(processed_file, "w") as f:
            json.dump(processed_content, f)
        
        # Add content from the processed file
        result = dm.add_content_from_processed_file(processed_file)
        self.assertTrue(result)
        
        # Verify new content was added
        self.assertIn("new-file.md", dm.data)
        
        # Verify content blocks now include the new content
        blocks = dm.get_content_blocks()
        new_block_exists = False
        for block in blocks:
            if block["text"] == "This is a new sentence.":
                new_block_exists = True
                break
        
        self.assertTrue(new_block_exists)
    
    def test_get_canonical_file(self):
        """Test getting the canonical file path."""
        # Create a DataManager with a specific file
        dm = DataManager(content_file=self.content_file)
        
        # Get canonical file
        canonical_file = dm.get_canonical_file()
        
        # Verify it matches the file we provided
        self.assertEqual(canonical_file, self.content_file)

if __name__ == "__main__":
    unittest.main()
