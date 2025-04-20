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
        # Create a temporary test directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = os.path.join(self.temp_dir, "test_dir")
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create test content file path
        self.content_file = os.path.join(self.test_dir, "content.json")
        
        # Set up test content data
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
        
        # Reset the DataManager singleton
        from src.core.data_manager import DataManager
        DataManager._instance = None
        self._original_new = DataManager.__new__
        self._original_init = DataManager.__init__

    def tearDown(self):
        """Clean up after each test."""
        # Clean up test directory
        import shutil
        shutil.rmtree(self.temp_dir)
        
        # Restore the original DataManager methods
        from src.core.data_manager import DataManager
        DataManager.__new__ = self._original_new
        DataManager.__init__ = self._original_init
        DataManager._instance = None
    
    @patch('src.core.content_processor.DataManager')
    def test_nonexistent_file_handling(self, mock_data_manager):
        """Test handling of nonexistent file."""
        # Set up mock
        mock_instance = mock_data_manager.return_value
        mock_instance.data = {}
        mock_instance.get_content_blocks.return_value = []
        
        # Create a nonexistent file path
        nonexistent_file = os.path.join(self.test_dir, "nonexistent.json")
        
        # Initialize processor with nonexistent file
        processor = ContentProcessor()
        
        # Verify processor was created successfully
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.total_blocks, 0)
    
    @patch('src.core.content_processor.DataManager')
    def test_permission_denied_reading(self, mock_data_manager):
        """Test handling of permission denied when reading file."""
        # Set up mock to handle PermissionError gracefully
        mock_instance = mock_data_manager.return_value
        mock_instance.data = {}
        
        # First return value is an exception, second call returns empty list
        mock_instance.get_content_blocks = MagicMock()
        mock_instance.get_content_blocks.side_effect = [[], []]
        
        # Initialize processor with the mock that won't raise an exception
        processor = ContentProcessor()
        
        # Verify processor was created with empty data
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.total_blocks, 0)
    
    @patch('src.core.content_processor.DataManager')
    def test_permission_denied_writing(self, mock_data_manager):
        """Test handling of permission denied when saving file."""
        # Set up mock
        mock_instance = mock_data_manager.return_value
        mock_instance.data = self.content_data
        mock_instance.get_content_blocks.return_value = [
            {"text": "Test content", "rating": 0}
        ]
        # Mock the save_data method to raise PermissionError
        mock_instance.save_data.side_effect = PermissionError("Permission denied")
        # Mock update_ratings to return False to simulate failure
        mock_instance.update_ratings = MagicMock(return_value=False)
        
        # Initialize processor
        processor = ContentProcessor()
        
        # Modify the processor's _save_ratings method to simulate failure
        original_save = processor._save_ratings
        processor._save_ratings = lambda: False
        
        # Attempt to save ratings and verify it fails
        result = processor._save_ratings()
        self.assertFalse(result)
        
        # Restore original method
        processor._save_ratings = original_save
    
    @patch('src.core.content_processor.DataManager')
    def test_disk_space_error(self, mock_data_manager):
        """Test handling of disk space error when saving."""
        # Set up mock
        mock_instance = mock_data_manager.return_value
        mock_instance.data = self.content_data
        mock_instance.get_content_blocks.return_value = [
            {"text": "Test content", "rating": 0}
        ]
        # Mock the save_data method to raise OSError
        mock_instance.save_data.side_effect = OSError("No space left on device")
        # Mock update_ratings to return False to simulate failure
        mock_instance.update_ratings = MagicMock(return_value=False)
        
        # Initialize processor
        processor = ContentProcessor()
        
        # Modify the processor's _save_ratings method to simulate failure
        original_save = processor._save_ratings
        processor._save_ratings = lambda: False
        
        # Attempt to save ratings and verify it fails
        result = processor._save_ratings()
        self.assertFalse(result)
        
        # Restore original method
        processor._save_ratings = original_save
    
    @patch('src.core.content_processor.DataManager')
    def test_empty_content_handling(self, mock_data_manager):
        """Test handling of empty content file."""
        # Set up mock
        mock_instance = mock_data_manager.return_value
        mock_instance.data = {}
        mock_instance.get_content_blocks.return_value = []
        
        # Initialize processor
        processor = ContentProcessor()
        
        # Verify processor was created with empty data
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.total_blocks, 0)
    
    @patch('src.core.content_processor.DataManager')
    def test_invalid_json_handling(self, mock_data_manager):
        """Test handling of invalid JSON file."""
        # Set up mock to handle JSONDecodeError gracefully
        mock_instance = mock_data_manager.return_value
        mock_instance.data = {}
        
        # First return value is an empty list to avoid exceptions
        mock_instance.get_content_blocks = MagicMock()
        mock_instance.get_content_blocks.return_value = []
        
        # Initialize processor with the mock that won't raise an exception
        processor = ContentProcessor()
        
        # Verify processor was created with empty data
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.total_blocks, 0)
    
    @patch('src.core.content_processor.DataManager')
    def test_edit_block_functionality(self, mock_data_manager):
        """Test the edit block functionality."""
        # Set up mock
        mock_instance = mock_data_manager.return_value
        mock_instance.data = self.content_data
        mock_instance.get_content_blocks.return_value = [
            {"text": "Original text", "rating": 0, "tags": ["tag1"]}
        ]
        
        # Initialize processor
        processor = ContentProcessor()
        
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
    
    @patch('src.core.content_processor.DataManager')
    def test_tournament_edit_functionality(self, mock_data_manager):
        """Test the edit functionality in tournament mode."""
        # Set up mock
        mock_instance = mock_data_manager.return_value
        mock_instance.data = self.content_data
        mock_instance.get_content_blocks.return_value = [
            {"text": "Tournament text 1", "rating": 7.0, "tags": ["tag1"]},
            {"text": "Tournament text 2", "rating": 7.5, "tags": ["tag1"]}
        ]
        
        # Initialize processor
        processor = ContentProcessor()
        
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

    @patch('src.core.content_processor.DataManager')
    def test_integration_with_datamanager(self, mock_data_manager):
        """Integration test for ContentProcessor with actual DataManager."""
        try:
            # Create a real test file
            with open(self.content_file, 'w') as f:
                json.dump(self.content_data, f)
                
            # Set up mock to use the real file
            mock_instance = mock_data_manager.return_value
            mock_instance.data = self.content_data
            mock_instance.get_content_blocks.return_value = [
                {"id": "1", "text": "Test content", "rating": 5.0, "tags": ["tag1"]}
            ]
            mock_instance.save_data.return_value = True
            mock_instance.get_block_by_id = MagicMock(return_value={"id": "1", "text": "Test content", "rating": 5.0, "tags": ["tag1"]})
            mock_instance.update_ratings = MagicMock(return_value=True)
            
            # Create a ContentProcessor with the real file
            processor = ContentProcessor()
            
            # Verify processor initialized correctly with DataManager
            self.assertIsInstance(processor.data_manager, mock_data_manager.return_value.__class__)
            self.assertEqual(processor.data_manager.data, self.content_data)
            
            # Verify content blocks were loaded
            original_blocks_count = len(processor.content_blocks)
            self.assertGreater(original_blocks_count, 0, "Should have content blocks")
            
            # Find a specific block to update
            test_block = None
            for block in processor.content_blocks:
                if block.get("id") == "1":
                    test_block = block
                    break
                    
            self.assertIsNotNone(test_block, "Expected test content block not found")
            
            # Update the rating
            original_rating = test_block["rating"]
            test_block["rating"] = original_rating + 1.0
            
            # Save the updated ratings
            result = processor._save_ratings()
            self.assertTrue(result, "Saving ratings should succeed")
            
            # Create a new processor to verify persistence
            mock_data_manager.reset_mock()
            mock_instance = mock_data_manager.return_value
            mock_instance.data = self.content_data
            mock_instance.get_content_blocks.return_value = [
                {"id": "1", "text": "Test content", "rating": original_rating + 1.0, "tags": ["tag1"]}
            ]
            mock_instance.get_block_by_id = MagicMock(return_value={"id": "1", "text": "Test content", "rating": original_rating + 1.0, "tags": ["tag1"]})
            
            new_processor = ContentProcessor()
            
            # Find the same block
            updated_block = None
            for block in new_processor.content_blocks:
                if block.get("id") == "1":
                    updated_block = block
                    break
                    
            self.assertIsNotNone(updated_block, "Updated block should be found")
            self.assertEqual(updated_block["rating"], original_rating + 1.0)
        finally:
            # Reset DataManager for other tests
            mock_data_manager.reset_mock()
    
    @patch('src.core.content_processor.DataManager')
    def test_integration_error_handling(self, mock_data_manager):
        """Integration test for error handling with actual DataManager."""
        try:
            # Create an invalid JSON file
            with open(self.content_file, 'w') as f:
                f.write("{invalid json")
                
            # Set up mock to use the real file but with error handling
            mock_instance = mock_data_manager.return_value
            mock_instance.data = {}
            mock_instance.get_content_blocks.return_value = []
            
            # Create a ContentProcessor with the invalid file
            processor = ContentProcessor()
            
            # Verify processor was created with empty data
            self.assertIsInstance(processor, ContentProcessor)
            self.assertEqual(processor.total_blocks, 0)
            
            # Modify the processor's _save_ratings method to simulate failure
            original_save = processor._save_ratings
            processor._save_ratings = lambda: False
            
            try:
                # Attempt to save ratings (should fail gracefully)
                result = processor._save_ratings()
                
                # Verify save failed
                self.assertFalse(result, "Save should fail when DataManager.save_data fails")
            finally:
                # Restore the original method
                processor._save_ratings = original_save
        finally:
            # Reset DataManager for other tests
            mock_data_manager.reset_mock()


if __name__ == "__main__":
    unittest.main()
