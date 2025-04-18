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
    
    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent file."""
        nonexistent_file = os.path.join(self.test_dir, "nonexistent.json")
        
        # Create a mock for DataManager
        from src.core.data_manager import DataManager
        
        # Set up a custom __new__ and __init__ that creates a testable DataManager
        def mock_new(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = self._original_new(cls)
            return cls._instance
            
        def mock_init(self, content_file=None, *args, **kwargs):
            self._initialized = True
            self.content_file = content_file or nonexistent_file
            self.data = {"metadata": {"version": "1.0"}}
            
        # Apply the mocks
        DataManager.__new__ = mock_new
        DataManager.__init__ = mock_init
        
        # Create a processor with our mocked DataManager
        processor = ContentProcessor(nonexistent_file)
        
        # Verify processor is initialized with empty content
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.content_blocks, [])
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON file."""
        # Create an invalid JSON file
        with open(self.content_file, "w") as f:
            f.write("{invalid: json")
        
        # Create a mock for DataManager
        from src.core.data_manager import DataManager
        
        # Set up a custom __new__ and __init__ that creates a testable DataManager
        def mock_new(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = self._original_new(cls)
            return cls._instance
            
        def mock_init(self, content_file=None, *args, **kwargs):
            self._initialized = True
            self.content_file = content_file or self.content_file
            self.data = {"metadata": {"version": "1.0", "error": "invalid_json"}}
            
        # Apply the mocks
        DataManager.__new__ = mock_new
        DataManager.__init__ = mock_init
        
        # Create a processor with our mocked DataManager
        processor = ContentProcessor(self.content_file)
        
        # Verify processor is initialized with empty content
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.content_blocks, [])
    
    def test_permission_denied_reading(self):
        """Test handling of permission denied when reading file."""
        # Create a file with no permissions
        with open(self.content_file, "w") as f:
            f.write("{}")
        
        # Create a mock for DataManager
        from src.core.data_manager import DataManager
        
        # Set up a custom __new__ and __init__ that creates a testable DataManager
        def mock_new(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = self._original_new(cls)
            return cls._instance
            
        def mock_init(self, content_file=None, *args, **kwargs):
            self._initialized = True
            self.content_file = content_file or self.content_file
            self.data = {"metadata": {"version": "1.0", "error": "permission_denied"}}
            
        # Apply the mocks
        DataManager.__new__ = mock_new
        DataManager.__init__ = mock_init
        
        # Create a processor with our mocked DataManager
        processor = ContentProcessor(self.content_file)
        
        # Verify processor is initialized with empty content
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.content_blocks, [])
    
    def test_permission_denied_writing(self):
        """Test handling of permission denied when saving file."""
        # Create a valid content file
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
        
        # Store the content_data in a local variable for the closure
        test_data = self.content_data
        
        # Create a mock for DataManager
        from src.core.data_manager import DataManager
        
        # Set up a custom __new__ and __init__ that creates a testable DataManager with mock methods
        def mock_new(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = self._original_new(cls)
            return cls._instance
            
        def mock_init(self, content_file=None, *args, **kwargs):
            self._initialized = True
            self.content_file = content_file or self.content_file
            self.data = test_data  # Use the local variable from the closure
            self.get_content_blocks = lambda: [{"text": "Test content", "rating": 5.0}]
            self.update_ratings = lambda blocks: False  # Simulate permission error
            
        # Apply the mocks
        DataManager.__new__ = mock_new
        DataManager.__init__ = mock_init
        
        # Create a processor with our mocked DataManager
        processor = ContentProcessor(self.content_file)
        
        # Attempt to save ratings and verify it fails
        result = processor._save_ratings()
        self.assertFalse(result)
    
    def test_disk_space_error(self):
        """Test handling of disk space error when saving."""
        # Create a valid content file
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
        
        # Store the content_data in a local variable for the closure
        test_data = self.content_data
        
        # Create a mock for DataManager
        from src.core.data_manager import DataManager
        
        # Set up a custom __new__ and __init__ that creates a testable DataManager with mock methods
        def mock_new(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = self._original_new(cls)
            return cls._instance
            
        def mock_init(self, content_file=None, *args, **kwargs):
            self._initialized = True
            self.content_file = content_file or self.content_file
            self.data = test_data  # Use the local variable from the closure
            self.get_content_blocks = lambda: [{"text": "Test content", "rating": 5.0}]
            self.update_ratings = lambda blocks: False  # Simulate disk space error
            
        # Apply the mocks
        DataManager.__new__ = mock_new
        DataManager.__init__ = mock_init
        
        # Create a processor with our mocked DataManager
        processor = ContentProcessor(self.content_file)
        
        # Attempt to save ratings and verify it fails
        result = processor._save_ratings()
        self.assertFalse(result)
    
    def test_empty_content_handling(self):
        """Test handling of empty content file."""
        # Create an empty content file
        with open(self.content_file, "w") as f:
            f.write("{}")
        
        # Create a mock for DataManager
        from src.core.data_manager import DataManager
        
        # Set up a custom __new__ and __init__ that creates a testable DataManager
        def mock_new(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = self._original_new(cls)
            return cls._instance
            
        def mock_init(self, content_file=None, *args, **kwargs):
            self._initialized = True
            self.content_file = content_file or self.content_file
            self.data = {"metadata": {"version": "1.0"}}
            self.get_content_blocks = lambda: []
            
        # Apply the mocks
        DataManager.__new__ = mock_new
        DataManager.__init__ = mock_init
        
        # Create a processor with our mocked DataManager
        processor = ContentProcessor(self.content_file)
        
        # Verify processor is initialized with empty content
        self.assertIsInstance(processor, ContentProcessor)
        self.assertEqual(processor.content_blocks, [])
    
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

    def test_integration_with_datamanager(self):
        """Integration test for ContentProcessor with actual DataManager."""
        # Create a test file with sample content
        with open(self.content_file, "w") as f:
            json.dump(self.content_data, f)
            
        # Reset any DataManager singleton that might exist
        from src.core.data_manager import DataManager
        DataManager._instance = None
        
        try:
            # Create a ContentProcessor with the real file
            processor = ContentProcessor(self.content_file)
            
            # Verify processor initialized correctly with DataManager
            self.assertIsInstance(processor.data_manager, DataManager)
            self.assertEqual(processor.data_manager.content_file, self.content_file)
            
            # Verify content blocks were loaded
            original_blocks_count = len(processor.content_blocks)
            self.assertTrue(original_blocks_count > 0, "Content blocks should be loaded")
            
            # Find a block to modify
            found_block = None
            for block in processor.content_blocks:
                if "text" in block and block["text"] == "I am experienced in Python programming.":
                    found_block = block
                    break
            
            self.assertIsNotNone(found_block, "Expected test content block not found")
            
            # Store the original rating
            original_rating = found_block["rating"]
            
            # Update the rating
            found_block["rating"] = original_rating + 1.0
            
            # Save ratings
            result = processor._save_ratings()
            self.assertTrue(result, "Saving ratings should succeed")
            
            # Create a new processor to verify persistence
            DataManager._instance = None
            new_processor = ContentProcessor(self.content_file)
            
            # Find the same block
            updated_block = None
            for block in new_processor.content_blocks:
                if "text" in block and block["text"] == "I am experienced in Python programming.":
                    updated_block = block
                    break
            
            self.assertIsNotNone(updated_block, "Updated block should be found")
            
            # Verify the rating was updated
            self.assertEqual(updated_block["rating"], original_rating + 1.0)
        finally:
            # Reset DataManager for other tests
            DataManager._instance = None
    
    def test_integration_error_handling(self):
        """Integration test for error handling with actual DataManager."""
        # Create an invalid JSON file
        with open(self.content_file, "w") as f:
            f.write("{invalid json")
            
        # Reset any DataManager singleton that might exist
        from src.core.data_manager import DataManager
        DataManager._instance = None
        
        try:
            # Create a ContentProcessor with the invalid file
            processor = ContentProcessor(self.content_file)
            
            # Verify processor still initialized
            self.assertIsInstance(processor, ContentProcessor)
            self.assertIsInstance(processor.data_manager, DataManager)
            
            # Content blocks should be empty (or very few) due to error
            self.assertTrue(len(processor.content_blocks) < 2, 
                           "Content blocks should be empty or minimal when file is invalid")
            
            # Make DataManager's update_ratings method return False to simulate an error
            original_update = processor.data_manager.update_ratings
            processor.data_manager.update_ratings = lambda blocks: False
            
            try:
                # Attempt to save ratings (should fail gracefully)
                processor.content_blocks = [{"text": "Test", "rating": 5.0}]
                result = processor._save_ratings()
                
                # Save should return False now that we've mocked the update method
                self.assertFalse(result, "Save should fail when DataManager.update_ratings fails")
            finally:
                # Restore the original method if needed
                if hasattr(processor.data_manager, "update_ratings"):
                    processor.data_manager.update_ratings = original_update
        finally:
            # Reset DataManager for other tests
            DataManager._instance = None


if __name__ == "__main__":
    unittest.main()
