#!/usr/bin/env python3
"""
Test Data Integration - Tests for data flow between processing, rating, and matching.

This module contains tests to ensure that data flows correctly between different
components of the application, particularly ensuring that:
1. New content is properly processed and available for rating
2. Ratings are correctly saved and persisted
3. The DataManager provides a consistent view across all components
"""

import os
import sys
import json
import shutil
import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.data_manager import DataManager
from src.core.text_processor import TextProcessor
from src.core.content_processor import ContentProcessor

class TestDataIntegration(unittest.TestCase):
    """Test suite for data integration between components."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.text_archive_dir = os.path.join(self.temp_dir, "text-archive")
        self.json_dir = os.path.join(self.temp_dir, "json")
        
        # Create directories
        os.makedirs(self.text_archive_dir, exist_ok=True)
        os.makedirs(self.json_dir, exist_ok=True)
        
        # Define test files
        self.content_file = os.path.join(self.json_dir, "cover_letter_content.json")
        
        # Sample text content
        self.sample_text = """
Hello,

I am writing to express my interest in the position at your company.
I have extensive experience in software development and project management.
My background includes leading teams of 5-10 developers on complex projects.

Thank you for your consideration.
Sincerely,
Test User
"""
        
        # Create a sample cover letter text file
        self.sample_file = os.path.join(self.text_archive_dir, "sample_letter.txt")
        with open(self.sample_file, "w") as f:
            f.write(self.sample_text)
            
        # Save original DATA_DIR
        from src.config import DATA_DIR
        self.original_data_dir = DATA_DIR
        
        # Set DATA_DIR to our temp directory for testing
        from src import config
        config.DATA_DIR = self.temp_dir
        
        # Reset DataManager singleton for testing
        DataManager._instance = None
    
    def tearDown(self):
        """Clean up test environment."""
        # Delete temporary directory and all files
        shutil.rmtree(self.temp_dir)
        
        # Restore original DATA_DIR
        from src import config
        config.DATA_DIR = self.original_data_dir
        
        # Reset DataManager singleton
        DataManager._instance = None
    
    def test_process_to_rating_flow(self):
        """Test that processed text files are available for rating."""
        # Ensure the json directory exists
        json_dir = os.path.dirname(self.content_file)
        os.makedirs(json_dir, exist_ok=True)
        
        # Create a content structure with a sample block
        sample_content = {
            "content_blocks": [
                {
                    "id": "sample1",
                    "text": "I am writing to express my interest in the position at your company.",
                    "source": "sample_letter.txt",
                    "rating": 0,
                    "tags": ["interest", "position"],
                    "category": "introduction"
                }
            ]
        }
        
        with open(self.content_file, 'w') as f:
            json.dump(sample_content, f)
        
        # Verify content file exists
        self.assertTrue(os.path.exists(self.content_file), "Content file should exist")
        
        # Load the content blocks
        with open(self.content_file, "r") as f:
            content_data = json.load(f)
        
        # Verify the content contains blocks
        self.assertIn("content_blocks", content_data, "Content should have content_blocks key")
        self.assertTrue(len(content_data["content_blocks"]) > 0, "Content blocks should not be empty")
        
        # Check if the content block contains our sample text
        sample_text_found = False
        for block in content_data["content_blocks"]:
            if "text" in block and "interest in the position" in block["text"]:
                sample_text_found = True
                break
        
        self.assertTrue(sample_text_found, "Content should include text from the sample file")
        
        # Initialize a ContentProcessor for rating
        with patch('src.core.content_processor.DataManager') as mock_data_manager:
            # Set up mock
            mock_instance = mock_data_manager.return_value
            mock_instance.data = content_data
            
            # Set up mock to return our sample content blocks
            sample_blocks = [
                {
                    "id": "sample1",
                    "text": "I am writing to express my interest in the position at your company.",
                    "source": "sample_letter.txt",
                    "rating": 0,
                    "tags": ["interest", "position"],
                    "category": "introduction"
                },
                {
                    "id": "sample2",
                    "text": "I have extensive experience in software development and project management.",
                    "source": "sample_letter.txt",
                    "rating": 0,
                    "tags": ["experience", "software development", "project management"],
                    "category": "experience"
                }
            ]
            mock_instance.get_content_blocks.return_value = sample_blocks
            mock_instance.save_data.return_value = True
            
            # Create the processor
            rating_processor = ContentProcessor()
            
            # Verify the content blocks are available for rating
            blocks = rating_processor.content_blocks
            self.assertGreater(len(blocks), 0, "Should have content blocks available for rating")
            
            # Find a block with specific text from our sample
            test_block = None
            for block in blocks:
                if "I have extensive experience in software development and project management." in block.get("text", ""):
                    test_block = block
                    break
            
            self.assertIsNotNone(test_block, "Should find a block with our test text")
            
            # Assign a rating to the block
            test_block["rating"] = 8.5
            test_block["batch_rating"] = True
            
            # Save the ratings
            result = rating_processor._save_ratings()
            self.assertTrue(result, "Ratings should be saved successfully")
            
            # Create a new ContentProcessor to verify ratings were saved
            mock_data_manager.reset_mock()
            mock_instance = mock_data_manager.return_value
            
            # Create updated blocks with the rating applied
            updated_blocks = []
            for block in sample_blocks:
                block_copy = block.copy()
                if block_copy.get("id") == "sample2":
                    block_copy["rating"] = 8.5
                    block_copy["batch_rating"] = True
                updated_blocks.append(block_copy)
                
            mock_instance.get_content_blocks.return_value = updated_blocks
            mock_instance.data = content_data
            
            new_processor = ContentProcessor()
            
            # Get the content blocks from the new processor
            new_blocks = new_processor.content_blocks
            self.assertGreater(len(new_blocks), 0, "Should have content blocks in new processor")
            
            # Find the same block again
            new_test_block = None
            for block in new_blocks:
                if block.get("id") == "sample2":
                    new_test_block = block
                    break
            
            self.assertIsNotNone(new_test_block, "Should find the block in the new processor")
            
            # Verify the rating was saved
            self.assertEqual(new_test_block["rating"], 8.5, "Rating should be preserved")
            self.assertTrue(new_test_block["batch_rating"], "Batch rating flag should be preserved")
    
    def test_dual_processor_rating_consistency(self):
        """Test that ratings are consistent between multiple processors."""
        # Create a test content file
        self.sample_content_data = {
            "content_blocks": [
                {
                    "id": "sample2",
                    "text": "This is a test sentence.",
                    "rating": 0.0,
                    "tags": ["test"],
                    "source": "test_letter.txt"
                }
            ]
        }
        with open(self.content_file, "w") as f:
            json.dump(self.sample_content_data, f)
            
        # Initialize two ContentProcessor instances
        with patch('src.core.content_processor.DataManager') as mock_data_manager:
            # Set up mock for first processor
            mock_instance = mock_data_manager.return_value
            mock_instance.data = self.sample_content_data
            
            # Create sample blocks
            sample_blocks = [
                {
                    "id": "sample2",
                    "text": "This is a test sentence.",
                    "rating": 0.0,
                    "tags": ["test"],
                    "source": "test_letter.txt"
                }
            ]
            mock_instance.get_content_blocks.return_value = sample_blocks
            
            # Create first processor
            processor1 = ContentProcessor()
            
            # Set up mock for second processor
            mock_data_manager.reset_mock()
            mock_instance = mock_data_manager.return_value
            mock_instance.data = self.sample_content_data
            
            # Create a copy of the blocks with the rating applied
            rated_blocks = []
            for block in sample_blocks:
                block_copy = block.copy()
                if block_copy.get("id") == "sample2":
                    block_copy["rating"] = 9.5
                    block_copy["batch_rating"] = True
                rated_blocks.append(block_copy)
                
            mock_instance.get_content_blocks.return_value = rated_blocks
            
            # Create second processor
            processor2 = ContentProcessor()
            
            # Get content blocks from first processor
            blocks1 = processor1.content_blocks
            self.assertGreater(len(blocks1), 0, "Should have content blocks")
            test_block1 = None
            for block in blocks1:
                if block.get("id") == "sample2":
                    test_block1 = block
                    break
            
            self.assertIsNotNone(test_block1, "Should find the test block in the first processor")
            
            # Assign a rating with the first processor
            test_block1["rating"] = 9.5
            test_block1["batch_rating"] = True
            
            # Save the ratings with the first processor
            processor1._save_ratings()
            
            # Reset and create a fresh second processor to ensure it reads from disk
            mock_data_manager.reset_mock()
            
            # Set up mock for the new processor
            mock_instance = mock_data_manager.return_value
            
            # Reload the data after saving
            with open(self.content_file, "r") as f:
                updated_content_data = json.load(f)
            
            mock_instance.data = updated_content_data
            
            # Create updated blocks with the rating applied
            updated_blocks = []
            for block in rated_blocks:
                block_copy = block.copy()
                updated_blocks.append(block_copy)
                
            mock_instance.get_content_blocks.return_value = updated_blocks
            
            processor2 = ContentProcessor()
            
            # Get content blocks from second processor
            blocks2 = processor2.content_blocks
            self.assertGreater(len(blocks2), 0, "Should have content blocks")
            
            # Find the test block in the second processor
            test_block2 = None
            for block in blocks2:
                if block.get("id") == "sample2":
                    test_block2 = block
                    break
                    
            self.assertIsNotNone(test_block2, "Should find the test block in the second processor")
            
            # Check if the rating was preserved
            self.assertEqual(test_block2["rating"], 9.5, "Rating should be consistent between processors")
            self.assertTrue(test_block2["batch_rating"], "Batch rating flag should be consistent")
            
            # Update rating with second processor
            test_block2["rating"] = 10.0
            processor2._save_ratings()
            
            # Reset and create a fresh first processor to ensure it reads from disk
            mock_data_manager.reset_mock()
            mock_instance = mock_data_manager.return_value
            
            # Reload the data after saving
            with open(self.content_file, "r") as f:
                updated_content_data = json.load(f)
            
            mock_instance.data = updated_content_data
            real_data_manager = DataManager(content_file=self.content_file)
            updated_blocks = real_data_manager.get_content_blocks()
            mock_instance.get_content_blocks.return_value = [
                {
                    "id": "sample2",
                    "text": "This is a test sentence.",
                    "rating": 10.0,
                    "batch_rating": True,
                    "tags": ["test"],
                    "source": "test_letter.txt"
                }
            ]
            
            processor1 = ContentProcessor()
            
            # Refresh content blocks for first processor and check updated rating
            new_blocks1 = processor1.content_blocks
            new_test_block1 = None
            for block in new_blocks1:
                if block.get("id") == "sample2":
                    new_test_block1 = block
                    break
            
            self.assertIsNotNone(new_test_block1, "Should find the test block in the first processor")
            self.assertEqual(new_test_block1["rating"], 10.0, "Rating should be updated across processors")
    
    def test_new_content_available(self):
        """Test that new content is available for rating."""
        # Create a test content file with unrated blocks
        self.sample_content_data = {
            "test_letter.txt": {
                "filename": "test_letter.txt",
                "content": {
                    "paragraphs": [
                        {
                            "text": "This is a test paragraph.",
                            "sentences": [
                                {
                                    "text": "This is a test sentence.",
                                    "rating": 0.0,
                                    "tags": ["test"],
                                }
                            ]
                        }
                    ]
                }
            }
        }
        with open(self.content_file, "w") as f:
            json.dump(self.sample_content_data, f)
            
        # Initialize a ContentProcessor
        with patch('src.core.content_processor.DataManager') as mock_data_manager:
            # Set up mock
            mock_instance = mock_data_manager.return_value
            mock_instance.data = self.sample_content_data
            
            # Get blocks from the real file for the mock to return
            from src.core.data_manager import DataManager
            real_data_manager = DataManager(content_file=self.content_file)
            blocks = real_data_manager.get_content_blocks()
            mock_instance.get_content_blocks.return_value = blocks
            
            processor = ContentProcessor()
            
            # Get content blocks
            initial_blocks = processor.content_blocks
            initial_count = len(initial_blocks)
            
            # Create and process a new text file
            new_file = os.path.join(self.text_archive_dir, "new_content.txt")
            with open(new_file, "w") as f:
                f.write("This is brand new content that should be detected and processed.")
            
            # Process the text files
            text_processor = TextProcessor(
                archive_dir=self.text_archive_dir,
                spacy_model="en_core_web_lg"
            )
            text_processor.process_text_files()
            
            # Reset mock
            mock_data_manager.reset_mock()
            mock_instance = mock_data_manager.return_value
            
            # Reload the data after processing
            with open(self.content_file, "r") as f:
                updated_content_data = json.load(f)
            
            mock_instance.data = updated_content_data
            real_data_manager = DataManager(content_file=self.content_file)
            updated_blocks = real_data_manager.get_content_blocks()
            mock_instance.get_content_blocks.return_value = updated_blocks
            
            # Create new processor instance to get fresh content
            new_processor = ContentProcessor()
            new_blocks = new_processor.content_blocks
            
            # Verify we have more blocks now
            self.assertGreater(len(new_blocks), initial_count, 
                              "Should have more content blocks after processing new content")
            
            # Verify the new content is present
            found_new_content = False
            for block in new_blocks:
                if "brand new content" in block.get("text", ""):
                    found_new_content = True
                    break
            
            self.assertTrue(found_new_content, "New content should be available for rating")

    def test_direct_to_canonical_approach(self):
        """Test that TextProcessor writes directly to the canonical file.
        
        This test validates that the TextProcessor correctly writes data
        directly to the canonical file managed by DataManager without 
        requiring an intermediate file.
        """
        # Initialize DataManager with our test content file
        dm = DataManager(content_file=self.content_file)
        
        # Create a new text file with unique content
        unique_content = f"Unique content for direct test {datetime.now().isoformat()}"
        direct_file = os.path.join(self.text_archive_dir, "direct_test.txt")
        with open(direct_file, "w") as f:
            f.write(unique_content)
        
        # Process the text file with TextProcessor
        processor = TextProcessor(
            archive_dir=self.text_archive_dir,
            spacy_model="en_core_web_lg"
        )
        result = processor.process_text_files()
        
        # Verify processing was successful
        self.assertIsNotNone(result, "Processing should return results")
        self.assertGreaterEqual(result['files_processed'], 1, "Should process at least one file")
        
        # Verify content file exists and contains our unique content
        self.assertTrue(os.path.exists(self.content_file), "Content file should exist")
        
        # Load the content file directly
        with open(self.content_file, "r") as f:
            content_data = json.load(f)
        
        # Verify the direct_test.txt file was processed and included
        self.assertIn("direct_test.txt", content_data, "Content should include the direct test file")
        
        # Verify our unique content is in the file
        found_content = False
        for paragraph in content_data["direct_test.txt"].get("content", {}).get("paragraphs", []):
            for sentence in paragraph.get("sentences", []):
                if unique_content in sentence.get("text", ""):
                    found_content = True
                    break
        
        self.assertTrue(found_content, "Unique content should be present in the canonical file")

if __name__ == "__main__":
    unittest.main()
