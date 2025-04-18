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
        self.processed_file = os.path.join(self.json_dir, "processed_text_files.json")
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
        # Initialize DataManager with our test content file
        dm = DataManager(content_file=self.content_file)
        
        # Process the text files
        processor = TextProcessor(
            archive_dir=self.text_archive_dir,
            output_file=self.processed_file,
            spacy_model="en_core_web_lg"
        )
        result = processor.process_text_files(force_reprocess=True)
        
        # Verify processing results
        self.assertIsNotNone(result, "Processing should return results")
        self.assertEqual(result['files_processed'], 1, "Should process 1 file")
        
        # Verify processed file exists
        self.assertTrue(os.path.exists(self.processed_file), "Processed file should exist")
        
        # Verify content file exists (created by DataManager)
        self.assertTrue(os.path.exists(self.content_file), "Content file should exist")
        
        # Load the processed content blocks
        with open(self.content_file, "r") as f:
            content_data = json.load(f)
        
        # Verify the content contains our sample file
        self.assertIn("sample_letter.txt", content_data, "Content should include the sample file")
        
        # Initialize a ContentProcessor for rating
        rating_processor = ContentProcessor(json_file=self.content_file)
        
        # Verify the content blocks are available for rating
        blocks = rating_processor.content_blocks
        self.assertGreater(len(blocks), 0, "Should have content blocks available for rating")
        
        # Find a block with specific text from our sample
        test_block = None
        for block in blocks:
            if "software development" in block.get("text", ""):
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
        new_processor = ContentProcessor(json_file=self.content_file)
        new_blocks = new_processor.content_blocks
        
        # Find the same block again
        new_test_block = None
        for block in new_blocks:
            if "software development" in block.get("text", ""):
                new_test_block = block
                break
        
        self.assertIsNotNone(new_test_block, "Should find the block in the new processor")
        self.assertEqual(new_test_block["rating"], 8.5, "Rating should be preserved")
        self.assertTrue(new_test_block["batch_rating"], "Batch rating flag should be preserved")
    
    def test_dual_processor_rating_consistency(self):
        """Test that ratings are consistent between multiple processors."""
        # Create a test content file with initial content
        initial_content = {
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
        
        # Write the initial content to our test content file
        os.makedirs(os.path.dirname(self.content_file), exist_ok=True)
        with open(self.content_file, "w") as f:
            json.dump(initial_content, f)
        
        # Reset DataManager singleton to ensure clean state
        DataManager._instance = None
        
        # Create two separate processor instances both pointing to our test content file
        processor1 = ContentProcessor(json_file=self.content_file)
        processor2 = ContentProcessor(json_file=self.content_file)
        
        # Get content blocks from first processor
        blocks1 = processor1.content_blocks
        self.assertGreater(len(blocks1), 0, "Should have content blocks")
        test_block1 = blocks1[0]
        
        # Assign a rating with the first processor
        test_block1["rating"] = 9.5
        test_block1["batch_rating"] = True
        
        # Save the ratings with the first processor
        processor1._save_ratings()
        
        # Reset and create a fresh second processor to ensure it reads from disk
        DataManager._instance = None
        processor2 = ContentProcessor(json_file=self.content_file)
        
        # Get content blocks from second processor
        blocks2 = processor2.content_blocks
        self.assertGreater(len(blocks2), 0, "Should have content blocks")
        test_block2 = blocks2[0]
        
        # Check if the rating was preserved
        self.assertEqual(test_block2["rating"], 9.5, "Rating should be consistent between processors")
        self.assertTrue(test_block2["batch_rating"], "Batch rating flag should be consistent")
        
        # Update rating with second processor
        test_block2["rating"] = 10.0
        processor2._save_ratings()
        
        # Reset and create a fresh first processor to ensure it reads from disk
        DataManager._instance = None
        processor1 = ContentProcessor(json_file=self.content_file)
        
        # Refresh content blocks for first processor and check updated rating
        new_blocks1 = processor1.content_blocks
        new_test_block1 = new_blocks1[0]
        self.assertEqual(new_test_block1["rating"], 10.0, "Rating should be updated across processors")
    
    def test_new_content_available(self):
        """Test that new content is available for rating."""
        # Start with an empty content file
        with open(self.content_file, "w") as f:
            json.dump({"metadata": {"version": "1.0"}}, f)
        
        # Reset DataManager singleton
        DataManager._instance = None
        
        # Create initial processor
        processor = ContentProcessor(json_file=self.content_file)
        initial_blocks = processor.content_blocks
        initial_count = len(initial_blocks)
        
        # Create and process a new text file
        new_file = os.path.join(self.text_archive_dir, "new_content.txt")
        with open(new_file, "w") as f:
            f.write("This is brand new content that should be detected and processed.")
        
        # Process the text files
        text_processor = TextProcessor(
            archive_dir=self.text_archive_dir,
            output_file=self.processed_file,
            spacy_model="en_core_web_lg"
        )
        text_processor.process_text_files(force_reprocess=True)
        
        # Reset DataManager singleton
        DataManager._instance = None
        
        # Create new processor instance to get fresh content
        new_processor = ContentProcessor(json_file=self.content_file)
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

if __name__ == "__main__":
    unittest.main()
