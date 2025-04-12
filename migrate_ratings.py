#!/usr/bin/env python3
"""
Migration script for transferring ratings from the old processed_cover_letters.json
to the new file in the refactored system.

This is a one-time migration script to preserve ratings and edited text
during the system refactoring.
"""

import os
import json
from datetime import datetime
import sys
from typing import Dict, List, Any

def migrate_ratings():
    """Migrate ratings from old file to new file."""
    # File paths
    old_file = "processed_cover_letters.json"
    new_file = "coverletter_wiz/data/processed_cover_letters.json"
    
    # Check if files exist
    if not os.path.exists(old_file):
        print(f"Error: Old file {old_file} not found.")
        return False
        
    if not os.path.exists(new_file):
        print(f"Error: New file {new_file} not found.")
        print("Please run the text processor first to generate the new file.")
        return False
    
    # Load old file
    try:
        with open(old_file, "r") as f:
            old_data = json.load(f)
        print(f"Loaded old data from {old_file}")
    except Exception as e:
        print(f"Error loading old file: {e}")
        return False
    
    # Load new file
    try:
        with open(new_file, "r") as f:
            new_data = json.load(f)
        print(f"Loaded new data from {new_file}")
    except Exception as e:
        print(f"Error loading new file: {e}")
        return False
    
    # Create backup of new file
    backup_file = f"{new_file}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
    try:
        with open(backup_file, "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"Created backup of new file: {backup_file}")
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False
    
    # Track migration statistics
    stats = {
        "files_processed": 0,
        "blocks_with_ratings": 0,
        "blocks_with_text_changes": 0,
        "total_blocks": 0
    }
    
    # Build a comprehensive map of all sentences in the old file
    old_sentences_map = {}
    
    # First, extract all sentences from the old file
    for filename, file_data in old_data.items():
        # Skip metadata
        if filename == "metadata":
            continue
            
        # Skip non-dictionary items or items without content
        if not isinstance(file_data, dict) or "content" not in file_data:
            continue
            
        # Process paragraphs
        if isinstance(file_data["content"], dict) and "paragraphs" in file_data["content"]:
            paragraphs = file_data["content"]["paragraphs"]
            
            for paragraph in paragraphs:
                # Store the full paragraph text
                paragraph_text = paragraph.get("text", "").strip()
                if paragraph_text:
                    old_sentences_map[paragraph_text] = {
                        "type": "paragraph",
                        "data": paragraph
                    }
                
                # Store individual sentences
                for sentence in paragraph.get("sentences", []):
                    text = sentence.get("text", "").strip()
                    if text:
                        old_sentences_map[text] = {
                            "type": "sentence",
                            "data": sentence
                        }
    
    # Process each file in the new data
    for filename, new_file_data in new_data.items():
        # Skip metadata
        if filename == "metadata":
            continue
            
        # Check if file exists in old data
        if filename not in old_data:
            print(f"File {filename} not found in old data, skipping.")
            continue
        
        old_file_data = old_data[filename]
        
        # Skip non-dictionary items or items without content
        if not isinstance(new_file_data, dict) or "content" not in new_file_data:
            continue
            
        # Process paragraphs
        if isinstance(new_file_data["content"], dict) and "paragraphs" in new_file_data["content"]:
            new_paragraphs = new_file_data["content"]["paragraphs"]
            
            # Update each paragraph
            for paragraph_index, paragraph in enumerate(new_paragraphs):
                paragraph_text = paragraph.get("text", "").strip()
                stats["total_blocks"] += 1
                
                # Check if this paragraph exists in old data
                if paragraph_text in old_sentences_map and old_sentences_map[paragraph_text]["type"] == "paragraph":
                    # Use the old paragraph text (in case there are subtle differences)
                    old_paragraph = old_sentences_map[paragraph_text]["data"]
                    paragraph["text"] = old_paragraph["text"]
                    
                    # Copy tags if available
                    if "tags" in old_paragraph and old_paragraph["tags"]:
                        paragraph["tags"] = old_paragraph["tags"]
                        
                    stats["blocks_with_text_changes"] += 1
                
                # Process sentences in the paragraph
                for sentence_index, sentence in enumerate(paragraph.get("sentences", [])):
                    text = sentence.get("text", "").strip()
                    if not text:
                        continue
                        
                    # Check if this sentence exists in old data
                    if text in old_sentences_map and old_sentences_map[text]["type"] == "sentence":
                        old_sentence = old_sentences_map[text]["data"]
                        
                        # Use the old sentence text (in case there are subtle differences)
                        sentence["text"] = old_sentence["text"]
                        
                        # Transfer rating
                        if "rating" in old_sentence and old_sentence["rating"] > 0:
                            sentence["rating"] = old_sentence["rating"]
                            stats["blocks_with_ratings"] += 1
                            
                            # Transfer other rating metadata
                            if "last_rated" in old_sentence:
                                sentence["last_rated"] = old_sentence["last_rated"]
                            if "batch_rating" in old_sentence:
                                sentence["batch_rating"] = old_sentence["batch_rating"]
                            
                            print(f"Transferred rating {sentence['rating']} for: {text[:50]}...")
                        
                        # Copy tags if available
                        if "tags" in old_sentence and old_sentence["tags"]:
                            sentence["tags"] = old_sentence["tags"]
                            
                        # Copy primary_tags if available
                        if "primary_tags" in old_sentence and old_sentence["primary_tags"]:
                            sentence["primary_tags"] = old_sentence["primary_tags"]
                            
                        # Copy tag_scores if available
                        if "tag_scores" in old_sentence and old_sentence["tag_scores"]:
                            sentence["tag_scores"] = old_sentence["tag_scores"]
                            
                        stats["blocks_with_text_changes"] += 1
        
        stats["files_processed"] += 1
    
    # Update metadata
    new_data["metadata"]["migrated_from"] = old_file
    new_data["metadata"]["migration_date"] = datetime.now().isoformat()
    new_data["metadata"]["migration_stats"] = stats
    
    # Save updated data
    try:
        with open(new_file, "w") as f:
            json.dump(new_data, f, indent=2)
        print(f"Saved updated data to {new_file}")
    except Exception as e:
        print(f"Error saving updated data: {e}")
        return False
    
    # Print migration summary
    print("\nMigration Summary:")
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total content blocks: {stats['total_blocks']}")
    print(f"Blocks with ratings transferred: {stats['blocks_with_ratings']}")
    print(f"Blocks with text changes: {stats['blocks_with_text_changes']}")
    
    return True

if __name__ == "__main__":
    success = migrate_ratings()
    sys.exit(0 if success else 1)
