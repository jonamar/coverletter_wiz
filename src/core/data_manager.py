#!/usr/bin/env python3
"""
Data Manager - Centralized data access layer for coverletter_wiz.

This module provides a unified interface for accessing and modifying cover letter content data,
ensuring consistent data handling across all components of the application.
"""

from __future__ import annotations

import os
import json
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path

# Import config for data directory paths
from src.config import DATA_DIR

# Constants
CONTENT_FILE = os.path.join(DATA_DIR, "json/cover_letter_content.json")

class DataManager:
    """
    Centralized data access manager for coverletter_wiz.
    
    This class provides a unified interface for accessing and modifying
    cover letter content data, ensuring all components use a consistent
    data model and preventing synchronization issues between multiple files.
    
    Features:
    - Single interface for all data operations
    - Consistent error handling and logging
    - Centralized file access and management
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, content_file: str = CONTENT_FILE):
        """Create a singleton instance of DataManager.
        
        Args:
            content_file: Path to the content JSON file.
        
        Returns:
            DataManager: Singleton instance of DataManager.
        """
        if cls._instance is None:
            cls._instance = super(DataManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, content_file: str = CONTENT_FILE):
        """Initialize the DataManager with the given content file.
        
        Args:
            content_file: Path to the content JSON file.
        """
        # Skip initialization if already initialized (singleton pattern)
        if self._initialized:
            return
            
        self.content_file = content_file
        self.data = self._load_content_data()
        self._initialized = True
    
    def _load_content_data(self) -> Dict[str, Any]:
        """Load content data from the JSON file.
        
        The function includes robust error handling for common file issues:
        - File not found
        - Malformed JSON
        - Empty content
        - Permission issues
        
        Returns:
            Content data dictionary with file metadata and content blocks.
        """
        try:
            # Check if file exists
            if not os.path.exists(self.content_file):
                print(f"Notice: Content file {self.content_file} does not exist.")
                print("Creating empty content structure.")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat()}}
                
            # Check if file is readable
            if not os.access(self.content_file, os.R_OK):
                print(f"Error: No read permission for {self.content_file}.")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "permission_denied"}}
                
            # Try to read the file
            with open(self.content_file, "r") as f:
                content_data = json.load(f)
                
            # Validate content structure
            if not isinstance(content_data, dict):
                print(f"Error: Content file {self.content_file} has invalid format (root not a dictionary).")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "invalid_format"}}
                
            # Check for empty content
            if not content_data:
                print(f"Warning: Content file {self.content_file} is empty.")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat()}}
                
            return content_data
            
        except json.JSONDecodeError as e:
            print(f"Error: Content file {self.content_file} contains invalid JSON: {e}")
            print(f"Line {e.lineno}, Column {e.colno}: {e.msg}")
            return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "invalid_json"}}
        except PermissionError:
            print(f"Error: Permission denied when trying to read {self.content_file}")
            return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "permission_denied"}}
        except Exception as e:
            print(f"Unexpected error loading content file {self.content_file}: {e}")
            traceback.print_exc()
            return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": str(e)}}
    
    def save_data(self) -> bool:
        """Save data to the content file.
        
        Returns:
            bool: True if the save was successful, False otherwise.
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.content_file), exist_ok=True)
            
            # Write to file
            with open(self.content_file, "w") as f:
                json.dump(self.data, f, indent=2)
            
            print(f"Data saved to {self.content_file}")
            return True
            
        except Exception as e:
            print(f"Error saving data: {e}")
            traceback.print_exc()
            return False
    
    def get_content_blocks(self) -> List[Dict[str, Any]]:
        """Extract and return all content blocks from the data.
        
        Returns:
            List[Dict[str, Any]]: List of content blocks.
        """
        unique_blocks: Dict[str, Dict[str, Any]] = {}
        
        for filename, file_data in self.data.items():
            # Skip metadata keys
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
                
            paragraphs = file_data.get("content", {}).get("paragraphs", [])
            
            for paragraph in paragraphs:
                paragraph_text = paragraph.get("text", "")
                blocks = paragraph.get("sentences", [])
                
                for block in blocks:
                    text = block.get("text", "").strip()
                    if not text:
                        continue
                        
                    # Check if this is a new block or if we should update an existing one
                    if text not in unique_blocks:
                        # Create a new entry
                        is_group = block.get("is_sentence_group", False)
                        component_sentences = block.get("component_sentences", [])
                        
                        unique_blocks[text] = {
                            "text": text,
                            "sources": [filename],
                            "rating": block.get("rating", 0),
                            "batch_rating": block.get("batch_rating", False),
                            "tags": block.get("tags", []),
                            "is_content_group": is_group,
                            "component_content": component_sentences,
                            "context": paragraph_text
                        }
                    else:
                        # Update existing entry
                        if filename not in unique_blocks[text]["sources"]:
                            unique_blocks[text]["sources"].append(filename)
                            
                        # Keep highest rating if multiple exist
                        if block.get("rating", 0) > unique_blocks[text].get("rating", 0):
                            unique_blocks[text]["rating"] = block.get("rating", 0)
                            unique_blocks[text]["batch_rating"] = block.get("batch_rating", False)
        
        return list(unique_blocks.values())
    
    def update_ratings(self, content_blocks: List[Dict[str, Any]]) -> bool:
        """Update ratings in the data based on the provided content blocks.
        
        Args:
            content_blocks: List of content blocks with updated ratings.
            
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            # Create a mapping of text to updated ratings
            updated_ratings = {}
            for block in content_blocks:
                text = block.get("text", "")
                if text:
                    updated_ratings[text] = {
                        "rating": block.get("rating", 0),
                        "batch_rating": block.get("batch_rating", False),
                        "tags": block.get("tags", [])
                    }
            
            # Update ratings in the data structure
            changes_made = False
            for filename, file_data in self.data.items():
                # Skip metadata and invalid structures
                if not isinstance(file_data, dict) or "content" not in file_data:
                    continue
                
                paragraphs = file_data.get("content", {}).get("paragraphs", [])
                for paragraph in paragraphs:
                    for sentence in paragraph.get("sentences", []):
                        text = sentence.get("text", "")
                        if text in updated_ratings:
                            # Check if there's a change to avoid unnecessary updates
                            old_rating = sentence.get("rating", 0)
                            new_rating = updated_ratings[text]["rating"]
                            
                            if old_rating != new_rating or sentence.get("batch_rating", False) != updated_ratings[text]["batch_rating"]:
                                sentence["rating"] = new_rating
                                sentence["batch_rating"] = updated_ratings[text]["batch_rating"]
                                changes_made = True
                                
                            # Copy tags if they've been updated
                            if "user_edited_tags" not in sentence and sorted(sentence.get("tags", [])) != sorted(updated_ratings[text]["tags"]):
                                sentence["tags"] = updated_ratings[text]["tags"].copy()
                                changes_made = True
            
            # Save if changes were made
            if changes_made:
                return self.save_data()
            else:
                print("No changes to ratings detected.")
                return True
                
        except Exception as e:
            print(f"Error updating ratings: {e}")
            traceback.print_exc()
            return False
    
    def add_content_from_processed_file(self, processed_file: str) -> bool:
        """Add content from a processed file to the current data.
        
        This method is used by the TextProcessor to add newly processed
        content to the content database.
        
        Args:
            processed_file: Path to the processed file.
            
        Returns:
            bool: True if addition was successful, False otherwise.
        """
        try:
            # Skip if the processed file doesn't exist
            if not os.path.exists(processed_file):
                return False
                
            # Load processed data
            with open(processed_file, "r") as f:
                processed_data = json.load(f)
                
            # Skip metadata key
            if "metadata" in processed_data:
                del processed_data["metadata"]
                
            # Track if we've added any new content
            new_content_added = False
            
            # Process each file's content
            for filename, file_data in processed_data.items():
                # Skip if not a valid content structure
                if not isinstance(file_data, dict) or "content" not in file_data:
                    continue
                    
                # Check if we already have this file's content
                if filename in self.data and not isinstance(self.data[filename], dict):
                    # Convert to proper structure if needed
                    self.data[filename] = {
                        "filename": filename,
                        "content": {"paragraphs": []}
                    }
                
                # Check if this is a new file
                if filename not in self.data:
                    # Add this file's content to our data
                    self.data[filename] = file_data
                    new_content_added = True
                    print(f"Added new content from file: {filename}")
                else:
                    # Check for new paragraphs/sentences in this file
                    existing_texts = set()
                    
                    # Get existing texts for de-duplication
                    if "content" in self.data[filename] and "paragraphs" in self.data[filename]["content"]:
                        for p in self.data[filename]["content"]["paragraphs"]:
                            for s in p.get("sentences", []):
                                existing_texts.add(s.get("text", "").strip())
                    
                    # Update last_modified and processed_date
                    if "last_modified" in file_data:
                        self.data[filename]["last_modified"] = file_data["last_modified"]
                    if "processed_date" in file_data:
                        self.data[filename]["processed_date"] = file_data["processed_date"]
                    
                    # Check for new content
                    for paragraph in file_data.get("content", {}).get("paragraphs", []):
                        new_paragraph = False
                        for sentence in paragraph.get("sentences", []):
                            text = sentence.get("text", "").strip()
                            if text and text not in existing_texts:
                                # We found a new sentence
                                new_content_added = True
                                new_paragraph = True
                                existing_texts.add(text)
                        
                        # Add new paragraph if it contains new content
                        if new_paragraph:
                            if "content" not in self.data[filename]:
                                self.data[filename]["content"] = {"paragraphs": []}
                            if "paragraphs" not in self.data[filename]["content"]:
                                self.data[filename]["content"]["paragraphs"] = []
                            
                            self.data[filename]["content"]["paragraphs"].append(paragraph)
            
            # Save if we added new content
            if new_content_added:
                print("New content was found and imported. Saving updated content data.")
                return self.save_data()
            else:
                print("No new content found to import.")
                return True
                
        except Exception as e:
            print(f"Error importing from processed file: {e}")
            traceback.print_exc()
            return False
            
    def get_all_unrated_blocks(self) -> List[Dict[str, Any]]:
        """Get all unrated content blocks.
        
        Returns:
            List[Dict[str, Any]]: List of unrated content blocks.
        """
        all_blocks = self.get_content_blocks()
        return [block for block in all_blocks if block.get("rating", 0) == 0]

    def get_file_content(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get content for a specific file.
        
        Args:
            filename: Name of the file to get content for.
            
        Returns:
            Optional[Dict[str, Any]]: File content data or None if not found.
        """
        if filename in self.data and isinstance(self.data[filename], dict):
            return self.data[filename]
        return None

    def file_exists(self, filename: str) -> bool:
        """Check if a file exists in the data.
        
        Args:
            filename: Name of the file to check.
            
        Returns:
            bool: True if the file exists, False otherwise.
        """
        return filename in self.data and isinstance(self.data[filename], dict)
    
    def get_last_modified(self, filename: str) -> str:
        """Get the last modified timestamp for a file.
        
        Args:
            filename: Name of the file to get the timestamp for.
            
        Returns:
            str: Last modified timestamp as ISO format string or empty string if not found.
        """
        if self.file_exists(filename) and "last_modified" in self.data[filename]:
            return self.data[filename]["last_modified"]
        return ""
    
    def get_content(self, filename: str) -> Dict[str, Any]:
        """Get the content for a file.
        
        Args:
            filename: Name of the file to get content for.
            
        Returns:
            Dict[str, Any]: Content data structure or empty dict if not found.
        """
        if self.file_exists(filename) and "content" in self.data[filename]:
            return self.data[filename]["content"]
        return {}
    
    def add_file(self, filename: str, file_data: Dict[str, Any]) -> bool:
        """Add or update a file in the data.
        
        Args:
            filename: Name of the file to add/update.
            file_data: File data structure.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.data[filename] = file_data
            return self.save_data()
        except Exception as e:
            print(f"Error adding file {filename}: {e}")
            traceback.print_exc()
            return False
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata from the data.
        
        Returns:
            Dict[str, Any]: Metadata dictionary or empty dict if not found.
        """
        if "metadata" in self.data and isinstance(self.data["metadata"], dict):
            return self.data["metadata"]
        return {}
    
    def update_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Update metadata in the data.
        
        Args:
            metadata: Metadata dictionary.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Merge with existing metadata if it exists
            if "metadata" in self.data and isinstance(self.data["metadata"], dict):
                self.data["metadata"].update(metadata)
            else:
                self.data["metadata"] = metadata
            return self.save_data()
        except Exception as e:
            print(f"Error updating metadata: {e}")
            traceback.print_exc()
            return False

    def add_or_update_file(self, filename: str, file_data: Dict[str, Any]) -> bool:
        """Add or update a file in the content data.
        
        Args:
            filename: Name of the file to add or update.
            file_data: File data to add or update.
            
        Returns:
            bool: True if addition/update was successful, False otherwise.
        """
        try:
            self.data[filename] = file_data
            return self.save_data()
        except Exception as e:
            print(f"Error adding/updating file {filename}: {e}")
            traceback.print_exc()
            return False

    def get_canonical_file(self) -> str:
        """Get the path to the canonical content file.
        
        Returns:
            str: Path to the canonical content file.
        """
        return self.content_file
