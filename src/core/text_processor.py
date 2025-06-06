#!/usr/bin/env python3
"""
Text Processor - Core module for processing cover letter text files.

This module handles processing text files from the text-archive directory,
extracting content blocks, generating tags using spaCy, and preserving ratings
across processing runs.
"""

from __future__ import annotations

import os
import sys
import json
import yaml
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set, Optional

import spacy

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.spacy_utils import assign_tags_with_spacy, identify_sentence_groups
from src.core.data_manager import DataManager
from src.config import DATA_DIR

class TextProcessor:
    """Core class for processing cover letter text files.
    
    This class handles reading text files, extracting content blocks,
    generating tags using spaCy, and preserving ratings across processing runs.
    """
    
    def __init__(self, spacy_model: str = "en_core_web_md") -> None:
        """Initialize the TextProcessor.
        
        Args:
            spacy_model: spaCy model to use for NLP processing.
            
        Raises:
            OSError: If the specified spaCy model cannot be loaded.
        """
        # Set default paths using external data directory
        self.archive_dir = os.path.join(DATA_DIR, "text-archive")
        self.spacy_model = spacy_model
        self.categories = self._load_categories()
        
        # Initialize data manager for direct access to content database
        self.data_manager = DataManager()
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            print(f"Error: spaCy model '{spacy_model}' not found.")
            print(f"Downloading {spacy_model}...")
            try:
                os.system(f"python -m spacy download {spacy_model}")
                self.nlp = spacy.load(spacy_model)
                print(f"Successfully downloaded and loaded {spacy_model}")
            except Exception as e:
                print(f"Failed to download spaCy model: {e}")
                print("Using en_core_web_lg as fallback")
                try:
                    self.nlp = spacy.load("en_core_web_lg")
                except:
                    print("Error: Could not load any spaCy model.")
                    print("Please install a spaCy model with: python -m spacy download en_core_web_lg")
                    raise
    
    def _load_categories(self, yaml_file: Optional[str] = None) -> Dict[str, Any]:
        """Load categories from a YAML file.
        
        Attempts to locate a categories YAML file in multiple potential locations.
        
        Args:
            yaml_file: Path to the YAML file containing categories.
            
        Returns:
            Categories data structure dictionary.
            
        Raises:
            yaml.YAMLError: If the YAML file is malformed.
        """
        from src.config import DATA_DIR
        
        # Try multiple possible locations for the categories file
        possible_paths = [
            yaml_file,  # Use provided path if available
            os.path.join(DATA_DIR, "config/categories.yaml"),  # External data directory
            "data/config/categories.yaml",  # Legacy standard location
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                        "data/config/categories.yaml"),  # Absolute path
            os.path.join(os.getcwd(), "data/config/categories.yaml")  # Current working directory
        ]
        
        for path in possible_paths:
            if not path:
                continue
                
            try:
                if os.path.exists(path):
                    with open(path, "r") as file:
                        categories = yaml.safe_load(file)
                    print(f"Loaded categories from {path}")
                    return categories
            except Exception as e:
                print(f"Error loading categories from {path}: {e}")
        
        # If we get here, we couldn't load categories
        print("Warning: Could not load categories. Using empty categories.")
        return {"categories": []}
    
    def process_text_files(self, force_reprocess: bool = False) -> Optional[Dict[str, Any]]:
        """Process all text files in the archive directory.
        
        Reads each text file, extracts content blocks, generates tags,
        and preserves existing ratings.
        
        Args:
            force_reprocess: Force reprocessing of all files even if unchanged.
            
        Returns:
            Processing statistics or None if error occurs.
            
        Raises:
            OSError: If there are issues with file access or the archive directory.
        """
        try:
            # Check if archive directory exists
            if not os.path.exists(self.archive_dir):
                print(f"Error: Archive directory {self.archive_dir} does not exist.")
                return None
            
            # Get list of text files
            text_files = [f for f in os.listdir(self.archive_dir) 
                         if f.endswith('.txt') and os.path.isfile(os.path.join(self.archive_dir, f))]
            
            if not text_files:
                print(f"No text files found in {self.archive_dir}")
                return None
            
            print(f"Found {len(text_files)} text files to process.")
            
            # Track processing statistics
            stats = {
                "files_processed": 0,
                "new_files": 0,
                "updated_files": 0,
                "unchanged_files": 0,
                "total_blocks": 0
            }
            
            # Process each file
            for filename in text_files:
                file_path = os.path.join(self.archive_dir, filename)
                
                # Check if file needs processing
                if not force_reprocess and self.data_manager.file_exists(filename):
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    mtime_str = datetime.fromtimestamp(mtime).isoformat()
                    
                    # Check if file has been modified since last processing
                    if self.data_manager.get_last_modified(filename) >= mtime_str:
                        print(f"Skipping {filename} (unchanged)")
                        stats["unchanged_files"] += 1
                        continue
                
                # Process the file
                print(f"Processing {filename}...")
                
                try:
                    # Read file content
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Process the content
                    processed_paragraphs, document_tags = self._process_content(content)
                    
                    # Preserve existing ratings
                    if self.data_manager.file_exists(filename):
                        processed_paragraphs = self._preserve_ratings(
                            self.data_manager.get_content(filename).get("paragraphs", []),
                            processed_paragraphs
                        )
                        stats["updated_files"] += 1
                    else:
                        stats["new_files"] += 1
                    
                    # Ensure all blocks have IDs
                    for paragraph in processed_paragraphs:
                        for sentence in paragraph.get("sentences", []):
                            if "id" not in sentence:
                                sentence["id"] = self.data_manager.generate_block_id()
                    
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    mtime_str = datetime.fromtimestamp(mtime).isoformat()
                    
                    # Update data structure
                    self.data_manager.add_file(filename, {
                        "filename": filename,
                        "last_modified": mtime_str,
                        "processed_date": datetime.now().isoformat(),
                        "content": {
                            "paragraphs": processed_paragraphs,
                            "document_tags": document_tags
                        }
                    })
                    
                    # Count content blocks
                    for paragraph in processed_paragraphs:
                        stats["total_blocks"] += len(paragraph.get("sentences", []))
                    
                    stats["files_processed"] += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    traceback.print_exc()
            
            # Update metadata
            self.data_manager.update_metadata({
                "version": "1.0",
                "created": self.data_manager.get_metadata().get("created", datetime.now().isoformat()),
                "updated": datetime.now().isoformat(),
                "files_count": stats["files_processed"],
                "content_blocks_count": stats["total_blocks"]
            })
            
            # Ensure all blocks have IDs (including previously existing ones)
            self.data_manager.ensure_block_ids()
            
            return stats
            
        except Exception as e:
            print(f"Error processing text files: {e}")
            traceback.print_exc()
            return None
    
    def _process_content(self, content: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Process text content and extract structured data.
        
        Analyzes text content using spaCy to identify paragraphs, sentences, tags, 
        and sentence groups for further processing.
        
        Args:
            content: Text content to process.
            
        Returns:
            A tuple containing (processed_paragraphs, document_tags) where:
              - processed_paragraphs is a list of paragraph dictionaries with sentences and tags
              - document_tags is a list of all unique tags identified in the document
        """
        # Split into paragraphs based on double newlines
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        # Process each paragraph
        processed_paragraphs = []
        document_tags: Set[str] = set()  # Use a set to avoid duplicate tags
        
        for paragraph in paragraphs:
            # Get paragraph-level tags with confidence scores
            paragraph_tags_with_scores = assign_tags_with_spacy(
                paragraph, 
                self.categories, 
                max_tags=5,
                nlp=self.nlp
            )
            
            # Extract just the tag names for the paragraph
            paragraph_tags = [tag["name"] for tag in paragraph_tags_with_scores]
            document_tags.update(paragraph_tags)
            
            # Use spaCy to identify sentence groups within the paragraph
            sentence_groups = identify_sentence_groups(paragraph, nlp=self.nlp)
            processed_sentences = []
            
            # Process each sentence or sentence group
            for group in sentence_groups:
                group_text = group["text"]
                is_group = group["is_sentence_group"]
                
                # Get tags with confidence scores for the sentence or group
                group_tags_with_scores = assign_tags_with_spacy(
                    group_text, 
                    self.categories, 
                    max_tags=5,
                    nlp=self.nlp,
                    context=f"From paragraph: {paragraph[:100]}..."
                )
                
                # Sort by confidence score and take top 2 as primary tags
                group_tags_with_scores.sort(key=lambda x: x["confidence"], reverse=True)
                primary_tags = [tag["name"] for tag in group_tags_with_scores[:2]]
                
                # Include additional tags but mark them as secondary
                all_tags = primary_tags.copy()
                for tag in group_tags_with_scores[2:]:
                    if tag["name"] not in all_tags:
                        all_tags.append(tag["name"])
                
                # Add the processed sentence or group
                processed_sentences.append({
                    "text": group_text,
                    "tags": all_tags,
                    "primary_tags": primary_tags,
                    "tag_scores": {tag["name"]: tag["confidence"] for tag in group_tags_with_scores},
                    "is_sentence_group": is_group,
                    "component_sentences": group.get("component_sentences", []),
                    "rating": 0.0  # Default rating
                })
            
            # Store paragraph with its sentences/groups and tags
            processed_paragraphs.append({
                "text": paragraph,
                "tags": paragraph_tags,
                "sentences": processed_sentences
            })
        
        return processed_paragraphs, list(document_tags)
    
    def _preserve_ratings(self, existing_paragraphs: List[Dict[str, Any]], new_paragraphs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preserve existing ratings when updating processed content.
        
        Matches sentences/content blocks from previous processing runs to maintain
        user-assigned ratings across file updates.
        
        Args:
            existing_paragraphs: Existing processed paragraphs with ratings.
            new_paragraphs: Newly processed paragraphs.
            
        Returns:
            Updated paragraphs with preserved ratings from existing data.
        """
        # Create a mapping of existing sentences by text for fast lookup
        existing_sentences = {}
        for paragraph in existing_paragraphs:
            for sentence in paragraph.get("sentences", []):
                # Use the text as the key for lookup
                existing_sentences[sentence.get("text", "")] = sentence
        
        # Now go through new paragraphs and copy ratings for matching sentences
        for paragraph in new_paragraphs:
            for sentence in paragraph.get("sentences", []):
                # Look for this sentence in the existing data
                text = sentence.get("text", "")
                if text in existing_sentences:
                    # Copy rating and other user-entered data
                    existing = existing_sentences[text]
                    sentence["rating"] = existing.get("rating", 0.0)
                    sentence["batch_rating"] = existing.get("batch_rating", False)
                    
                    # Preserve any manual tag edits by comparing tag sets
                    if "user_edited_tags" in existing:
                        sentence["tags"] = existing.get("tags", [])
                        sentence["user_edited_tags"] = True
        
        return new_paragraphs
