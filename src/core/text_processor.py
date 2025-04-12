#!/usr/bin/env python3
"""
Text Processor - Core module for processing cover letter text files.

This module handles processing text files from the text-archive directory,
extracting content blocks, and generating tags using spaCy.
"""

import os
import json
import uuid
import spacy
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
import traceback

from src.utils.spacy_utils import identify_sentence_groups, assign_tags_with_spacy

class TextProcessor:
    """
    Core class for processing cover letter text files.
    
    This class handles reading text files, extracting content blocks,
    generating tags using spaCy, and preserving ratings across processing runs.
    """
    
    def __init__(self, archive_dir: str, output_file: str, spacy_model: str = "en_core_web_md"):
        """
        Initialize the TextProcessor.
        
        Args:
            archive_dir: Directory containing text files to process
            output_file: Output JSON file for processed content
            spacy_model: spaCy model to use for NLP processing
        """
        self.archive_dir = archive_dir
        self.output_file = output_file
        self.spacy_model = spacy_model
        self.categories = self._load_categories()
        
        # Load existing data if available
        self.existing_data = self._load_existing_data()
        
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
                print("Using en_core_web_sm as fallback")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except:
                    print("Error: Could not load any spaCy model.")
                    print("Please install a spaCy model with: python -m spacy download en_core_web_sm")
                    raise
    
    def _load_categories(self, yaml_file: str = None) -> Dict:
        """
        Load categories from a YAML file.
        
        Args:
            yaml_file: Path to the YAML file containing categories
            
        Returns:
            Dict: Categories data structure
        """
        # Try multiple possible locations for the categories file
        possible_paths = [
            yaml_file,  # Use provided path if available
            "data/config/categories.yaml",  # Standard location
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
    
    def _load_existing_data(self) -> Dict:
        """
        Load existing processed data if available.
        
        Returns:
            Dict: Existing processed data or empty structure
        """
        try:
            # First check the output file path
            if os.path.exists(self.output_file):
                with open(self.output_file, "r") as f:
                    data = json.load(f)
                print(f"Loaded existing data from {self.output_file}")
                return data
                
            # If not found, check for the file in the project root directory
            root_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                    "processed_cover_letters.json")
            if os.path.exists(root_file):
                with open(root_file, "r") as f:
                    data = json.load(f)
                print(f"Loaded existing data from {root_file}")
                return data
                
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in existing file: {e}")
            print(f"Creating backup of corrupted file...")
            backup_file = f"{self.output_file}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
            try:
                os.rename(self.output_file, backup_file)
                print(f"Backup created: {backup_file}")
            except Exception as e:
                print(f"Error creating backup: {e}")
        except Exception as e:
            print(f"Error loading existing data: {e}")
        
        # Return empty structure if no existing data or error
        return {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "updated": datetime.now().isoformat()
            }
        }
    
    def process_text_files(self, force_reprocess: bool = False) -> Optional[Dict]:
        """
        Process all text files in the archive directory.
        
        Args:
            force_reprocess: Force reprocessing of all files even if unchanged
            
        Returns:
            Optional[Dict]: Processing statistics or None if error
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
                if not force_reprocess and filename in self.existing_data:
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    mtime_str = datetime.fromtimestamp(mtime).isoformat()
                    
                    # Check if file has been modified since last processing
                    if "last_modified" in self.existing_data[filename]:
                        last_modified = self.existing_data[filename]["last_modified"]
                        
                        # Skip if file hasn't been modified
                        if last_modified >= mtime_str:
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
                    if filename in self.existing_data:
                        processed_paragraphs = self._preserve_ratings(
                            self.existing_data[filename].get("content", {}).get("paragraphs", []),
                            processed_paragraphs
                        )
                        stats["updated_files"] += 1
                    else:
                        stats["new_files"] += 1
                    
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    mtime_str = datetime.fromtimestamp(mtime).isoformat()
                    
                    # Update data structure
                    self.existing_data[filename] = {
                        "filename": filename,
                        "last_modified": mtime_str,
                        "processed_date": datetime.now().isoformat(),
                        "content": {
                            "paragraphs": processed_paragraphs,
                            "document_tags": document_tags
                        }
                    }
                    
                    # Count content blocks
                    for paragraph in processed_paragraphs:
                        stats["total_blocks"] += len(paragraph.get("sentences", []))
                    
                    stats["files_processed"] += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    traceback.print_exc()
            
            # Update metadata
            self.existing_data["metadata"] = {
                "version": "1.0",
                "created": self.existing_data.get("metadata", {}).get("created", datetime.now().isoformat()),
                "updated": datetime.now().isoformat(),
                "files_count": stats["files_processed"],
                "content_blocks_count": stats["total_blocks"]
            }
            
            # Save updated data
            self._save_data()
            
            return stats
            
        except Exception as e:
            print(f"Error processing text files: {e}")
            traceback.print_exc()
            return None
    
    def _process_content(self, content: str) -> Tuple[List[Dict], List[str]]:
        """
        Process text content and extract structured data.
        
        Args:
            content: Text content to process
            
        Returns:
            Tuple[List[Dict], List[str]]: Processed paragraphs and document tags
        """
        # Split into paragraphs based on double newlines
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        
        # Process each paragraph
        processed_paragraphs = []
        document_tags = set()  # Use a set to avoid duplicate tags
        
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
    
    def _preserve_ratings(self, existing_paragraphs: List[Dict], new_paragraphs: List[Dict]) -> List[Dict]:
        """
        Preserve existing ratings when updating processed content.
        
        Args:
            existing_paragraphs: Existing processed paragraphs with ratings
            new_paragraphs: Newly processed paragraphs
            
        Returns:
            List[Dict]: Updated paragraphs with preserved ratings
        """
        # Create a mapping of sentence text to its existing rating
        sentence_ratings = {}
        
        # Extract all existing sentences and their ratings
        for paragraph in existing_paragraphs:
            sentences = paragraph.get("sentences", [])
            
            for sentence in sentences:
                text = sentence.get("text", "").strip()
                
                if not text:
                    continue
                    
                # Store rating and other important fields
                if "rating" in sentence and sentence["rating"] > 0:
                    sentence_ratings[text] = {
                        "rating": sentence["rating"],
                        "last_rated": sentence.get("last_rated", ""),
                        "batch_rating": sentence.get("batch_rating", False)
                    }
        
        # Apply existing ratings to new paragraphs
        for paragraph in new_paragraphs:
            sentences = paragraph.get("sentences", [])
            
            for sentence in sentences:
                text = sentence.get("text", "").strip()
                
                if text in sentence_ratings:
                    # Copy over the existing rating and related fields
                    sentence["rating"] = sentence_ratings[text]["rating"]
                    if sentence_ratings[text].get("last_rated"):
                        sentence["last_rated"] = sentence_ratings[text]["last_rated"]
                    if sentence_ratings[text].get("batch_rating"):
                        sentence["batch_rating"] = sentence_ratings[text]["batch_rating"]
        
        return new_paragraphs
    
    def _save_data(self) -> bool:
        """
        Save processed data to the output file.
        
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            # Save data to file
            with open(self.output_file, "w") as f:
                json.dump(self.existing_data, f, indent=2)
            
            print(f"Saved processed data to {self.output_file}")
            return True
            
        except Exception as e:
            print(f"Error saving data to {self.output_file}: {e}")
            traceback.print_exc()
            return False
