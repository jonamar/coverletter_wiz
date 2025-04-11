"""
DejaWrite - Cover Letter Processing Tool

This script processes cover letters from text files and extracts structured data including:
- Paragraphs and sentences/sentence groups
- Automatically generated tags using spaCy's semantic similarity and keyword matching
- Sentence ratings (preserved across processing runs)

The script organizes cover letter content for later use in the cover letter generation
pipeline, working alongside job_analyzer.py, sentence_rater.py, and cover_letter_matcher.py.

Requirements:
- Python 3.6+
- spaCy with en_core_web_sm or en_core_web_md model
- categories.yaml file with tag categories

CLI Usage:
    python deja_write.py write
        Process all text files in the text-archive directory and update the JSON file.
        New files will be added, and existing files will be updated while preserving ratings.

    python deja_write.py write --force
        Force reprocessing of all files even if unchanged.

Output:
    The script generates a processed_cover_letters.json file containing structured data
    from all cover letters with tags and ratings.
"""

import os
import json
import uuid
import spacy
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from spacy_utils import identify_sentence_groups, assign_tags_with_spacy

def load_categories(yaml_file):
    """Load categories from a YAML file."""
    try:
        with open(yaml_file, "r") as file:
            categories = yaml.safe_load(file)
        return categories
    except Exception as e:
        print(f"Error loading categories from YAML file: {e}")
        return None

def get_category_tags(text, categories, max_tags=5, context=""):
    """
    Assign tags to text using spaCy-based semantic similarity and keyword matching.
    Returns tags with confidence scores.
    """
    try:
        # Get tags with confidence scores
        tags_with_scores = assign_tags_with_spacy(text, categories, max_tags)
        return tags_with_scores
    except Exception as e:
        print(f"Error in spaCy-based tagging: {e}")
        return []

def process_cover_letter(content):
    """Process a single cover letter's content and return structured data."""
    nlp = spacy.load("en_core_web_sm")
    
    # Split into paragraphs based on double newlines
    paragraphs = content.split("\n\n")
    
    # Process each paragraph
    processed_paragraphs = []
    document_tags = set()  # Use a set to avoid duplicate tags
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # Get paragraph-level tags with confidence scores
        paragraph_tags_with_scores = get_category_tags(paragraph, categories)
        
        # Extract just the tag names for the paragraph
        paragraph_tags = [tag["name"] for tag in paragraph_tags_with_scores]
        document_tags.update(paragraph_tags)
        
        # Use spaCy to identify sentence groups within the paragraph
        sentence_groups = identify_sentence_groups(paragraph)
        processed_sentences = []
        
        # Process each sentence or sentence group
        for group in sentence_groups:
            group_text = group["text"]
            is_group = group["is_sentence_group"]
            
            # Get tags with confidence scores for the sentence or group
            group_tags_with_scores = get_category_tags(
                group_text, 
                categories, 
                max_tags=5,  # Get more tags initially to have better selection
                context=f"From paragraph: {paragraph}"
            )
            
            # Sort by confidence score and take top 2 as primary tags
            group_tags_with_scores.sort(key=lambda x: x["confidence"], reverse=True)
            primary_tags = [tag["name"] for tag in group_tags_with_scores[:2]]
            
            # Include additional tags but mark them as secondary (for backward compatibility)
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
                "component_sentences": group.get("component_sentences", [])
            })
        
        # Store paragraph with its sentences/groups and tags
        processed_paragraphs.append({
            "text": paragraph,
            "tags": paragraph_tags,
            "sentences": processed_sentences
        })
    
    return processed_paragraphs, list(document_tags)

def preserve_existing_sentence_ratings(existing_data, new_paragraphs):
    """
    Preserve existing ratings for sentences and sentence groups when updating the JSON file.
    
    Args:
        existing_data: The existing data structure with ratings
        new_paragraphs: The newly processed paragraphs
        
    Returns:
        Updated paragraphs with preserved ratings
    """
    # Create a mapping of sentence text to its existing rating
    sentence_ratings = {}
    
    # Extract all existing sentences and their ratings
    for filename, file_data in existing_data.items():
        # Skip non-file entries like metadata
        if not isinstance(file_data, dict) or "content" not in file_data:
            continue
            
        paragraphs = file_data["content"].get("paragraphs", [])
        
        for paragraph in paragraphs:
            sentences = paragraph.get("sentences", [])
            
            for sentence in sentences:
                text = sentence.get("text", "").strip()
                
                if not text:
                    continue
                    
                # Store rating and other important fields
                if "rating" in sentence:
                    if text not in sentence_ratings or sentence_ratings[text].get("rating", 0) == 0:
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

def process_cover_letters_batch(archive_dir, output_file, force_reprocess=False):
    """Process all cover letters in the archive directory and save to a single JSON file."""
    archive_path = Path(archive_dir)
    output_path = Path(output_file)
    
    # Load categories
    global categories  # Make categories available to all functions
    categories = load_categories("categories.yaml")
    if not categories:
        print("Failed to load categories. Proceeding without automatic tagging.")
        return
    
    # Initialize new data structure
    processed_data = {}
    
    # Load existing data if output file exists
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
            processed_data = existing_data
    else:
        existing_data = {}
    
    # Get all text files from archive
    text_files = list(archive_path.glob('*.txt'))
    
    # Track changes
    new_files_processed = 0
    
    for text_file in text_files:
        source_file = text_file.name
        
        # Skip if already processed and content hasn't changed, unless force_reprocess is True
        if not force_reprocess and source_file in processed_data:
            file_mtime = os.path.getmtime(text_file)
            if file_mtime <= processed_data[source_file].get("metadata", {}).get("last_modified", 0):
                continue
        
        # Read and process the file
        with open(text_file, 'r') as f:
            content = f.read()
        
        # Process paragraphs and get document tags
        processed_paragraphs, document_tags = process_cover_letter(content)
        
        # Preserve existing ratings if this is an update to an existing file
        if existing_data:
            processed_paragraphs = preserve_existing_sentence_ratings(existing_data, processed_paragraphs)
        
        # Create entry with new structure
        entry = {
            "id": str(uuid.uuid4()) if source_file not in processed_data else processed_data[source_file].get("id", str(uuid.uuid4())),
            "source_file": source_file,
            "content": {
                "paragraphs": processed_paragraphs,
                "document_tags": document_tags
            },
            "metadata": {
                "last_modified": os.path.getmtime(text_file),
                "ratings": {},
                "notes": ""
            }
        }
        
        # Add to processed data
        processed_data[source_file] = entry
        new_files_processed += 1
    
    # Preserve metadata fields from existing data
    if "batch_ratings_done" in existing_data:
        processed_data["batch_ratings_done"] = existing_data["batch_ratings_done"]
    if "refinement_done" in existing_data:
        processed_data["refinement_done"] = existing_data["refinement_done"]
    if "filtered_sentences" in existing_data:
        processed_data["filtered_sentences"] = existing_data["filtered_sentences"]
    
    # Update last processed date
    processed_data["last_processed_date"] = datetime.now().isoformat()
    
    # Save all processed data to JSON file
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    
    print(f"Processing complete. {new_files_processed} new or updated files processed.")
    print(f"All processed cover letters saved to: {output_file}")
    
    return new_files_processed

if __name__ == "__main__":
    import sys
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process cover letters and save to JSON")
    parser.add_argument("command", choices=["write"], help="Command to execute")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all files even if unchanged")
    
    # Parse arguments
    args = parser.parse_args()
    
    archive_dir = "text-archive"
    output_file = "processed_cover_letters.json"
    
    if args.command == "write":
        new_files = process_cover_letters_batch(archive_dir, output_file, force_reprocess=args.force)
    else:
        print(f"Unknown command: {args.command}")
        print("Usage: python deja_write.py write [--force]")