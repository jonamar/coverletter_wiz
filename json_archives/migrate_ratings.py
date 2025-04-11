#!/usr/bin/env python3
"""
Rating Migration Script

This script:
1. Archives the existing processed_cover_letters.json file
2. Runs deja_write.py to generate fresh data
3. Migrates ratings from archived data to the new sentence groups
4. Prepares the data for a new round of sentence rating
"""

import json
import os
import shutil
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import spacy
from collections import defaultdict

# Constants
JSON_FILE = "processed_cover_letters.json"
ARCHIVE_DIR = "json_archives"
MIN_SIMILARITY_THRESHOLD = 0.85  # Threshold for considering sentences similar

# Load spaCy model for text similarity
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Warning: en_core_web_md not found. Using en_core_web_sm instead.")
    print("For better results, run: python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_sm")

def archive_json_file() -> str:
    """
    Archive the existing JSON file with a timestamp.
    
    Returns:
        Path to the archived file
    """
    # Create archive directory if it doesn't exist
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    
    # Generate archive filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_filename = f"{ARCHIVE_DIR}/processed_cover_letters_{timestamp}.json"
    
    # Copy the file to the archive
    if os.path.exists(JSON_FILE):
        shutil.copy2(JSON_FILE, archive_filename)
        print(f"Archived existing data to: {archive_filename}")
        return archive_filename
    else:
        print(f"No existing {JSON_FILE} found to archive.")
        return ""

def regenerate_json_data():
    """Run deja_write.py to generate fresh data."""
    print("\nRegenerating cover letter data...")
    result = subprocess.run(["python", "deja_write.py", "write", "--force"], 
                           capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Successfully regenerated data.")
        print(result.stdout.strip())
    else:
        print("Error regenerating data:")
        print(result.stderr.strip())
        exit(1)

def extract_all_sentences(data: Dict) -> List[Dict]:
    """
    Extract all sentences from the JSON data.
    
    Args:
        data: The JSON data containing cover letters
        
    Returns:
        List of all sentences with their ratings and metadata
    """
    all_sentences = []
    
    for file_key, file_data in data.items():
        if file_key == "last_processed_date":
            continue
            
        # Skip non-dictionary values
        if not isinstance(file_data, dict):
            continue
            
        # Access content -> paragraphs -> sentences
        content = file_data.get("content", {})
        if not isinstance(content, dict):
            continue
            
        paragraphs = content.get("paragraphs", [])
        for paragraph in paragraphs:
            for sentence in paragraph.get("sentences", []):
                # Add source file and paragraph info to each sentence
                sentence_copy = sentence.copy()
                sentence_copy["source_file"] = file_key
                sentence_copy["paragraph"] = paragraph.get("text", "")
                all_sentences.append(sentence_copy)
    
    return all_sentences

def find_best_match(sentence: str, candidates: List[Dict]) -> Tuple[Dict, float]:
    """
    Find the best matching sentence from candidates based on text similarity.
    
    Args:
        sentence: The sentence text to match
        candidates: List of candidate sentences
        
    Returns:
        Tuple of (best matching sentence, similarity score)
    """
    sentence_doc = nlp(sentence)
    best_match = None
    best_score = 0
    
    for candidate in candidates:
        candidate_text = candidate.get("text", "")
        candidate_doc = nlp(candidate_text)
        
        # Skip if either doesn't have vector
        if not sentence_doc.has_vector or not candidate_doc.has_vector:
            continue
            
        similarity = sentence_doc.similarity(candidate_doc)
        
        if similarity > best_score:
            best_score = similarity
            best_match = candidate
    
    return best_match, best_score

def migrate_ratings(archive_file: str, current_file: str):
    """
    Migrate ratings from archived data to new sentence groups.
    
    Args:
        archive_file: Path to the archived JSON file
        current_file: Path to the current JSON file
    """
    print("\nMigrating ratings from archived data...")
    
    # Load archived and current data
    with open(archive_file, 'r') as f:
        archived_data = json.load(f)
    
    with open(current_file, 'r') as f:
        current_data = json.load(f)
    
    # Extract all sentences from archived data
    archived_sentences = extract_all_sentences(archived_data)
    
    # Filter to only include sentences with ratings
    rated_sentences = [s for s in archived_sentences if s.get("rating", 0) > 0]
    print(f"Found {len(rated_sentences)} previously rated sentences/groups")
    
    # Track migration statistics
    migrated_count = 0
    group_ratings_count = 0
    
    # Process each file in the current data
    for file_key, file_data in current_data.items():
        if file_key == "last_processed_date":
            continue
            
        # Skip non-dictionary values
        if not isinstance(file_data, dict):
            continue
            
        # Access content -> paragraphs -> sentences
        content = file_data.get("content", {})
        if not isinstance(content, dict):
            continue
            
        paragraphs = content.get("paragraphs", [])
        for paragraph in paragraphs:
            for sentence in paragraph.get("sentences", []):
                # For individual sentences, try to find direct matches
                if not sentence.get("is_sentence_group", False):
                    best_match, similarity = find_best_match(sentence["text"], rated_sentences)
                    
                    if best_match and similarity >= MIN_SIMILARITY_THRESHOLD:
                        # Transfer rating and metadata
                        sentence["rating"] = best_match.get("rating", 0)
                        sentence["last_rated"] = best_match.get("last_rated", "")
                        sentence["batch_rating"] = best_match.get("batch_rating", True)
                        migrated_count += 1
                
                # For sentence groups, calculate average rating from component sentences
                else:
                    component_sentences = sentence.get("component_sentences", [])
                    if not component_sentences:
                        continue
                        
                    # Find matches for each component sentence
                    ratings = []
                    for comp_sent in component_sentences:
                        best_match, similarity = find_best_match(comp_sent, rated_sentences)
                        if best_match and similarity >= MIN_SIMILARITY_THRESHOLD:
                            ratings.append(best_match.get("rating", 0))
                    
                    # If we found ratings for any component sentences, calculate average
                    if ratings:
                        avg_rating = sum(ratings) / len(ratings)
                        sentence["rating"] = avg_rating
                        sentence["last_rated"] = datetime.datetime.now().isoformat()
                        sentence["batch_rating"] = True
                        group_ratings_count += 1
    
    # Save the updated data
    with open(current_file, 'w') as f:
        json.dump(current_data, f, indent=4)
    
    print(f"Migration complete:")
    print(f"- Migrated {migrated_count} individual sentence ratings")
    print(f"- Created {group_ratings_count} averaged ratings for sentence groups")

def main():
    """Main function to run the migration process."""
    print("=== Cover Letter Rating Migration ===")
    
    # Step 1: Archive existing JSON file
    archive_file = archive_json_file()
    
    # Step 2: Regenerate JSON data
    regenerate_json_data()
    
    # Step 3: Migrate ratings if we have an archive
    if archive_file:
        migrate_ratings(archive_file, JSON_FILE)
    
    # Step 4: Provide instructions for next steps
    print("\nNext Steps:")
    print("1. Review the migrated data")
    print("2. Run the sentence rater to continue rating: python sentence_rater.py")
    print("3. Once satisfied with ratings, use the cover letter matcher")

if __name__ == "__main__":
    main()
