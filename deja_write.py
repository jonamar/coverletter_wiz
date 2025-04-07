"""
DejaWrite - Cover Letter Processing Tool

This script processes cover letters from text files and extracts structured data including:
- Paragraphs and sentences
- Automatically generated tags using a local LLM (deepseek-r1:8b via Ollama)
- Sentence ratings (preserved across processing runs)

The script organizes cover letter content for later use in the cover letter generation
pipeline, working alongside job_analyzer.py, sentence_rater.py, and cover_letter_matcher.py.

Requirements:
- Python 3.6+
- spaCy with en_core_web_sm model
- Ollama with deepseek-r1:8b model installed
- categories.yaml file with tag categories

CLI Usage:
    python deja_write.py write
        Process all text files in the text-archive directory and update the JSON file.
        New files will be added, and existing files will be updated while preserving ratings.

Output:
    The script generates a processed_cover_letters.json file containing structured data
    from all cover letters with tags and ratings.
"""

import os
import json
import uuid
import spacy
import yaml
import ollama
from pathlib import Path
from datetime import datetime

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
    """Query the local LLM to classify the text with relevant category tags."""
    try:
        # Prepare the list of all possible tags
        all_tags = []
        for category, items in categories.items():
            all_tags.extend(items)
        tags_string = ", ".join(all_tags)

        # Prepare the prompt for the LLM
        prompt = f"""Let's think about this step by step:

1. First, read and understand the following text{' and its context' if context else ''}:
{f'Context: {context}' if context else ''}
Text: "{text}"

2. Here are the available tags to choose from:
{tags_string}

3. Think through these questions:
- What are the main themes in this text?
- What skills or competencies are demonstrated?
- What outcomes or impacts are mentioned?
- What values are expressed?

4. Based on your analysis, select up to {max_tags} most relevant tags from the provided list.

Output your final tags in this exact format:
TAGS: tag1, tag2, tag3
"""
        # Query the LLM
        response = ollama.generate(model="deepseek-r1:8b", prompt=prompt)
        completion = ""
        for chunk in response:
            if isinstance(chunk, tuple) and chunk[0] == "response":
                completion += chunk[1]
        
        # Extract tags from the response - look for the TAGS: prefix
        tags = []
        for line in completion.split('\n'):
            if line.strip().startswith('TAGS:'):
                tags_part = line.replace('TAGS:', '').strip()
                tags = [tag.strip() for tag in tags_part.split(',')]
                break
        
        # Filter out any tags that aren't in our original list
        valid_tags = [tag for tag in tags if tag in all_tags]
        return valid_tags[:max_tags]

    except Exception as e:
        print(f"Error connecting to the local LLM: {e}")
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
        
        # Tokenize into sentences using spaCy
        doc = nlp(paragraph)
        processed_sentences = []
        
        # Get paragraph-level tags
        paragraph_tags = get_category_tags(paragraph, categories)
        document_tags.update(paragraph_tags)
        
        # Process each sentence
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            # Get sentence-level tags, including paragraph as context
            sentence_tags = get_category_tags(
                sentence_text, 
                categories, 
                max_tags=3,  # Fewer tags per sentence
                context=f"From paragraph: {paragraph}"
            )
            
            processed_sentences.append({
                "text": sentence_text,
                "tags": sentence_tags
            })
        
        # Store paragraph with its sentences and tags
        processed_paragraphs.append({
            "text": paragraph,
            "tags": paragraph_tags,
            "sentences": processed_sentences
        })
    
    return processed_paragraphs, list(document_tags)

def preserve_existing_sentence_ratings(existing_data, new_paragraphs):
    """
    Preserve existing ratings for sentences when updating the JSON file.
    
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

def process_cover_letters_batch(archive_dir, output_file):
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
        
        # Skip if already processed and content hasn't changed
        if source_file in processed_data:
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
        
        # Update the data
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
    
    # Save all data to the output file
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    
    return new_files_processed

if __name__ == "__main__":
    import sys
    
    archive_dir = "text-archive"
    output_file = "processed_cover_letters.json"
    
    # Check for the "write" command
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        if not os.path.exists(archive_dir):
            print(f"Error: Archive directory '{archive_dir}' not found.")
            exit(1)
        
        new_files = process_cover_letters_batch(archive_dir, output_file)
        print(f"Processing complete. {new_files} new or updated files processed.")
        print(f"All processed cover letters saved to: {output_file}")
    else:
        print("Usage: python deja_write.py write")
        print("This will process all text files in the text-archive directory and update the JSON file.")