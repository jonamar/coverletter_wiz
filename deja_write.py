import os
import json
import uuid
import spacy
from pathlib import Path

def process_cover_letter(content):
    """Process a single cover letter's content and return structured data."""
    nlp = spacy.load("en_core_web_sm")
    
    # Split into paragraphs based on double newlines
    paragraphs = content.split("\n\n")
    
    # Process each paragraph
    processed_paragraphs = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # Tokenize into sentences using spaCy
        doc = nlp(paragraph)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Store paragraph and its sentences
        processed_paragraphs.append({
            "paragraph": paragraph,
            "sentences": sentences
        })
    
    return processed_paragraphs

def process_cover_letters_batch(archive_dir, output_file):
    """Process all cover letters in the archive directory and save to a single JSON file."""
    archive_path = Path(archive_dir)
    output_path = Path(output_file)
    
    # Load existing data if output file exists
    existing_data = {}
    if output_path.exists():
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
    
    # Get all text files from archive
    text_files = list(archive_path.glob('*.txt'))
    
    # Track changes
    new_files_processed = 0
    
    for text_file in text_files:
        source_file = text_file.name
        
        # Skip if already processed and content hasn't changed
        if source_file in existing_data:
            file_mtime = os.path.getmtime(text_file)
            if file_mtime <= existing_data[source_file].get('last_modified', 0):
                continue
        
        # Read and process the file
        with open(text_file, 'r') as f:
            content = f.read()
        
        # Create entry
        entry = {
            "id": str(uuid.uuid4()),
            "source_file": source_file,
            "paragraphs": process_cover_letter(content),
            "tags": [],
            "ratings": {},
            "notes": "",
            "last_modified": os.path.getmtime(text_file)
        }
        
        # Update the data
        existing_data[source_file] = entry
        new_files_processed += 1
    
    # Save all data to the output file
    with open(output_path, 'w') as f:
        json.dump(existing_data, f, indent=4)
    
    return new_files_processed

if __name__ == "__main__":
    archive_dir = "text-archive"
    output_file = "processed_cover_letters.json"
    
    if not os.path.exists(archive_dir):
        print(f"Error: Archive directory '{archive_dir}' not found.")
        exit(1)
    
    new_files = process_cover_letters_batch(archive_dir, output_file)
    print(f"Processing complete. {new_files} new or updated files processed.")
    print(f"All processed cover letters saved to: {output_file}")