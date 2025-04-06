#!/usr/bin/env python3
"""
Cover Letter Matcher - A tool to match high-rated sentences to job requirements.

This tool finds the best sentences from your cover letters that match the tags
in a job posting, organized by priority level.
"""

import json
import spacy
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set

# Constants
JOBS_FILE = "analyzed_jobs.json"
SENTENCES_FILE = "processed_cover_letters.json"
MIN_RATING_THRESHOLD = 6.0  # Minimum rating to consider a sentence

def load_jobs(jobs_file: str) -> Dict:
    """Load job data from JSON file."""
    with open(jobs_file, 'r') as f:
        return json.load(f)

def load_sentences(sentences_file: str) -> Dict:
    """Load sentence data from JSON file."""
    with open(sentences_file, 'r') as f:
        return json.load(f)

def update_job_ids(jobs_data: Dict) -> Dict:
    """
    Update job IDs to be sequential numbers instead of UUIDs.
    
    Args:
        jobs_data: Original jobs data
        
    Returns:
        Updated jobs data with sequential IDs
    """
    updated_jobs = {"jobs": []}
    id_mapping = {}  # Map original UUIDs to new sequential IDs
    
    for i, job in enumerate(jobs_data.get("jobs", []), 1):
        # Store mapping from UUID to sequential ID
        original_id = job.get("id", "")
        id_mapping[original_id] = i
        
        # Create updated job with sequential ID
        updated_job = job.copy()
        updated_job["id"] = i
        updated_job["original_id"] = original_id  # Keep original ID for reference
        updated_jobs["jobs"].append(updated_job)
    
    # Store the ID mapping in the jobs data
    updated_jobs["id_mapping"] = id_mapping
    
    return updated_jobs

def get_all_rated_sentences(sentences_data: Dict) -> List[Dict]:
    """
    Extract all sentences with ratings from the processed cover letters.
    
    Args:
        sentences_data: Data from processed_cover_letters.json
        
    Returns:
        List of sentence objects with text, rating, and tags
    """
    all_sentences = []
    
    # Skip metadata keys
    for file_key, file_data in sentences_data.items():
        if not isinstance(file_data, dict) or "content" not in file_data:
            continue
        
        paragraphs = file_data.get("content", {}).get("paragraphs", [])
        
        for paragraph in paragraphs:
            sentences = paragraph.get("sentences", [])
            
            for sentence in sentences:
                # Only include sentences with ratings above threshold
                rating = sentence.get("rating", 0)
                if rating >= MIN_RATING_THRESHOLD:
                    all_sentences.append({
                        "text": sentence.get("text", ""),
                        "rating": rating,
                        "tags": sentence.get("tags", []),
                        "source": file_key
                    })
    
    # Sort by rating (highest first)
    all_sentences.sort(key=lambda x: x.get("rating", 0), reverse=True)
    return all_sentences

def find_matching_sentences(job: Dict, all_sentences: List[Dict]) -> Dict:
    """
    Find sentences that match job tags, organized by priority level.
    
    Args:
        job: Job data from analyzed_jobs.json
        all_sentences: List of all rated sentences
        
    Returns:
        Dict with matched sentences by priority level
    """
    matches = {
        "high_priority": [],
        "medium_priority": [],
        "low_priority": []
    }
    
    # Get tags by priority level
    high_priority_tags = job.get("tags", {}).get("high_priority", [])
    medium_priority_tags = job.get("tags", {}).get("medium_priority", [])
    low_priority_tags = job.get("tags", {}).get("low_priority", [])
    
    # Find matches for each priority level
    for tag in high_priority_tags:
        tag_matches = []
        for sentence in all_sentences:
            if tag in sentence.get("tags", []):
                tag_matches.append(sentence)
        
        # Only keep top 3 matches for each tag
        matches["high_priority"].append({
            "tag": tag,
            "sentences": tag_matches[:3]
        })
    
    for tag in medium_priority_tags:
        tag_matches = []
        for sentence in all_sentences:
            if tag in sentence.get("tags", []):
                tag_matches.append(sentence)
        
        matches["medium_priority"].append({
            "tag": tag,
            "sentences": tag_matches[:3]
        })
    
    for tag in low_priority_tags:
        tag_matches = []
        for sentence in all_sentences:
            if tag in sentence.get("tags", []):
                tag_matches.append(sentence)
        
        matches["low_priority"].append({
            "tag": tag,
            "sentences": tag_matches[:3]
        })
    
    return matches

def display_job_info(job: Dict) -> None:
    """Display basic job information."""
    print("\n" + "="*80)
    print(f"JOB #{job['id']}: {job['job_title']} at {job['org_name']}")
    print("="*80)
    print(f"\nSUMMARY: {job['summary']}")
    print(f"URL: {job['url']}")
    print(f"DATE SCRAPED: {job['date_scraped']}")
    
    print("\nPRIORITY TAGS:")
    print("  HIGH PRIORITY:", ", ".join(job.get("tags", {}).get("high_priority", [])))
    print("  MEDIUM PRIORITY:", ", ".join(job.get("tags", {}).get("medium_priority", [])))
    print("  LOW PRIORITY:", ", ".join(job.get("tags", {}).get("low_priority", [])))
    print("-"*80)

def display_matches(job: Dict, matches: Dict) -> None:
    """
    Display matched sentences in a readable format.
    
    Args:
        job: Job data
        matches: Dict with matched sentences by priority
    """
    display_job_info(job)
    
    # Display high priority matches
    print("\nHIGH PRIORITY MATCHES:")
    for tag_match in matches["high_priority"]:
        tag = tag_match["tag"]
        sentences = tag_match["sentences"]
        
        if sentences:
            print(f"\n  TAG: {tag}")
            for i, sentence in enumerate(sentences, 1):
                print(f"    {i}. \"{sentence['text']}\" (Rating: {sentence['rating']})")
        else:
            print(f"\n  TAG: {tag} - No matching sentences found")
    
    # Display medium priority matches
    print("\nMEDIUM PRIORITY MATCHES:")
    for tag_match in matches["medium_priority"]:
        tag = tag_match["tag"]
        sentences = tag_match["sentences"]
        
        if sentences:
            print(f"\n  TAG: {tag}")
            for i, sentence in enumerate(sentences, 1):
                print(f"    {i}. \"{sentence['text']}\" (Rating: {sentence['rating']})")
        else:
            print(f"\n  TAG: {tag} - No matching sentences found")
    
    # Display low priority matches
    print("\nLOW PRIORITY MATCHES:")
    for tag_match in matches["low_priority"]:
        tag = tag_match["tag"]
        sentences = tag_match["sentences"]
        
        if sentences:
            print(f"\n  TAG: {tag}")
            for i, sentence in enumerate(sentences, 1):
                print(f"    {i}. \"{sentence['text']}\" (Rating: {sentence['rating']})")
        else:
            print(f"\n  TAG: {tag} - No matching sentences found")
    
    print("\n" + "="*80)

def list_available_jobs(jobs_data: Dict) -> None:
    """List all available jobs with their sequential IDs."""
    print("\nAVAILABLE JOBS:")
    print("-"*80)
    for job in jobs_data.get("jobs", []):
        print(f"Job #{job['id']}: {job['job_title']} at {job['org_name']}")
    print("-"*80)

def main():
    """Main function to run the cover letter matcher."""
    parser = argparse.ArgumentParser(description="Match cover letter sentences to job requirements")
    parser.add_argument("--job-id", type=int, help="Sequential ID of the job to analyze")
    parser.add_argument("--list", action="store_true", help="List all available jobs")
    args = parser.parse_args()
    
    # Load job data
    try:
        jobs_data = load_jobs(JOBS_FILE)
        # Update job IDs to be sequential
        jobs_data = update_job_ids(jobs_data)
    except Exception as e:
        print(f"Error loading job data: {e}")
        return
    
    # List available jobs if requested
    if args.list:
        list_available_jobs(jobs_data)
        return
    
    # If no job ID provided, list jobs and exit
    if not args.job_id:
        print("No job ID provided. Here are the available jobs:")
        list_available_jobs(jobs_data)
        print("\nUse --job-id <ID> to analyze a specific job")
        return
    
    # Find the job with the specified ID
    target_job = None
    for job in jobs_data.get("jobs", []):
        if job.get("id") == args.job_id:
            target_job = job
            break
    
    if not target_job:
        print(f"No job found with ID {args.job_id}")
        list_available_jobs(jobs_data)
        return
    
    # Load sentence data
    try:
        sentences_data = load_sentences(SENTENCES_FILE)
    except Exception as e:
        print(f"Error loading sentence data: {e}")
        return
    
    # Get all rated sentences
    all_sentences = get_all_rated_sentences(sentences_data)
    
    # Find matching sentences
    matches = find_matching_sentences(target_job, all_sentences)
    
    # Display matches
    display_matches(target_job, matches)
    
    print(f"\nCompleted analysis for Job #{args.job_id}: {target_job['job_title']} at {target_job['org_name']}")

if __name__ == "__main__":
    main()
