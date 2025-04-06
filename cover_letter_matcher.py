#!/usr/bin/env python3
"""
Cover Letter Matcher - A tool to match high-rated sentences to job requirements.

This tool finds the best sentences from your cover letters that match the tags
in a job posting, organized by priority level.
"""

import json
import spacy
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Constants
JOBS_FILE = "analyzed_jobs.json"
SENTENCES_FILE = "processed_cover_letters.json"
REPORTS_DIR = "job-reports"
MIN_RATING_THRESHOLD = 6.0  # Minimum rating to consider a sentence

# Configurable scoring weights
SCORING_WEIGHTS = {
    "high_priority_match": 0.5,    # Weight for matching a high priority tag
    "medium_priority_match": 0.3,  # Weight for matching a medium priority tag
    "low_priority_match": 0.2,     # Weight for matching a low priority tag
    "multi_tag_bonus": 0.1,        # Additional bonus for each tag after the first
}

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
    Find sentences that match job tags, organized by sentence rather than tag.
    
    Args:
        job: Job data from analyzed_jobs.json
        all_sentences: List of all rated sentences
        
    Returns:
        Dict with matched sentences and tag information
    """
    # Get tags by priority level
    high_priority_tags = job.get("tags", {}).get("high_priority", [])
    medium_priority_tags = job.get("tags", {}).get("medium_priority", [])
    low_priority_tags = job.get("tags", {}).get("low_priority", [])
    
    # Track sentences that match tags
    sentence_matches = defaultdict(dict)  # Maps sentence text to match info
    
    # Find all tag matches for each sentence
    for sentence in all_sentences:
        sentence_text = sentence.get("text", "")
        sentence_tags = set(sentence.get("tags", []))
        
        # Initialize match info
        if sentence_text not in sentence_matches:
            sentence_matches[sentence_text] = {
                "text": sentence_text,
                "rating": sentence.get("rating", 0),
                "source": sentence.get("source", ""),
                "matched_tags": {
                    "high": [],
                    "medium": [],
                    "low": []
                },
                "all_tags": sentence.get("tags", [])
            }
        
        # Check which job tags this sentence matches
        for tag in high_priority_tags:
            if tag in sentence_tags:
                sentence_matches[sentence_text]["matched_tags"]["high"].append(tag)
        
        for tag in medium_priority_tags:
            if tag in sentence_tags:
                sentence_matches[sentence_text]["matched_tags"]["medium"].append(tag)
        
        for tag in low_priority_tags:
            if tag in sentence_tags:
                sentence_matches[sentence_text]["matched_tags"]["low"].append(tag)
    
    # Calculate scores for each sentence
    scored_sentences = []
    
    for text, match_info in sentence_matches.items():
        # Skip sentences that don't match any tags
        if not any(match_info["matched_tags"].values()):
            continue
            
        # Count matches by priority
        high_match_count = len(match_info["matched_tags"]["high"])
        medium_match_count = len(match_info["matched_tags"]["medium"])
        low_match_count = len(match_info["matched_tags"]["low"])
        total_match_count = high_match_count + medium_match_count + low_match_count
        
        # Skip sentences that don't match any tags
        if total_match_count == 0:
            continue
        
        # Calculate score using configurable weights
        base_score = match_info["rating"]
        priority_bonus = (
            high_match_count * SCORING_WEIGHTS["high_priority_match"] +
            medium_match_count * SCORING_WEIGHTS["medium_priority_match"] +
            low_match_count * SCORING_WEIGHTS["low_priority_match"]
        )
        
        # Add bonus for matching multiple tags (only for tags after the first)
        multi_tag_bonus = 0
        if total_match_count > 1:
            multi_tag_bonus = (total_match_count - 1) * SCORING_WEIGHTS["multi_tag_bonus"]
        
        final_score = base_score + priority_bonus + multi_tag_bonus
        
        # Create scored sentence entry
        scored_sentence = {
            "text": text,
            "rating": match_info["rating"],
            "score": final_score,
            "matched_tags": match_info["matched_tags"],
            "all_tags": match_info["all_tags"],
            "match_count": total_match_count,
            "source": match_info["source"]
        }
        
        scored_sentences.append(scored_sentence)
    
    # Sort sentences by score (highest first)
    scored_sentences.sort(key=lambda x: x["score"], reverse=True)
    
    # Organize matches for traditional tag-based display (for backward compatibility)
    tag_based_matches = {
        "high_priority": [],
        "medium_priority": [],
        "low_priority": []
    }
    
    # Process high priority tags
    for tag in high_priority_tags:
        tag_matches = []
        for sentence in scored_sentences:
            if tag in sentence["matched_tags"]["high"]:
                tag_matches.append({
                    "text": sentence["text"],
                    "rating": sentence["rating"],
                    "adjusted_score": sentence["score"],
                    "tags": sentence["all_tags"],
                    "match_count": sentence["match_count"],
                    "source": sentence["source"]
                })
        
        tag_based_matches["high_priority"].append({
            "tag": tag,
            "sentences": tag_matches[:3]  # Only keep top 3 matches for each tag
        })
    
    # Process medium priority tags
    for tag in medium_priority_tags:
        tag_matches = []
        for sentence in scored_sentences:
            if tag in sentence["matched_tags"]["medium"]:
                tag_matches.append({
                    "text": sentence["text"],
                    "rating": sentence["rating"],
                    "adjusted_score": sentence["score"],
                    "tags": sentence["all_tags"],
                    "match_count": sentence["match_count"],
                    "source": sentence["source"]
                })
        
        tag_based_matches["medium_priority"].append({
            "tag": tag,
            "sentences": tag_matches[:3]  # Only keep top 3 matches for each tag
        })
    
    # Process low priority tags
    for tag in low_priority_tags:
        tag_matches = []
        for sentence in scored_sentences:
            if tag in sentence["matched_tags"]["low"]:
                tag_matches.append({
                    "text": sentence["text"],
                    "rating": sentence["rating"],
                    "adjusted_score": sentence["score"],
                    "tags": sentence["all_tags"],
                    "match_count": sentence["match_count"],
                    "source": sentence["source"]
                })
        
        tag_based_matches["low_priority"].append({
            "tag": tag,
            "sentences": tag_matches[:3]  # Only keep top 3 matches for each tag
        })
    
    return {
        "sentence_based": scored_sentences,
        "tag_based": tag_based_matches
    }

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
        matches: Dict with matched sentences
    """
    display_job_info(job)
    
    # Display sentences sorted by score
    print("\nMATCHING SENTENCES (sorted by score):")
    
    for i, sentence in enumerate(matches["sentence_based"], 1):
        # Format matched tags by priority
        high_tags = ", ".join(sentence["matched_tags"]["high"])
        medium_tags = ", ".join(sentence["matched_tags"]["medium"])
        low_tags = ", ".join(sentence["matched_tags"]["low"])
        
        tag_info = []
        if high_tags:
            tag_info.append(f"High: {high_tags}")
        if medium_tags:
            tag_info.append(f"Medium: {medium_tags}")
        if low_tags:
            tag_info.append(f"Low: {low_tags}")
        
        tags_str = "; ".join(tag_info)
        
        print(f"\n{i}. \"{sentence['text']}\"")
        print(f"   Score: {sentence['score']:.1f} (Rating: {sentence['rating']}, Matches: {sentence['match_count']})")
        print(f"   Tags: {tags_str}")
    
    print("\n" + "="*80)

def generate_markdown_report(job: Dict, matches: Dict) -> str:
    """
    Generate a markdown report for the job matches.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        
    Returns:
        str: Markdown content
    """
    md = f"# Job Match Report: {job['job_title']} at {job['org_name']}\n\n"
    md += f"**Job ID:** {job['id']}  \n"
    md += f"**URL:** {job['url']}  \n"
    md += f"**Date Scraped:** {job['date_scraped']}  \n\n"
    
    md += f"## Summary\n\n{job['summary']}\n\n"
    
    md += "## Priority Tags\n\n"
    md += "### High Priority\n\n"
    for tag in job.get("tags", {}).get("high_priority", []):
        md += f"- {tag}\n"
    
    md += "\n### Medium Priority\n\n"
    for tag in job.get("tags", {}).get("medium_priority", []):
        md += f"- {tag}\n"
    
    md += "\n### Low Priority\n\n"
    for tag in job.get("tags", {}).get("low_priority", []):
        md += f"- {tag}\n"
    
    # New sentence-centric format
    md += "\n## Matching Sentences\n\n"
    
    for i, sentence in enumerate(matches["sentence_based"], 1):
        md += f"{i}. \"{sentence['text']}\"\n\n"
        md += f"   Score: {sentence['score']:.1f} (Rating: {sentence['rating']})\n"
        
        # List matched tags by priority
        if sentence["matched_tags"]["high"]:
            md += f"   High Priority Tags: {', '.join(sentence['matched_tags']['high'])}\n"
        if sentence["matched_tags"]["medium"]:
            md += f"   Medium Priority Tags: {', '.join(sentence['matched_tags']['medium'])}\n"
        if sentence["matched_tags"]["low"]:
            md += f"   Low Priority Tags: {', '.join(sentence['matched_tags']['low'])}\n"
        
        md += "\n"
    
    # Add information about scoring weights
    md += "## Scoring Information\n\n"
    md += "Sentences are scored using the following formula:\n\n"
    md += "```\nScore = Base Rating + Priority Bonuses + Multi-Tag Bonus\n```\n\n"
    md += "Where:\n"
    md += f"- High Priority Match: +{SCORING_WEIGHTS['high_priority_match']} per tag\n"
    md += f"- Medium Priority Match: +{SCORING_WEIGHTS['medium_priority_match']} per tag\n"
    md += f"- Low Priority Match: +{SCORING_WEIGHTS['low_priority_match']} per tag\n"
    md += f"- Multi-Tag Bonus: +{SCORING_WEIGHTS['multi_tag_bonus']} for each additional tag after the first\n"
    
    return md

def save_markdown_report(job: Dict, matches: Dict) -> str:
    """
    Save a markdown report for the job matches.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        
    Returns:
        str: Path to the saved report
    """
    # Create reports directory if it doesn't exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Generate filename from org name and job title
    org_name = job['org_name'].lower().replace(' ', '-')
    job_title = job['job_title'].lower().replace(' ', '-')
    filename = f"{org_name}-{job_title}-{job['id']}.md"
    filepath = os.path.join(REPORTS_DIR, filename)
    
    # Generate markdown content
    md_content = generate_markdown_report(job, matches)
    
    # Save to file
    with open(filepath, 'w') as f:
        f.write(md_content)
    
    return filepath

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
    parser.add_argument("--report", action="store_true", help="Generate a markdown report")
    
    # Add arguments for configurable scoring weights
    parser.add_argument("--high-weight", type=float, help=f"Weight for high priority matches (default: {SCORING_WEIGHTS['high_priority_match']})")
    parser.add_argument("--medium-weight", type=float, help=f"Weight for medium priority matches (default: {SCORING_WEIGHTS['medium_priority_match']})")
    parser.add_argument("--low-weight", type=float, help=f"Weight for low priority matches (default: {SCORING_WEIGHTS['low_priority_match']})")
    parser.add_argument("--multi-tag-bonus", type=float, help=f"Bonus for each additional tag match (default: {SCORING_WEIGHTS['multi_tag_bonus']})")
    
    args = parser.parse_args()
    
    # Update scoring weights if provided
    if args.high_weight is not None:
        SCORING_WEIGHTS["high_priority_match"] = args.high_weight
    if args.medium_weight is not None:
        SCORING_WEIGHTS["medium_priority_match"] = args.medium_weight
    if args.low_weight is not None:
        SCORING_WEIGHTS["low_priority_match"] = args.low_weight
    if args.multi_tag_bonus is not None:
        SCORING_WEIGHTS["multi_tag_bonus"] = args.multi_tag_bonus
    
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
    
    # Generate report if requested or display matches
    if args.report:
        report_path = save_markdown_report(target_job, matches)
        print(f"\nReport generated: {report_path}")
    else:
        display_matches(target_job, matches)
    
    print(f"\nCompleted analysis for Job #{args.job_id}: {target_job['job_title']} at {target_job['org_name']}")
    print(f"To generate a markdown report, run with --report flag")

if __name__ == "__main__":
    main()
