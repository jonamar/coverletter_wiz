#!/usr/bin/env python3
"""
Cover Letter Matcher - A tool to match high-rated sentences to job requirements.

This tool finds the best sentences from your cover letters that match the tags
in a job posting, organized by priority level.

Usage:
    python cover_letter_matcher.py [options]

Options:
    --job-id INT             Sequential ID of the job to analyze
    --list                   List all available jobs with their IDs
    --report                 Generate a markdown report in the job-reports directory
    --cover-letter           Include a cover letter draft in the report (requires --report)
                             or display it in the terminal
    --print-prompt           Print the LLM prompt instead of generating a cover letter
                             (works with --cover-letter)
    
    # Scoring weights (customize how sentences are scored):
    --high-weight FLOAT      Weight for high priority tag matches (default: 0.5)
    --medium-weight FLOAT    Weight for medium priority tag matches (default: 0.3)
    --low-weight FLOAT       Weight for low priority tag matches (default: 0.2)
    --multi-tag-bonus FLOAT  Bonus for each additional tag match (default: 0.1)

Examples:
    # List all available jobs
    python cover_letter_matcher.py --list
    
    # Analyze a specific job and display matches in the terminal
    python cover_letter_matcher.py --job-id 1
    
    # Generate a markdown report for a job
    python cover_letter_matcher.py --job-id 1 --report
    
    # Generate a report with a cover letter draft
    python cover_letter_matcher.py --job-id 1 --report --cover-letter
    
    # Print the LLM prompt that would be used to generate a cover letter
    python cover_letter_matcher.py --job-id 1 --cover-letter --print-prompt
    
    # Generate a report with the LLM prompt instead of the cover letter
    python cover_letter_matcher.py --job-id 1 --report --cover-letter --print-prompt
    
    # Customize scoring weights
    python cover_letter_matcher.py --job-id 1 --high-weight 0.7 --medium-weight 0.4
"""

import json
import spacy
import argparse
import os
import requests
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

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-r1:8b"  # Using the same model as in job_analyzer.py

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

def query_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """
    Query the Ollama API with a prompt.
    
    Args:
        prompt: The prompt to send to the Ollama API
        model: The model to use for generation
        
    Returns:
        The generated text
    """
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    except Exception as e:
        print(f"Error querying Ollama API: {e}")
        return f"Error: {e}"

def generate_cover_letter(job: Dict, matches: Dict, print_prompt_only: bool = False) -> str:
    """
    Generate a cover letter from matching sentences using Ollama.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        print_prompt_only: If True, return the prompt instead of querying Ollama
        
    Returns:
        str: Generated cover letter or prompt if print_prompt_only is True
    """
    # Extract top matching sentences (up to 20)
    top_sentences = [s["text"] for s in matches["sentence_based"][:20]]
    
    # Create a numbered list of the sentences for the LLM
    sentences_list = "\n".join([f"{i+1}. {s}" for i, s in enumerate(top_sentences)])
    
    # Build the prompt for the LLM
    prompt = f"""
You are a professional career coach helping to create a cover letter. 
I'm applying for the role of {job['job_title']} at {job['org_name']}.

JOB SUMMARY:
{job['summary']}

KEY JOB TAGS (in order of priority):
High Priority: {', '.join(job['tags']['high_priority'])}
Medium Priority: {', '.join(job['tags']['medium_priority'])}
Low Priority: {', '.join(job['tags']['low_priority'])}

Here are my best matching sentences from previous cover letters (ordered by relevance):
{sentences_list}

Please create a professional cover letter draft using the following template:
'''
Hello [name of org],

[note my values and their connection with the org]

[three sections, combining the best sentences into logical paragraph groupings, balancing flow and the prioritized tags. each section should have a simple heading.]

Onwards,
Jon
'''

Important instructions:
1. Adjust the original sentences as minimally as possible
2. Combine sentences that flow well together into three logical sections
3. Add simple headings for each section
4. The first paragraph should connect my values with the organization's mission
5. Don't fabricate information - only use content from the provided sentences
6. Only output the cover letter, no explanations

Cover letter:
"""
    
    # If print_prompt_only is True, just return the prompt
    if print_prompt_only:
        return prompt
    
    # Otherwise, query the LLM
    cover_letter = query_ollama(prompt)
    
    return cover_letter

def generate_markdown_report(job: Dict, matches: Dict, include_cover_letter: bool = False, print_prompt_only: bool = False) -> str:
    """
    Generate a markdown report for the job matches.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        include_cover_letter: Whether to include a cover letter draft
        print_prompt_only: If True, print the prompt instead of generating a cover letter
        
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
    
    # Add cover letter section if requested
    if include_cover_letter:
        md += "\n## Cover Letter Draft\n\n"
        
        if print_prompt_only:
            md += "The following is the prompt that would be sent to the LLM to generate a cover letter.\n"
            md += "You can use this prompt with any LLM of your choice.\n\n"
            md += "```\n"  # Format as code block for better formatting
            
            prompt = generate_cover_letter(job, matches, print_prompt_only=True)
            md += prompt + "\n"
            
            md += "```\n\n"
        else:
            md += "The following is an automatically generated cover letter draft based on your top-scoring sentences.\n"
            md += "Feel free to edit and personalize it to better match your voice and specific circumstances.\n\n"
            md += "```\n"  # Format as code block for better formatting
            
            cover_letter = generate_cover_letter(job, matches)
            md += cover_letter + "\n"
            
            md += "```\n\n"
            md += "_Note: This draft was generated by AI and may need human review and customization._\n"
    
    return md

def save_markdown_report(job: Dict, matches: Dict, include_cover_letter: bool = False, print_prompt_only: bool = False) -> str:
    """
    Save a markdown report for the job matches.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        include_cover_letter: Whether to include a cover letter draft
        print_prompt_only: If True, print the prompt instead of generating a cover letter
        
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
    md_content = generate_markdown_report(job, matches, include_cover_letter, print_prompt_only)
    
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
    parser.add_argument("--cover-letter", action="store_true", help="Include a cover letter draft in the report")
    parser.add_argument("--print-prompt", action="store_true", help="Print the LLM prompt instead of generating a cover letter")
    
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
        report_path = save_markdown_report(target_job, matches, args.cover_letter, args.print_prompt)
        print(f"\nReport generated: {report_path}")
    # If just print-prompt is requested (without report), print the prompt to terminal
    elif args.print_prompt and args.cover_letter:
        prompt = generate_cover_letter(target_job, matches, print_prompt_only=True)
        print("\nLLM PROMPT FOR COVER LETTER GENERATION:")
        print("="*80)
        print(prompt)
        print("="*80)
    else:
        display_matches(target_job, matches)
    
    print(f"\nCompleted analysis for Job #{args.job_id}: {target_job['job_title']} at {target_job['org_name']}")
    
    # Print helpful usage information
    if args.report and not args.cover_letter:
        print(f"To include a cover letter draft in the report, add the --cover-letter flag")
    if not args.print_prompt and (args.report or args.cover_letter):
        print(f"To print the LLM prompt instead of generating a cover letter, add the --print-prompt flag")

if __name__ == "__main__":
    main()
