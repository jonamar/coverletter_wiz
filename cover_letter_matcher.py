#!/usr/bin/env python3
"""
Cover Letter Matcher - A tool to match high-rated sentences and sentence groups to job requirements.

This tool finds the best sentences and sentence groups from your cover letters that match 
the tags in a job posting, organized by priority level.

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

    # LLM model selection:
    --llm, --model STR       LLM model to use (default: gemma3:12b)
    --multi-llm, --models STR  Run analysis with multiple LLM models and generate separate reports for each
    --list-models            List available Ollama models and exit

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
    
    # List available Ollama models
    python cover_letter_matcher.py --list-models
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
import re
import time

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
DEFAULT_LLM_MODEL = "gemma3:12b"  # Default model, can be overridden via CLI

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
    Extract all sentences and sentence groups with ratings from the processed cover letters.
    
    Args:
        sentences_data: Data from processed_cover_letters.json
        
    Returns:
        List of sentence objects with text, rating, tags, and group information
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
                    # Include fields that indicate if this is a sentence group
                    is_group = sentence.get("is_sentence_group", False)
                    component_sentences = sentence.get("component_sentences", [])
                    
                    all_sentences.append({
                        "text": sentence.get("text", ""),
                        "rating": rating,
                        "tags": sentence.get("tags", []),
                        "source": file_key,
                        "is_sentence_group": is_group,
                        "component_sentences": component_sentences
                    })
    
    # Sort by rating (highest first)
    all_sentences.sort(key=lambda x: x.get("rating", 0), reverse=True)
    return all_sentences

def find_matching_sentences(job: Dict, all_sentences: List[Dict]) -> Dict:
    """
    Find sentences and sentence groups that match job tags, organized by sentence rather than tag.
    
    Args:
        job: Job data from analyzed_jobs.json
        all_sentences: List of all rated sentences and sentence groups
        
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
                "all_tags": sentence.get("tags", []),
                "is_sentence_group": sentence.get("is_sentence_group", False),
                "component_sentences": sentence.get("component_sentences", [])
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
    
    # Calculate match scores
    for sentence_text, match_info in sentence_matches.items():
        # Count how many tags matched in each priority level
        high_matches = len(match_info["matched_tags"]["high"])
        medium_matches = len(match_info["matched_tags"]["medium"])
        low_matches = len(match_info["matched_tags"]["low"])
        
        # Apply weights based on priority
        score = (high_matches * SCORING_WEIGHTS["high_priority_match"] +
                 medium_matches * SCORING_WEIGHTS["medium_priority_match"] +
                 low_matches * SCORING_WEIGHTS["low_priority_match"])
        
        # Add a bonus for matching multiple tags
        total_matches = high_matches + medium_matches + low_matches
        if total_matches > 1:
            score += (total_matches - 1) * SCORING_WEIGHTS["multi_tag_bonus"]
        
        # Multiply by the sentence rating (0-10) to get a final score
        # This ensures that higher-rated sentences are preferred
        rating = match_info["rating"]
        final_score = score * (rating / 10)  # Normalize rating to 0-1
        
        match_info["score"] = final_score
        match_info["match_count"] = total_matches
    
    # Filter out sentences with no matches
    matched_sentences = {
        text: info for text, info in sentence_matches.items()
        if info.get("match_count", 0) > 0
    }
    
    # Sort by score
    sorted_matches = sorted(
        matched_sentences.values(),
        key=lambda x: x.get("score", 0),
        reverse=True
    )
    
    return {
        "matches": sorted_matches,
        "total": len(sorted_matches)
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
    matched_sentences = matches.get("matches", [])
    total_matches = matches.get("total", 0)
    
    print(f"\nFound {total_matches} matching sentences for this job.\n")
    
    for i, sentence in enumerate(matched_sentences[:10], 1):  # Show top 10
        score = sentence.get("score", 0)
        rating = sentence.get("rating", 0)
        
        print(f"{i}. Score: {score:.2f} (Rating: {rating}/10)")
        
        print(f"{sentence.get('text', '')}")
        
        # Print matched tags by priority
        matched_tags = sentence.get("matched_tags", {})
        if matched_tags.get("high"):
            print(f"   High Priority Matches: {', '.join(matched_tags['high'])}")
        if matched_tags.get("medium"):
            print(f"   Medium Priority Matches: {', '.join(matched_tags['medium'])}")
        if matched_tags.get("low"):
            print(f"   Low Priority Matches: {', '.join(matched_tags['low'])}")
        
        print()
    
    if total_matches > 10:
        print(f"...and {total_matches - 10} more matches. Generate a report to see all matches.")

def query_ollama(prompt: str, model: str = DEFAULT_LLM_MODEL) -> str:
    """
    Query the Ollama API with a prompt.
    
    Args:
        prompt: The prompt to send to the Ollama API
        model: The model to use for generation
        
    Returns:
        The generated text
    """
    try:
        print(f"Using LLM model: {model}")
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

def clean_cover_letter(text: str) -> str:
    """
    Clean the cover letter text by removing 'think' sections and other artifacts.
    
    Args:
        text: The raw cover letter text
        
    Returns:
        Cleaned cover letter text
    """
    # Remove <think>...</think> sections
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # Remove any other potential artifacts or placeholders
    cleaned_text = re.sub(r'\[Platform where you saw the advertisement\]', 'your job board', cleaned_text)
    
    # Clean up any excessive newlines
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    return cleaned_text.strip()

def generate_cover_letter(job: Dict, matches: Dict, print_prompt_only: bool = False, model: str = DEFAULT_LLM_MODEL) -> str:
    """
    Generate a cover letter from matching sentences using Ollama.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        print_prompt_only: If True, return the prompt instead of querying Ollama
        model: The LLM model to use for generation
        
    Returns:
        str: Generated cover letter or prompt if print_prompt_only is True
    """
    import time
    start_time = time.time()
    
    # Create an overview of the job for the LLM
    job_overview = f"JOB TITLE: {job['job_title']}\nORGANIZATION: {job['org_name']}\nSUMMARY: {job['summary']}"
    
    # Add job tags
    job_tags = "JOB TAGS:\n"
    job_tags += "High Priority: " + ", ".join(job['tags'].get('high_priority', [])) + "\n"
    job_tags += "Medium Priority: " + ", ".join(job['tags'].get('medium_priority', [])) + "\n"
    job_tags += "Low Priority: " + ", ".join(job['tags'].get('low_priority', [])) + "\n"
    
    # Gather matched sentences by score
    matched_sentences = []
    
    # Check if matches is a list or a dict with a different structure
    if isinstance(matches, list):
        # Handle the case where matches is a list
        for match in matches:
            if isinstance(match, dict) and match.get("score", 0) > 0:
                matched_sentences.append({
                    "text": match.get("text", ""),
                    "score": match.get("score", 0),
                    "is_group": match.get("is_sentence_group", False)
                })
    elif "matches" in matches:
        # Handle the case where matches has a "matches" key containing a list
        for match in matches.get("matches", []):
            if isinstance(match, dict) and match.get("score", 0) > 0:
                matched_sentences.append({
                    "text": match.get("text", ""),
                    "score": match.get("score", 0),
                    "is_group": match.get("is_sentence_group", False)
                })
    else:
        # Handle the case where matches is a dict mapping sentence text to match data
        for match_text, match_data in matches.items():
            if isinstance(match_data, dict) and match_data.get("score", 0) > 0:
                matched_sentences.append({
                    "text": match_text,
                    "score": match_data.get("score", 0),
                    "is_group": match_data.get("is_sentence_group", False)
                })
    
    # Sort sentences by score (highest first)
    matched_sentences.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Select top sentences (limit to 15)
    top_sentences = matched_sentences[:15]
    
    # Prepare the sentence list for the prompt
    sentence_list = ""
    for i, sent in enumerate(top_sentences, 1):
        sentence_list += f"{i}. (Score: {sent.get('score', 0):.2f}) {sent.get('text', '')}\n\n"
    
    # Create the improved prompt with emphasis on specific achievements
    prompt = f"""You are an expert cover letter writer. I need you to create a professional cover letter based on the following job information and my best sentences from previous cover letters that match this job's requirements:

{job_overview}

{job_tags}

MY BEST MATCHING SENTENCES (already sorted by relevance):
{sentence_list}

Please create a cohesive, professional cover letter using these sentences where appropriate. The cover letter should:
1. Have a proper greeting, introduction, body, and conclusion
2. Incorporate the provided sentences when they fit naturally, but rewrite or adapt them as needed
3. Add transitions and additional content to make the letter flow logically
4. Focus on how my skills and experience match the job requirements
5. Emphasize SPECIFIC ACHIEVEMENTS related to the high priority job tags
6. Include concrete examples of past work that demonstrate my capabilities
7. Keep the total length to about 300-400 words
8. Do not mention scores, rankings, or that sentences were provided to you
9. Sound natural and not repetitive
10. DO NOT include any "think" sections or notes to yourself in the output

Write the complete cover letter from scratch, but incorporate my best sentences as appropriate. Format it as a proper business letter with address blocks, date, greeting, and signature.
"""
    
    if print_prompt_only:
        return prompt
    
    # Query Ollama for the cover letter
    generated_text = query_ollama(prompt, model)
    
    # Clean the generated text
    cleaned_text = clean_cover_letter(generated_text)
    
    # Calculate and print the time taken
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Cover letter generation time with {model}: {elapsed_time:.2f} seconds")
    
    return cleaned_text

def generate_markdown_report(job: Dict, matches: Dict, include_cover_letter: bool = False, print_prompt_only: bool = False, model: str = DEFAULT_LLM_MODEL, keywords: List[str] = None) -> str:
    """
    Generate a markdown report for the job matches.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        include_cover_letter: Whether to include a cover letter draft
        print_prompt_only: If True, print the prompt instead of generating a cover letter
        model: The LLM model to use for generation
        keywords: Optional list of keywords to search for
        
    Returns:
        str: Markdown content
    """
    org_name = job.get("org_name", "Unknown Organization")
    job_title = job.get("job_title", "Unknown Position")
    job_id = job.get("id", "")
    job_url = job.get("url", "")
    summary = job.get("summary", "No summary available")
    
    high_priority = job.get("tags", {}).get("high_priority", [])
    medium_priority = job.get("tags", {}).get("medium_priority", [])
    low_priority = job.get("tags", {}).get("low_priority", [])
    
    matched_sentences = matches.get("matches", [])
    total_matches = matches.get("total", 0)
    
    # Start building the markdown content
    md_content = f"""# Job Analysis Report

## Job #{job_id}: {job_title} at {org_name}

**URL:** [{job_url}]({job_url})

**Summary:** {summary}

## Job Requirements

### High Priority
{', '.join(high_priority) if high_priority else 'None'}

### Medium Priority
{', '.join(medium_priority) if medium_priority else 'None'}

### Low Priority
{', '.join(low_priority) if low_priority else 'None'}

"""

    # Add top sentences by category section
    top_sentences = get_top_sentences_by_category(matches)
    md_content += "## Top Sentences by Category\n\n"
    
    # High priority sentences
    md_content += "### High Priority\n\n"
    if top_sentences["high"]:
        for i, sentence in enumerate(top_sentences["high"], 1):  # Show top 3
            score = sentence.get("score", 0)
            rating = sentence.get("rating", 0)
            md_content += f"**{i}. Score: {score:.2f} (Rating: {rating}/10)**\n\n"
            md_content += f"{sentence.get('text', '')}\n\n"
            # Show matched tags
            matched_tags = sentence.get("matched_tags", {}).get("high", [])
            if matched_tags:
                md_content += f"*Matched tags: {', '.join(matched_tags)}*\n\n"
            md_content += "---\n\n"
    else:
        md_content += "*No high priority matches found*\n\n"
    
    # Medium priority sentences
    md_content += "### Medium Priority\n\n"
    if top_sentences["medium"]:
        for i, sentence in enumerate(top_sentences["medium"], 1):  # Show top 3
            score = sentence.get("score", 0)
            rating = sentence.get("rating", 0)
            md_content += f"**{i}. Score: {score:.2f} (Rating: {rating}/10)**\n\n"
            md_content += f"{sentence.get('text', '')}\n\n"
            # Show matched tags
            matched_tags = sentence.get("matched_tags", {}).get("medium", [])
            if matched_tags:
                md_content += f"*Matched tags: {', '.join(matched_tags)}*\n\n"
            md_content += "---\n\n"
    else:
        md_content += "*No medium priority matches found*\n\n"
    
    # Low priority sentences
    md_content += "### Low Priority\n\n"
    if top_sentences["low"]:
        for i, sentence in enumerate(top_sentences["low"], 1):  # Show top 3
            score = sentence.get("score", 0)
            rating = sentence.get("rating", 0)
            md_content += f"**{i}. Score: {score:.2f} (Rating: {rating}/10)**\n\n"
            md_content += f"{sentence.get('text', '')}\n\n"
            # Show matched tags
            matched_tags = sentence.get("matched_tags", {}).get("low", [])
            if matched_tags:
                md_content += f"*Matched tags: {', '.join(matched_tags)}*\n\n"
            md_content += "---\n\n"
    else:
        md_content += "*No low priority matches found*\n\n"
    
    # Add keyword search results if keywords provided
    if keywords:
        keyword_matches = search_sentences_by_keywords(matches, keywords)
        md_content += f"## Keyword Search Results: {', '.join(keywords)}\n\n"
        
        if keyword_matches:
            for i, sentence in enumerate(keyword_matches[:10], 1):  # Show top 10
                score = sentence.get("score", 0)
                rating = sentence.get("rating", 0)
                md_content += f"**{i}. Score: {score:.2f} (Rating: {rating}/10)**\n\n"
                md_content += f"{sentence.get('text', '')}\n\n"
                
                # Show matched tags by priority
                matched_tags = sentence.get("matched_tags", {})
                if matched_tags.get("high"):
                    md_content += f"*High Priority Matches: {', '.join(matched_tags['high'])}*\n\n"
                if matched_tags.get("medium"):
                    md_content += f"*Medium Priority Matches: {', '.join(matched_tags['medium'])}*\n\n"
                if matched_tags.get("low"):
                    md_content += f"*Low Priority Matches: {', '.join(matched_tags['low'])}*\n\n"
                
                md_content += "---\n\n"
            
            if len(keyword_matches) > 10:
                md_content += f"*...and {len(keyword_matches) - 10} more matches.*\n\n"
        else:
            md_content += "*No matches found for the provided keywords.*\n\n"
    
    # Add all matching sentences section
    md_content += f"## All Matching Content ({total_matches} matches)\n\n"
    
    # Add matching sentences
    for i, sentence in enumerate(matched_sentences, 1):
        score = sentence.get("score", 0)
        rating = sentence.get("rating", 0)
        all_tags = sentence.get("all_tags", [])
        
        md_content += f"### {i}. Score: {score:.2f} (Rating: {rating}/10)\n\n"
        
        md_content += f"{sentence.get('text', '')}\n\n"
        
        # Show matched tags by priority
        matched_tags = sentence.get("matched_tags", {})
        if matched_tags.get("high"):
            md_content += f"**High Priority Matches:** {', '.join(matched_tags['high'])}\n\n"
        if matched_tags.get("medium"):
            md_content += f"**Medium Priority Matches:** {', '.join(matched_tags['medium'])}\n\n"
        if matched_tags.get("low"):
            md_content += f"**Low Priority Matches:** {', '.join(matched_tags['low'])}\n\n"
        
        # Show all tags
        md_content += f"**All Tags:** {', '.join(all_tags)}\n\n"
        
        # Show source file
        source = sentence.get("source", "Unknown")
        md_content += f"**Source:** {source}\n\n"
        
        md_content += "---\n\n"
    
    # Add cover letter if requested
    if include_cover_letter:
        md_content += "## Cover Letter Draft\n\n"
        
        if print_prompt_only:
            prompt = generate_cover_letter(job, matches, print_prompt_only=True, model=model)
            md_content += "### LLM Prompt\n\n"
            md_content += f"```\n{prompt}\n```\n\n"
        else:
            cover_letter = generate_cover_letter(job, matches, model=model)
            md_content += cover_letter + "\n\n"
    
    # Add footer
    md_content += "---\n"
    md_content += f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    return md_content

def save_markdown_report(job: Dict, matches: Dict, include_cover_letter: bool = False, print_prompt_only: bool = False, model: str = DEFAULT_LLM_MODEL, keywords: List[str] = None) -> str:
    """
    Save a markdown report for the job matches.
    
    Args:
        job: Job data
        matches: Dict with matched sentences
        include_cover_letter: Whether to include a cover letter draft
        print_prompt_only: If True, print the prompt instead of generating a cover letter
        model: The LLM model to use for generation
        keywords: Optional list of keywords to search for
        
    Returns:
        str: Path to the saved report
    """
    # Create reports directory if it doesn't exist
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Generate report content
    report_content = generate_markdown_report(job, matches, include_cover_letter, print_prompt_only, model, keywords)
    
    # Create a filename based on job details
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = job['job_title'].lower().replace(' ', '-')
    org_name = job['org_name'].lower().replace(' ', '-')
    model_name = model.replace(':', '-')  # Replace colons with hyphens for filename safety
    
    if include_cover_letter:
        filename = f"{timestamp}_{job_name}_{org_name}_{model_name}_with_cover_letter.md"
    else:
        filename = f"{timestamp}_{job_name}_{org_name}_{model_name}.md"
    
    filepath = os.path.join(REPORTS_DIR, filename)
    
    # Write the report to file
    with open(filepath, 'w') as f:
        f.write(report_content)
    
    return filepath

def list_available_jobs(jobs_data: Dict) -> None:
    """List all available jobs with their sequential IDs."""
    print("\nAVAILABLE JOBS:")
    print("-"*80)
    for job in jobs_data.get("jobs", []):
        print(f"Job #{job['id']}: {job['job_title']} at {job['org_name']}")
    print("-"*80)

def list_available_llm_models():
    """List all available Ollama models."""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print("Available Ollama models:")
        print(result.stdout)
    except Exception as e:
        print(f"Error listing models: {e}")

def run_multi_model_analysis(job_id: int, models: List[str], report: bool = False, 
                            cover_letter: bool = False, print_prompt: bool = False) -> None:
    """
    Run the cover letter matcher with multiple LLM models and generate separate reports.
    
    Args:
        job_id: Sequential ID of the job to analyze
        models: List of LLM models to use
        report: Whether to generate markdown reports
        cover_letter: Whether to include cover letter drafts
        print_prompt: Whether to print prompts instead of generating cover letters
    """
    import time
    
    # Load job data
    try:
        jobs_data = load_jobs(JOBS_FILE)
        # Update job IDs to be sequential
        jobs_data = update_job_ids(jobs_data)
    except Exception as e:
        print(f"Error loading job data: {e}")
        return
    
    # Find the job with the specified ID
    target_job = None
    for job in jobs_data.get("jobs", []):
        if job.get("id") == job_id:
            target_job = job
            break
    
    if not target_job:
        print(f"No job found with ID {job_id}")
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
    
    # Run analysis with each model
    report_paths = []
    timing_results = []
    
    for model in models:
        print(f"\n{'='*40}")
        print(f"RUNNING WITH MODEL: {model}")
        print(f"{'='*40}")
        
        start_time = time.time()
        
        if report:
            report_path = save_markdown_report(target_job, matches, cover_letter, print_prompt, model)
            report_paths.append((model, report_path))
            print(f"Report generated: {report_path}")
        elif print_prompt and cover_letter:
            prompt = generate_cover_letter(target_job, matches, print_prompt_only=True, model=model)
            print("\nLLM PROMPT FOR COVER LETTER GENERATION:")
            print("="*80)
            print(prompt)
            print("="*80)
        elif cover_letter:
            cover_letter_text = generate_cover_letter(target_job, matches, model=model)
            print(f"\nGENERATED COVER LETTER WITH {model}:")
            print("="*80)
            print(cover_letter_text)
            print("="*80)
        else:
            print(f"Model {model} - No output requested (use --report or --cover-letter)")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        timing_results.append((model, elapsed_time))
        print(f"Total processing time for {model}: {elapsed_time:.2f} seconds")
    
    # Print summary of reports
    if report and report_paths:
        print("\nSUMMARY OF GENERATED REPORTS:")
        print("="*80)
        for model, path in report_paths:
            print(f"Model: {model} - Report: {path}")
        print("="*80)
    
    # Print timing summary
    print("\nTIMING SUMMARY:")
    print("="*80)
    for model, elapsed_time in timing_results:
        print(f"Model: {model} - Total time: {elapsed_time:.2f} seconds")
    print("="*80)
    
    print(f"\nCompleted multi-model analysis for Job #{job_id}: {target_job['job_title']} at {target_job['org_name']}")

def get_top_sentences_by_category(matches: Dict, top_n: int = 3) -> Dict:
    """
    Get the top N sentences for each tag priority category.
    Each sentence will appear only once in the highest priority category it matches.
    
    Args:
        matches: Dict with matched sentences
        top_n: Number of top sentences to return for each category (default: 3)
        
    Returns:
        Dict with top sentences organized by priority category
    """
    matched_sentences = matches.get("matches", [])
    
    # Initialize result structure
    result = {
        "high": [],
        "medium": [],
        "low": []
    }
    
    # Track sentences we've already added
    used_sentences = set()
    
    # First pass: add to high priority
    for sentence in matched_sentences:
        sentence_text = sentence.get("text", "")
        matched_tags = sentence.get("matched_tags", {})
        
        if matched_tags.get("high") and len(result["high"]) < top_n and sentence_text not in used_sentences:
            result["high"].append(sentence)
            used_sentences.add(sentence_text)
    
    # Second pass: add to medium priority
    for sentence in matched_sentences:
        sentence_text = sentence.get("text", "")
        matched_tags = sentence.get("matched_tags", {})
        
        if matched_tags.get("medium") and len(result["medium"]) < top_n and sentence_text not in used_sentences:
            result["medium"].append(sentence)
            used_sentences.add(sentence_text)
    
    # Third pass: add to low priority
    for sentence in matched_sentences:
        sentence_text = sentence.get("text", "")
        matched_tags = sentence.get("matched_tags", {})
        
        if matched_tags.get("low") and len(result["low"]) < top_n and sentence_text not in used_sentences:
            result["low"].append(sentence)
            used_sentences.add(sentence_text)
    
    return result

def search_sentences_by_keywords(matches: Dict, keywords: List[str]) -> List[Dict]:
    """
    Search for sentences that match specific keywords.
    
    Args:
        matches: Dict with matched sentences
        keywords: List of keywords to search for
        
    Returns:
        List of matching sentences sorted by score
    """
    if not keywords:
        return []
    
    matched_sentences = matches.get("matches", [])
    keyword_matches = []
    
    # Convert keywords to lowercase for case-insensitive matching
    keywords_lower = [k.lower() for k in keywords]
    
    # Process each sentence
    for sentence in matched_sentences:
        text = sentence.get("text", "").lower()
        all_tags = [tag.lower() for tag in sentence.get("all_tags", [])]
        
        # Check if any keyword matches the sentence text or tags
        for keyword in keywords_lower:
            if keyword in text or any(keyword in tag for tag in all_tags):
                keyword_matches.append(sentence)
                break
    
    # Sort by score
    keyword_matches.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return keyword_matches

def main():
    """Main function to run the cover letter matcher."""
    parser = argparse.ArgumentParser(description="Match cover letter sentences to job requirements")
    parser.add_argument("--job-id", type=int, help="Sequential ID of the job to analyze")
    parser.add_argument("--list", action="store_true", help="List all available jobs")
    parser.add_argument("--report", action="store_true", help="Generate a markdown report")
    parser.add_argument("--cover-letter", action="store_true", help="Include a cover letter draft in the report")
    parser.add_argument("--print-prompt", action="store_true", help="Print the LLM prompt instead of generating a cover letter")
    
    # Add arguments for search functionality
    parser.add_argument("--keywords", nargs="+", help="Search for sentences matching these keywords")
    parser.add_argument("--top-n", type=int, default=3, help="Number of top sentences to show per category (default: 3)")
    
    # Add arguments for configurable scoring weights
    parser.add_argument("--high-weight", type=float, help=f"Weight for high priority matches (default: {SCORING_WEIGHTS['high_priority_match']})")
    parser.add_argument("--medium-weight", type=float, help=f"Weight for medium priority matches (default: {SCORING_WEIGHTS['medium_priority_match']})")
    parser.add_argument("--low-weight", type=float, help=f"Weight for low priority matches (default: {SCORING_WEIGHTS['low_priority_match']})")
    parser.add_argument("--multi-tag-bonus", type=float, help=f"Bonus for each additional tag match (default: {SCORING_WEIGHTS['multi_tag_bonus']})")
    
    # Add arguments for LLM model selection
    parser.add_argument("--llm", "--model", dest="llm_model", default=DEFAULT_LLM_MODEL,
                        help=f"LLM model to use (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--multi-llm", "--models", dest="llm_models", nargs="+",
                        help="Run analysis with multiple LLM models and generate separate reports for each")
    parser.add_argument("--list-models", action="store_true", 
                        help="List available Ollama models and exit")
    
    args = parser.parse_args()
    
    # List available LLM models if requested
    if args.list_models:
        list_available_llm_models()
        return
    
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
    
    # Check if multi-model analysis is requested
    if args.llm_models:
        run_multi_model_analysis(args.job_id, args.llm_models, args.report, args.cover_letter, args.print_prompt)
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
        report_path = save_markdown_report(target_job, matches, args.cover_letter, args.print_prompt, args.llm_model, args.keywords)
        print(f"\nReport generated: {report_path}")
    # If just print-prompt is requested (without report), print the prompt to terminal
    elif args.print_prompt and args.cover_letter:
        prompt = generate_cover_letter(target_job, matches, print_prompt_only=True, model=args.llm_model)
        print("\nLLM PROMPT FOR COVER LETTER GENERATION:")
        print("="*80)
        print(prompt)
        print("="*80)
    # If just cover-letter is requested (without report), generate and display the cover letter
    elif args.cover_letter and not args.report:
        cover_letter = generate_cover_letter(target_job, matches, model=args.llm_model)
        print("\nGENERATED COVER LETTER:")
        print("="*80)
        print(cover_letter)
        print("="*80)
    # If keywords are provided but no report, show keyword search results
    elif args.keywords and not args.report:
        keyword_matches = search_sentences_by_keywords(matches, args.keywords)
        print(f"\nKEYWORD SEARCH RESULTS FOR: {', '.join(args.keywords)}")
        print("="*80)
        if keyword_matches:
            for i, sentence in enumerate(keyword_matches[:10], 1):  # Show top 10
                score = sentence.get("score", 0)
                rating = sentence.get("rating", 0)
                print(f"{i}. Score: {score:.2f} (Rating: {rating}/10)")
                print(f"   {sentence.get('text', '')}")
                print()
            
            if len(keyword_matches) > 10:
                print(f"...and {len(keyword_matches) - 10} more matches. Generate a report to see all matches.")
        else:
            print("No matches found for the provided keywords.")
    else:
        # Display top sentences by category
        if not args.report:
            top_sentences = get_top_sentences_by_category(matches, args.top_n)
            
            print(f"\nTOP {args.top_n} SENTENCES BY CATEGORY:")
            print("="*80)
            
            # High priority sentences
            print("\nHIGH PRIORITY:")
            if top_sentences["high"]:
                for i, sentence in enumerate(top_sentences["high"], 1):
                    score = sentence.get("score", 0)
                    rating = sentence.get("rating", 0)
                    print(f"{i}. Score: {score:.2f} (Rating: {rating}/10)")
                    print(f"   {sentence.get('text', '')}")
                    
                    # Show matched tags
                    matched_tags = sentence.get("matched_tags", {}).get("high", [])
                    if matched_tags:
                        print(f"   Matched tags: {', '.join(matched_tags)}")
                    print()
            else:
                print("   No high priority matches found")
            
            # Medium priority sentences
            print("\nMEDIUM PRIORITY:")
            if top_sentences["medium"]:
                for i, sentence in enumerate(top_sentences["medium"], 1):
                    score = sentence.get("score", 0)
                    rating = sentence.get("rating", 0)
                    print(f"{i}. Score: {score:.2f} (Rating: {rating}/10)")
                    print(f"   {sentence.get('text', '')}")
                    
                    # Show matched tags
                    matched_tags = sentence.get("matched_tags", {}).get("medium", [])
                    if matched_tags:
                        print(f"   Matched tags: {', '.join(matched_tags)}")
                    print()
            else:
                print("   No medium priority matches found")
            
            # Low priority sentences
            print("\nLOW PRIORITY:")
            if top_sentences["low"]:
                for i, sentence in enumerate(top_sentences["low"], 1):
                    score = sentence.get("score", 0)
                    rating = sentence.get("rating", 0)
                    print(f"{i}. Score: {score:.2f} (Rating: {rating}/10)")
                    print(f"   {sentence.get('text', '')}")
                    
                    # Show matched tags
                    matched_tags = sentence.get("matched_tags", {}).get("low", [])
                    if matched_tags:
                        print(f"   Matched tags: {', '.join(matched_tags)}")
                    print()
            else:
                print("   No low priority matches found")
            
            print("\nFor all matches, use --report to generate a full report")
        
        # Also display regular matches
        display_matches(target_job, matches)
    
    print(f"\nCompleted analysis for Job #{args.job_id}: {target_job['job_title']} at {target_job['org_name']}")
    
    # Print helpful usage information
    if args.report and not args.cover_letter:
        print(f"To include a cover letter draft in the report, add the --cover-letter flag")
    if not args.print_prompt and (args.report or args.cover_letter):
        print(f"To print the LLM prompt instead of generating a cover letter, add the --print-prompt flag")
    if args.cover_letter:
        print(f"LLM model used: {args.llm_model}")
    if not args.keywords:
        print(f"To search for specific keywords, use --keywords <keyword1> <keyword2> ...")

if __name__ == "__main__":
    main()
