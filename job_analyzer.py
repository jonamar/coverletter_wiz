#!/usr/bin/env python3
"""
Job Analyzer

A tool that scrapes job postings, analyzes them with a local LLM (via Ollama),
and categorizes them based on a predefined set of tags.
"""

import os
import re
import json
import uuid
import spacy
import yaml
import ollama
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime

# Output file for storing analyzed jobs
ANALYZED_JOBS_FILE = "analyzed_jobs.json"

def load_categories(yaml_file):
    """Load categories from a YAML file."""
    try:
        with open(yaml_file, "r") as file:
            categories = yaml.safe_load(file)
        return categories
    except Exception as e:
        print(f"Error loading categories from YAML file: {e}")
        return None

def fetch_job_posting(url):
    """
    Fetch job posting content from a URL.
    
    Args:
        url (str): URL of the job posting
        
    Returns:
        tuple: (cleaned_text, html_content)
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        html_content = response.text
        return extract_main_content(html_content), html_content
    except Exception as e:
        print(f"Error fetching job posting: {e}")
        return None, None

def extract_main_content(html_content):
    """
    Extract the main content from an HTML page, focusing on job description.
    
    Args:
        html_content (str): HTML content
        
    Returns:
        str: Cleaned text content
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove unwanted elements
    for element in soup.select("script, style, nav, footer, header, aside"):
        element.decompose()
    
    # Look for common job posting containers
    potential_containers = [
        "job-description", "description", "jobDescriptionText", 
        "job-details", "jobDetails", "job-post", "job_description",
        "posting-body", "vacancy-details", "job-posting"
    ]
    
    # Try to find a specific job container first
    main_content = None
    for container in potential_containers:
        elements = soup.select(f"#{container}, .{container}, [data-test='{container}']")
        if elements:
            main_content = elements[0]
            break
    
    # If no container found, use main or article elements
    if not main_content:
        main_content = soup.find("main") or soup.find("article")
    
    # If still not found, use the body but try to be smart about it
    if not main_content:
        main_content = soup.body
    
    # Get the text and clean it up
    if main_content:
        # Get all paragraphs and list items
        text_blocks = main_content.find_all(["p", "li", "h1", "h2", "h3", "h4", "h5", "h6"])
        text_content = "\n\n".join([block.get_text(strip=True) for block in text_blocks if block.get_text(strip=True)])
        
        # Clean up the text
        cleaned_text = re.sub(r'\s+', ' ', text_content).strip()
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text
    
    # Fallback: just get all text
    return soup.get_text(separator="\n\n", strip=True)

def analyze_job_posting(url, job_text, categories):
    """
    Analyze a job posting using Ollama.
    
    Args:
        url (str): URL of the job posting
        job_text (str): Extracted job posting text
        categories (dict): Categories from YAML file
        
    Returns:
        dict: Analyzed job data
    """
    try:
        # Prepare the list of all possible tags
        all_tags = []
        for category, items in categories.items():
            all_tags.extend(items)
        tags_string = ", ".join(all_tags)
        
        # Get basic job info and summary
        job_info_prompt = f"""Given the following job posting text, extract:
1. The organization/company name
2. The job title
3. A one-sentence summary of the position (max 25 words)

Job posting text:
"{job_text[:2000]}"

Output your answer in this format exactly:
ORG: [Organization Name]
TITLE: [Job Title]
SUMMARY: [One sentence summary]
"""
        
        # Get basic job info
        response = ollama.generate(model="deepseek-r1:8b", prompt=job_info_prompt)
        job_info_completion = ""
        for chunk in response:
            if isinstance(chunk, tuple) and chunk[0] == "response":
                job_info_completion += chunk[1]
        
        # Extract basic job info
        org_name = "Unknown Organization"
        job_title = "Unknown Position" 
        summary = "No summary available"
        
        for line in job_info_completion.split('\n'):
            line = line.strip()
            if line.startswith('ORG:'):
                org_name = line.replace('ORG:', '').strip()
            elif line.startswith('TITLE:'):
                job_title = line.replace('TITLE:', '').strip()
            elif line.startswith('SUMMARY:'):
                summary = line.replace('SUMMARY:', '').strip()
        
        # Create tags prompt
        tags_prompt = f"""Let's analyze this job posting carefully:

Job posting: "{job_text[:3000]}"

Based on this job posting, I need you to identify the most relevant tags from this list: {tags_string}

Think about:
1. The key skills and competencies required
2. The industry domain
3. The expected outcomes and impacts
4. Team and organizational values

Select exactly 10 tags total, and group them into priority levels:
- HIGH PRIORITY: 3-4 tags that are absolutely essential to this job
- MEDIUM PRIORITY: 3-4 tags that are important but not critical
- LOW PRIORITY: 2-3 tags that are relevant but less emphasized

Output your tags in this exact format:
HIGH: tag1, tag2, tag3, tag4
MEDIUM: tag5, tag6, tag7
LOW: tag8, tag9, tag10
"""
        
        # Get tags with priorities
        response = ollama.generate(model="deepseek-r1:8b", prompt=tags_prompt)
        tags_completion = ""
        for chunk in response:
            if isinstance(chunk, tuple) and chunk[0] == "response":
                tags_completion += chunk[1]
        
        # Extract prioritized tags
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for line in tags_completion.split('\n'):
            line = line.strip()
            if line.startswith('HIGH:'):
                tags_part = line.replace('HIGH:', '').strip()
                high_priority = [tag.strip() for tag in tags_part.split(',')]
            elif line.startswith('MEDIUM:'):
                tags_part = line.replace('MEDIUM:', '').strip()
                medium_priority = [tag.strip() for tag in tags_part.split(',')]
            elif line.startswith('LOW:'):
                tags_part = line.replace('LOW:', '').strip()
                low_priority = [tag.strip() for tag in tags_part.split(',')]
        
        # Filter tags to ensure they're in our original list
        valid_high = [tag for tag in high_priority if tag in all_tags]
        valid_medium = [tag for tag in medium_priority if tag in all_tags]
        valid_low = [tag for tag in low_priority if tag in all_tags]
        
        # Create job data structure
        job_data = {
            "id": str(uuid.uuid4()),
            "org_name": org_name,
            "job_title": job_title,
            "url": url,
            "date_scraped": datetime.now().isoformat(),
            "summary": summary,
            "tags": {
                "high_priority": valid_high,
                "medium_priority": valid_medium,
                "low_priority": valid_low
            },
            "raw_text": job_text
        }
        
        return job_data
    
    except Exception as e:
        print(f"Error analyzing job posting: {e}")
        return None

def save_job_data(job_data, output_file=ANALYZED_JOBS_FILE):
    """
    Save analyzed job data to JSON file.
    
    Args:
        job_data (dict): Analyzed job data
        output_file (str): Path to output file
    """
    try:
        # Initialize or load existing data
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"jobs": []}
        
        # Check if this job URL is already in the data
        job_urls = [job["url"] for job in data["jobs"]]
        if job_data["url"] in job_urls:
            # Update existing job entry
            for i, job in enumerate(data["jobs"]):
                if job["url"] == job_data["url"]:
                    data["jobs"][i] = job_data
                    print(f"Updated existing job: {job_data['job_title']} at {job_data['org_name']}")
                    break
        else:
            # Add new job entry
            data["jobs"].append(job_data)
            print(f"Added new job: {job_data['job_title']} at {job_data['org_name']}")
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Job data saved to {output_file}")
        return True
    
    except Exception as e:
        print(f"Error saving job data: {e}")
        return False

def analyze_job_from_url(url):
    """
    Main function to analyze a job posting from a URL.
    
    Args:
        url (str): URL of the job posting
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Load categories
    categories = load_categories("categories.yaml")
    if not categories:
        print("Failed to load categories. Cannot analyze job posting.")
        return False
    
    # Fetch job posting
    job_text, _ = fetch_job_posting(url)
    if not job_text:
        print("Failed to fetch job posting content.")
        return False
    
    # Analyze job posting
    job_data = analyze_job_posting(url, job_text, categories)
    if not job_data:
        print("Failed to analyze job posting.")
        return False
    
    # Save job data
    success = save_job_data(job_data)
    
    return success

def display_job_analysis(job_data):
    """
    Display a summary of the job analysis.
    
    Args:
        job_data (dict): Analyzed job data
    """
    print("\n" + "="*80)
    print(f"JOB ANALYSIS: {job_data['job_title']} at {job_data['org_name']}")
    print("="*80)
    
    print(f"\nSUMMARY: {job_data['summary']}")
    print(f"\nURL: {job_data['url']}")
    print(f"DATE SCRAPED: {job_data['date_scraped']}")
    
    print("\nTAGS:")
    print("  HIGH PRIORITY:")
    for tag in job_data['tags']['high_priority']:
        print(f"    - {tag}")
    
    print("  MEDIUM PRIORITY:")
    for tag in job_data['tags']['medium_priority']:
        print(f"    - {tag}")
    
    print("  LOW PRIORITY:")
    for tag in job_data['tags']['low_priority']:
        print(f"    - {tag}")
    
    print("\nSaved to:", ANALYZED_JOBS_FILE)
    print("="*80 + "\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python job_analyzer.py <job_posting_url>")
        sys.exit(1)
    
    url = sys.argv[1]
    print(f"Analyzing job posting at: {url}")
    
    if analyze_job_from_url(url):
        # Load the saved data to display
        with open(ANALYZED_JOBS_FILE, 'r') as f:
            data = json.load(f)
        
        # Find the job we just analyzed
        for job in data["jobs"]:
            if job["url"] == url:
                display_job_analysis(job)
                break
        
        print("Job analysis complete!")
    else:
        print("Job analysis failed.")
