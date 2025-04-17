#!/usr/bin/env python3
"""
Job Analyzer - Core module for analyzing job postings.

This module handles scraping and analyzing job postings using spaCy for NLP
processing and local LLM for information extraction.
"""

from __future__ import annotations

import os
import re
import json
import uuid
import spacy
import yaml
import ollama
import requests
import traceback
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Set
import sys

from src.config import DEFAULT_LLM_MODEL, DATA_DIR

# Constants
DEFAULT_OUTPUT_FILE = os.path.join(DATA_DIR, "json/analyzed_jobs.json")

class JobAnalyzer:
    """Core class for analyzing job postings.
    
    This class handles fetching, extracting, and analyzing job postings using
    spaCy for tag analysis and a local LLM for basic information extraction.
    """
    
    def __init__(self, output_file: str = DEFAULT_OUTPUT_FILE, llm_model: str = DEFAULT_LLM_MODEL) -> None:
        """Initialize the JobAnalyzer.
        
        Args:
            output_file: Path to the output file for storing analyzed jobs.
            llm_model: Default LLM model to use for analysis.
        """
        self.output_file = output_file
        self.llm_model = llm_model
        self.categories = self._load_categories()
        
    def _load_categories(self, yaml_file: str = "config/categories.yaml") -> Dict[str, Any]:
        """Load categories from a YAML file.
        
        Attempts to find the YAML file in multiple potential locations.
        
        Args:
            yaml_file: Path to the YAML file containing categories.
            
        Returns:
            Categories data dictionary with category hierarchies.
            
        Raises:
            FileNotFoundError: If the categories file cannot be found.
            yaml.YAMLError: If the YAML file is malformed.
        """
        try:
            # Try different potential paths
            paths_to_try = [
                os.path.join(DATA_DIR, yaml_file),
                yaml_file,
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), yaml_file),
                os.path.join(os.getcwd(), yaml_file)
            ]
            
            for path in paths_to_try:
                if os.path.exists(path):
                    with open(path, "r") as file:
                        categories = yaml.safe_load(file)
                    print(f"Loaded categories from {path}")
                    return categories
            
            print(f"Warning: Categories file not found at any of these locations: {paths_to_try}")
            return {}
        except Exception as e:
            print(f"Error loading categories from YAML file: {e}")
            return {}
    
    def fetch_job_posting(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Fetch job posting content from a URL.
        
        Downloads the HTML content from the specified URL and extracts
        the main job description text.
        
        Args:
            url: URL of the job posting.
            
        Returns:
            A tuple containing (cleaned_text, html_content) or (None, None) if fetching fails.
            
        Raises:
            requests.RequestException: If there's an error fetching the URL.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            html_content = response.text
            return self._extract_main_content(html_content), html_content
        except Exception as e:
            print(f"Error fetching job posting: {e}")
            return None, None
    
    def _extract_main_content(self, html_content: str) -> str:
        """Extract the main content from an HTML page, focusing on job description.
        
        Uses BeautifulSoup to parse the HTML and extract the most relevant
        job description text, filtering out navigation, scripts, and headers.
        
        Args:
            html_content: HTML content of the job posting page.
            
        Returns:
            Cleaned text content with the job description.
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
    
    def analyze_job_posting(self, url: str, job_text: str) -> Optional[Dict[str, Any]]:
        """Analyze a job posting using spaCy for tag analysis and Ollama for basic info extraction.
        
        This method performs comprehensive job posting analysis:
        1. Extracts basic information (organization, title, summary) using LLM
        2. Identifies skills, technologies, and requirements in the posting
        3. Categorizes requirements into priority levels
        
        Args:
            url: URL of the job posting.
            job_text: Extracted job posting text.
            
        Returns:
            Analyzed job data dictionary or None if analysis fails.
            
        Raises:
            RuntimeError: If there are issues with the LLM connection.
            ValueError: If job text is empty or too short for meaningful analysis.
        """
        if not job_text:
            print("Error: No job text provided for analysis.")
            return None
            
        if len(job_text.strip()) < 100:
            print("Warning: Job text is very short, analysis may be incomplete.")
            
        try:
            # Get basic job info and summary (using LLM)
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
            print(f"Using LLM model: {self.llm_model}")
            try:
                response = ollama.generate(model=self.llm_model, prompt=job_info_prompt)
                job_info_completion = ""
                for chunk in response:
                    if isinstance(chunk, tuple) and chunk[0] == "response":
                        job_info_completion += chunk[1]
            except Exception as e:
                error_message = str(e)
                if "connection refused" in error_message.lower():
                    raise RuntimeError("Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download")
                elif "model not found" in error_message.lower():
                    raise RuntimeError(f"Model '{self.llm_model}' not found in Ollama. Please check available models with 'ollama list' or download this model with 'ollama pull {self.llm_model}'.")
                else:
                    raise RuntimeError(f"Error communicating with Ollama: {e}")
            
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
            
            # Validate extracted information
            if org_name == "Unknown Organization" and job_title == "Unknown Position":
                print("Warning: LLM failed to extract organization name and job title.")
                print("LLM output was:")
                print(job_info_completion)
                
                # Try to extract from URL as fallback
                if "lever.co" in url:
                    parts = url.split("/")
                    if len(parts) > 4:
                        org_name = parts[3].replace("-", " ").title()
                elif "greenhouse.io" in url:
                    parts = url.split("/")
                    if len(parts) > 4:
                        org_name = parts[4].replace("-", " ").title()
                
                print(f"Extracted organization from URL as fallback: {org_name}")
            
            # Use spaCy-based approach to generate prioritized tags
            print("Using spaCy for tag analysis")
            prioritized_tags = self._analyze_tags(job_text)
            
            # Validate tags
            if not any(prioritized_tags.values()):
                print("Warning: No tags were extracted from the job posting.")
                print("This may indicate an issue with the text extraction or analysis.")
            
            # Create job data structure
            job_data = {
                "id": str(uuid.uuid4()),
                "org_name": org_name,
                "job_title": job_title,
                "url": url,
                "date_scraped": datetime.now().isoformat(),
                "summary": summary,
                "tags": prioritized_tags,
                "raw_text": job_text,
                "llm_model": self.llm_model
            }
            
            return job_data
        
        except RuntimeError as e:
            # Runtime errors have already been formatted with helpful messages
            print(f"Error analyzing job posting: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error analyzing job posting: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None
    
    def _analyze_tags(self, job_text: str) -> Dict[str, List[str]]:
        """Analyze the job text and generate prioritized tags using spaCy.
        
        Uses NLP processing to extract tags from job text and categorize them
        into high, medium, and low priority based on frequency and importance.
        
        Args:
            job_text: Job posting text.
            
        Returns:
            Prioritized tags in high, medium, and low priority categories.
            
        Raises:
            ImportError: If the spaCy utilities cannot be imported.
        """
        try:
            # Import here to avoid circular imports
            from src.utils.spacy_utils import prioritize_tags_for_job
            
            # Use the spaCy utility function to prioritize tags
            return prioritize_tags_for_job(job_text, self.categories)
        except ImportError:
            # Try relative import if package import fails
            try:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from utils.spacy_utils import prioritize_tags_for_job
                return prioritize_tags_for_job(job_text, self.categories)
            except Exception as e:
                print(f"Error importing spacy_utils: {e}")
                
                # Fallback to basic analysis if spaCy utilities are unavailable
                return self._basic_tag_analysis(job_text)
    
    def _basic_tag_analysis(self, job_text: str) -> Dict[str, List[str]]:
        """Basic tag analysis without spaCy, used as fallback.
        
        Performs simple keyword matching to identify potential skills, technologies, 
        and soft skills when the spaCy-based analysis is unavailable.
        
        Args:
            job_text: Job posting text.
            
        Returns:
            Dictionary with categorized tags in high, medium, and low priority lists.
        """
        # Convert to lowercase for case-insensitive matching
        text_lower = job_text.lower()
        
        # Define key terms to look for
        skills = ["python", "javascript", "typescript", "react", "node", "java", 
                 "c++", "sql", "nosql", "agile", "scrum", "leadership", "management"]
        
        tech = ["aws", "azure", "gcp", "cloud", "docker", "kubernetes", "linux", 
               "git", "devops", "ci/cd", "rest", "api", "microservices"]
        
        soft_skills = ["communication", "teamwork", "problem solving", "creativity", 
                      "adaptability", "time management", "collaboration"]
        
        # Find matches
        skill_matches = [skill for skill in skills if skill in text_lower]
        tech_matches = [t for t in tech if t in text_lower]
        soft_skill_matches = [soft for soft in soft_skills if soft in text_lower]
        
        # Create prioritized tags
        high_priority = skill_matches[:5]  # Most important technical skills
        medium_priority = tech_matches[:5] + skill_matches[5:8]  # Tech and additional skills
        low_priority = soft_skill_matches + skill_matches[8:] + tech_matches[5:]  # Soft skills and remainder
        
        return {
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority
        }
    
    def save_job_data(self, job_data: Dict[str, Any]) -> bool:
        """Save analyzed job data to JSON file.
        
        This method either appends the new job data to the existing data file
        or creates a new file if one doesn't exist. It ensures the JSON structure
        is properly maintained and handles existing job data with the same URL.
        
        Args:
            job_data: Analyzed job data dictionary.
            
        Returns:
            True if the save was successful, False otherwise.
            
        Raises:
            OSError: If there are issues with file access or creation.
            json.JSONDecodeError: If the existing file contains invalid JSON.
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            
            # Load existing job data
            existing_data = {}
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, "r") as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: {self.output_file} contains invalid JSON. Creating a new file.")
            
            # Use current date as key for data structure
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Initialize structure if needed
            if "jobs" not in existing_data:
                existing_data["jobs"] = []
                
            if "metadata" not in existing_data:
                existing_data["metadata"] = {
                    "last_updated": datetime.now().isoformat(),
                    "count": 0
                }
            
            # Check if we already have this job URL to avoid duplicates
            url = job_data.get("url", "")
            existing_job_index = None
            
            for i, job in enumerate(existing_data["jobs"]):
                if job.get("url") == url:
                    existing_job_index = i
                    break
                    
            # Update or append job data
            if existing_job_index is not None:
                print(f"Updating existing job data for URL: {url}")
                existing_data["jobs"][existing_job_index] = job_data
            else:
                print(f"Adding new job data for URL: {url}")
                existing_data["jobs"].append(job_data)
                
            # Update metadata
            existing_data["metadata"]["last_updated"] = datetime.now().isoformat()
            existing_data["metadata"]["count"] = len(existing_data["jobs"])
            
            # Add sequential IDs
            for i, job in enumerate(existing_data["jobs"], 1):
                job["id"] = i
            
            # Save the data
            with open(self.output_file, "w") as f:
                json.dump(existing_data, f, indent=2)
                
            print(f"Job data saved to {self.output_file}")
            return True
            
        except Exception as e:
            print(f"Error saving job data: {e}")
            traceback.print_exc()
            return False
    
    def display_job_analysis(self, job_data: Dict[str, Any]) -> None:
        """Display a summary of the job analysis.
        
        Prints a formatted summary of the job analysis, including 
        organization, title, summary, and the prioritized tags.
        
        Args:
            job_data: Analyzed job data dictionary.
        """
        if not job_data:
            print("No job data to display.")
            return
            
        print("\n" + "="*80)
        print(f"JOB ANALYSIS: {job_data.get('job_title', 'Unknown Position')}")
        print("="*80)
        print(f"Organization: {job_data.get('org_name', 'Unknown')}")
        print(f"URL: {job_data.get('url', 'N/A')}")
        print(f"\nSummary: {job_data.get('summary', 'No summary available')}")
        print("\nTags:")
        
        # Display prioritized tags
        tags = job_data.get("tags", {})
        
        print("\nHigh Priority:")
        for tag in tags.get("high_priority", []):
            print(f"  - {tag}")
            
        print("\nMedium Priority:")
        for tag in tags.get("medium_priority", []):
            print(f"  - {tag}")
            
        print("\nLow Priority:")
        for tag in tags.get("low_priority", []):
            print(f"  - {tag}")
            
        print("\n" + "="*80)
        
    def analyze_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch and analyze a job posting from a URL.
        
        Convenience method that combines fetching and analyzing a job posting
        into a single operation. Handles all necessary error checking and reporting.
        
        Args:
            url: URL of the job posting to analyze.
            
        Returns:
            Analyzed job data dictionary or None if analysis fails.
            
        Raises:
            ValueError: If the URL is invalid or doesn't contain job content.
            RuntimeError: If there are issues with the LLM connection.
        """
        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            print("Error: Invalid URL. URL must start with http:// or https://")
            return None
            
        print(f"Fetching job posting from {url}")
        job_text, html_content = self.fetch_job_posting(url)
        
        if not job_text:
            print("Error: Could not extract text from job posting URL.")
            return None
        
        print(f"Successfully extracted {len(job_text)} characters of job text.")
        print("Analyzing job posting...")
        
        job_data = self.analyze_job_posting(url, job_text)
        
        if job_data:
            self.save_job_data(job_data)
            self.display_job_analysis(job_data)
            
        return job_data
