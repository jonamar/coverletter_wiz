#!/usr/bin/env python3
"""
Content Matcher - Core module for matching content blocks to job requirements.

This module finds the best content blocks from cover letters that match
the tags in a job posting, organized by priority level.
"""

from __future__ import annotations

import json
import spacy
import os
import re
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Set, Optional, Any, Union, DefaultDict
from collections import defaultdict

import ollama
import requests

from src.config import DATA_DIR, DEFAULT_LLM_MODEL

# Constants
DEFAULT_JOBS_FILE = os.path.join(DATA_DIR, "json/analyzed_jobs.json")
DEFAULT_CONTENT_FILE = os.path.join(DATA_DIR, "json/cover_letter_content.json")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
MIN_RATING_THRESHOLD = 6.0  # Minimum rating to consider a content block

# Configurable scoring weights
SCORING_WEIGHTS = {
    "high_priority_match": 0.5,    # Weight for matching a high priority tag
    "medium_priority_match": 0.3,  # Weight for matching a medium priority tag
    "low_priority_match": 0.2,     # Weight for matching a low priority tag
    "multi_tag_bonus": 0.1,        # Additional bonus for each tag after the first
}

# Ollama API settings
OLLAMA_API_URL = "http://localhost:11434/api/generate"

class ContentMatcher:
    """Core class for matching content blocks to job requirements.
    
    This class finds the best content blocks from cover letters that match
    the tags in a job posting, organized by priority level. It provides methods
    for content matching, cover letter generation, and report creation.
    
    Attributes:
        jobs_file: Path to the JSON file containing analyzed jobs.
        content_file: Path to the JSON file containing content blocks.
        llm_model: Default LLM model to use for cover letter generation.
        jobs_data: Loaded job data from the jobs_file.
        content_data: Loaded content data from the content_file.
        sequential_jobs: Job data with sequential IDs for easier reference.
        rated_content_blocks: Processed list of all rated content blocks.
    """
    
    def __init__(self, 
                jobs_file: str = DEFAULT_JOBS_FILE, 
                content_file: str = DEFAULT_CONTENT_FILE,
                llm_model: str = DEFAULT_LLM_MODEL) -> None:
        """Initializes the ContentMatcher with job and content data.
        
        Args:
            jobs_file: Path to the JSON file containing analyzed jobs.
            content_file: Path to the JSON file containing content blocks.
            llm_model: Default LLM model to use for cover letter generation.
            
        Raises:
            FileNotFoundError: If either the jobs_file or content_file cannot be found.
            json.JSONDecodeError: If either file contains invalid JSON.
        """
        self.jobs_file = jobs_file
        self.content_file = content_file
        self.llm_model = llm_model
        self.jobs_data = self._load_jobs()
        self.content_data = self._load_content()
        self.sequential_jobs = self._update_job_ids(self.jobs_data)
        self.rated_content_blocks = self._get_all_rated_content_blocks()
        
    def _load_jobs(self) -> Dict[str, Any]:
        """Loads job data from JSON file.
        
        Returns:
            Dictionary containing job data with 'jobs' key containing a list of job objects.
            
        Raises:
            FileNotFoundError: If the jobs file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        try:
            with open(self.jobs_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {self.jobs_file} not found.")
            return {"jobs": []}
    
    def _load_content(self) -> Dict[str, Any]:
        """Loads content data from JSON file.
        
        Returns:
            Dictionary mapping document IDs to their content data.
            
        Raises:
            FileNotFoundError: If the content file doesn't exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        try:
            with open(self.content_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {self.content_file} not found.")
            return {}
    
    def _update_job_ids(self, jobs_data: Dict[str, Any]) -> Dict[str, Any]:
        """Updates job IDs to be sequential numbers instead of UUIDs.
        
        Converts the original UUID-based job IDs to sequential numeric IDs
        for easier reference, while preserving the original IDs.
        
        Args:
            jobs_data: Original jobs data dictionary with 'jobs' list.
            
        Returns:
            Updated jobs data dictionary with sequential IDs and an ID mapping.
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
    
    def _get_all_rated_content_blocks(self) -> List[Dict[str, Any]]:
        """Extracts all content blocks with ratings from the processed cover letters.
        
        Filters content blocks to include only those with ratings above the minimum
        threshold (MIN_RATING_THRESHOLD) and sorts them by rating in descending order.
        
        Returns:
            List of content block dictionaries, each containing text, rating, tags, source,
            and other metadata for content matching.
        """
        all_blocks = []
        
        # Skip metadata keys
        for file_key, file_data in self.content_data.items():
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
            
            paragraphs = file_data.get("content", {}).get("paragraphs", [])
            
            for paragraph in paragraphs:
                blocks = paragraph.get("sentences", [])  # Original naming in JSON
                
                for block in blocks:
                    # Only include blocks with ratings above threshold
                    rating = block.get("rating", 0)
                    if rating >= MIN_RATING_THRESHOLD:
                        # Include fields that indicate if this is a content group
                        is_group = block.get("is_sentence_group", False)
                        component_content = block.get("component_sentences", [])
                        
                        all_blocks.append({
                            "text": block.get("text", ""),
                            "rating": rating,
                            "tags": block.get("tags", []),
                            "source": file_key,
                            "is_content_group": is_group,  # Renamed from is_sentence_group
                            "component_content": component_content  # Renamed from component_sentences
                        })
        
        # Sort by rating (highest first)
        all_blocks.sort(key=lambda x: x.get("rating", 0), reverse=True)
        return all_blocks
    
    def find_matching_content(self, job_id: int) -> Dict[str, Any]:
        """Finds content blocks that match job tags, organized by block rather than tag.
        
        This method matches content blocks to job tags from the specified job,
        calculating a match score based on tag priorities and returning a sorted
        list of the best matching content blocks.
        
        Args:
            job_id: Sequential ID of the job to analyze.
            
        Returns:
            Dictionary containing:
                - 'matches': List of matching content blocks with scores and matched tags
                - 'job': The job object data
                
        Raises:
            ValueError: If job with the specified ID is not found.
        """
        # Find the job by its sequential ID
        job = next((j for j in self.sequential_jobs["jobs"] if j["id"] == job_id), None)
        if not job:
            print(f"Error: Job with ID {job_id} not found.")
            return {"matches": [], "job": None}
        
        # Get tags by priority level
        high_priority_tags = job.get("tags", {}).get("high_priority", [])
        medium_priority_tags = job.get("tags", {}).get("medium_priority", [])
        low_priority_tags = job.get("tags", {}).get("low_priority", [])
        
        # Track content blocks that match tags
        content_matches: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)  # Maps content text to match info
        
        # Find all tag matches for each content block
        for block in self.rated_content_blocks:
            block_text = block.get("text", "")
            block_tags = set(block.get("tags", []))
            
            # Initialize match info
            if block_text not in content_matches:
                content_matches[block_text] = {
                    "text": block_text,
                    "rating": block.get("rating", 0),
                    "source": block.get("source", ""),
                    "matched_tags": {
                        "high": [],
                        "medium": [],
                        "low": []
                    },
                    "is_content_group": block.get("is_content_group", False),
                    "component_content": block.get("component_content", []),
                    "score": 0.0,
                    "match_count": 0
                }
            
            # Check for tag matches
            for tag in high_priority_tags:
                if tag.lower() in block_tags:
                    if tag not in content_matches[block_text]["matched_tags"]["high"]:
                        content_matches[block_text]["matched_tags"]["high"].append(tag)
                        content_matches[block_text]["score"] += SCORING_WEIGHTS["high_priority_match"]
                        content_matches[block_text]["match_count"] += 1
            
            for tag in medium_priority_tags:
                if tag.lower() in block_tags:
                    if tag not in content_matches[block_text]["matched_tags"]["medium"]:
                        content_matches[block_text]["matched_tags"]["medium"].append(tag)
                        content_matches[block_text]["score"] += SCORING_WEIGHTS["medium_priority_match"]
                        content_matches[block_text]["match_count"] += 1
            
            for tag in low_priority_tags:
                if tag.lower() in block_tags:
                    if tag not in content_matches[block_text]["matched_tags"]["low"]:
                        content_matches[block_text]["matched_tags"]["low"].append(tag)
                        content_matches[block_text]["score"] += SCORING_WEIGHTS["low_priority_match"]
                        content_matches[block_text]["match_count"] += 1
        
        # Add multi-tag bonus for blocks that match multiple tags
        for text, match_info in content_matches.items():
            match_count = match_info["match_count"]
            if match_count > 1:
                # Add bonus for each additional tag after the first
                multi_tag_bonus = (match_count - 1) * SCORING_WEIGHTS["multi_tag_bonus"]
                content_matches[text]["score"] += multi_tag_bonus
        
        # Filter out blocks with no matches
        content_matches = {text: info for text, info in content_matches.items() 
                          if info["match_count"] > 0}
        
        # Convert to list and sort by score (highest first)
        matches_list = list(content_matches.values())
        matches_list.sort(key=lambda x: (x["score"], x["rating"]), reverse=True)
        
        return {"matches": matches_list, "job": job}
    
    def query_ollama(self, prompt: str, model: Optional[str] = None) -> str:
        """Queries the Ollama API with a prompt.
        
        Args:
            prompt: The prompt text to send to the Ollama API.
            model: The model to use for generation. If None, uses the default model.
            
        Returns:
            Generated text response from the LLM.
            
        Raises:
            RuntimeError: If there are connection issues with Ollama.
        """
        if model is None:
            model = self.llm_model
            
        print(f"Querying Ollama with model: {model}")
        try:
            response = ollama.generate(model=model, prompt=prompt)
            completion = ""
            for chunk in response:
                if isinstance(chunk, tuple) and chunk[0] == "response":
                    completion += chunk[1]
            return completion
        except Exception as e:
            print(f"Error querying Ollama: {e}")
            return f"Error generating text: {str(e)}"
    
    def clean_cover_letter(self, text: str) -> str:
        """Cleans the cover letter text by removing 'think' sections and other artifacts.
        
        Performs text cleanup by removing various formatting artifacts that might
        be present in the generated text, such as thinking sections, comments,
        and excessive newlines.
        
        Args:
            text: The raw cover letter text to clean.
            
        Returns:
            Cleaned cover letter text.
        """
        # Remove thinking sections
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'\[thinking\].*?\[/thinking\]', '', text, flags=re.DOTALL)
        
        # Remove any markdown-style comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Clean up any multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def generate_cover_letter(self, job_id: int, print_prompt_only: bool = False) -> str:
        """Generates a cover letter draft using the top-matching content blocks.
        
        This method uses a local LLM through Ollama to create a cover letter draft
        based on the job details and best-matching content blocks from the user's
        content database.
        
        Args:
            job_id: Sequential ID of the job to analyze.
            print_prompt_only: If True, return the prompt instead of generating a cover letter.
            
        Returns:
            Generated cover letter text or the prompt if print_prompt_only is True.
            
        Raises:
            ValueError: If job with the specified ID is not found.
            RuntimeError: If there are issues with the Ollama connection or model.
        """
        # Get job info and matching content
        matches_data = self.find_matching_content(job_id)
        job = matches_data["job"]
        matches = matches_data["matches"]
        
        if not job:
            raise ValueError(f"Job with ID {job_id} not found")
            
        if not matches:
            print("Warning: No matching content blocks found. Cover letter may be generic.")
            
        # Build prompt with job info and matching content
        org_name = job.get("org_name", "the company")
        job_title = job.get("job_title", "the position")
        job_summary = job.get("summary", "")
        
        prompt = f"""You are a professional cover letter writer. Write a cover letter for a job application to {org_name} for the position of {job_title}.

Job Summary: {job_summary}

The following are relevant excerpts from the applicant's previous cover letters that match this job's requirements. 
Use these excerpts to craft a cohesive, professional cover letter that highlights the applicant's relevant experience:

"""
        
        # Add top matches to the prompt (limit to 8 to avoid making the prompt too long)
        top_matches = sorted(matches, key=lambda x: (x["score"], x["rating"]), reverse=True)[:8]
        
        for i, match in enumerate(top_matches, 1):
            prompt += f"{i}. \"{match['text']}\"\n\n"
            
        prompt += """
Guidelines:
1. Organize the content in a clear, professional format with proper paragraphs
2. Begin with a greeting and introduction
3. IMPORTANT: Preserve the original phrasing and wording from the provided excerpts as much as possible
4. Avoid duplicating content or repeating the same points multiple times
5. Close with a call to action and contact information placeholder
6. Keep the letter to 400-500 words maximum
7. Don't use bullet points or numbered lists
8. Do NOT mention specific companies by name except for the one being applied to
9. Maintain the applicant's authentic voice by keeping their original expressions intact

Write the full cover letter now:"""

        # Return just the prompt if requested
        if print_prompt_only:
            return prompt
            
        try:
            # Generate the cover letter
            print(f"Generating cover letter using {self.llm_model}...")
            
            # Try to call Ollama for generation
            try:
                response = ollama.generate(model=self.llm_model, prompt=prompt)
                cover_letter = ""
                for chunk in response:
                    if isinstance(chunk, tuple) and chunk[0] == "response":
                        cover_letter += chunk[1]
            except Exception as e:
                error_message = str(e)
                if "connection refused" in error_message.lower():
                    raise RuntimeError("Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download")
                elif "model not found" in error_message.lower():
                    raise RuntimeError(f"Model '{self.llm_model}' not found in Ollama. Please check available models with 'ollama list' or download this model with 'ollama pull {self.llm_model}'.")
                else:
                    raise RuntimeError(f"Error communicating with Ollama: {e}")
            
            # Validate the generated cover letter
            if not cover_letter or len(cover_letter.strip()) < 100:
                print("Warning: Generated cover letter is unusually short or empty.")
                print("LLM may have failed to generate proper content.")
                return f"Error: Failed to generate a proper cover letter. The output was too short: {cover_letter}"
                
            # Check if the response looks like an error message
            error_indicators = ["error", "exception", "cannot", "unable to", "failed to"]
            if any(indicator in cover_letter.lower() for indicator in error_indicators) and len(cover_letter) < 300:
                print("Warning: Generated text looks like an error message rather than a cover letter.")
                return f"Error: LLM returned an error instead of a cover letter: {cover_letter}"
                
            return cover_letter
            
        except RuntimeError as e:
            # Runtime errors have already been formatted with helpful messages
            return f"Error generating cover letter: {e}"
        except Exception as e:
            traceback_str = traceback.format_exc()
            print(f"Unexpected error generating cover letter: {e}")
            print(f"Error type: {type(e).__name__}")
            print(traceback_str)
            return f"Error generating cover letter: An unexpected error occurred: {e}"
    
    def _get_top_content_by_category(self, matches: List[Dict[str, Any]], top_n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """Gets the top N content blocks for each tag priority category.
        
        Each block will appear only once in the highest priority category it matches.
        
        Args:
            matches: List of matched content blocks.
            top_n: Number of top blocks to return for each category.
            
        Returns:
            Dictionary with top blocks organized by priority category (high, medium, low).
        """
        # Initialize result structure
        top_by_category = {
            "high": [],
            "medium": [],
            "low": []
        }
        
        # Track which blocks have been assigned to a category
        assigned_blocks: Set[str] = set()
        
        # First pass: assign blocks to high priority category
        for block in matches:
            if block["text"] in assigned_blocks:
                continue
                
            if block["matched_tags"]["high"]:
                top_by_category["high"].append(block)
                assigned_blocks.add(block["text"])
                
                if len(top_by_category["high"]) >= top_n:
                    break
        
        # Second pass: assign blocks to medium priority category
        for block in matches:
            if block["text"] in assigned_blocks:
                continue
                
            if block["matched_tags"]["medium"]:
                top_by_category["medium"].append(block)
                assigned_blocks.add(block["text"])
                
                if len(top_by_category["medium"]) >= top_n:
                    break
        
        # Third pass: assign blocks to low priority category
        for block in matches:
            if block["text"] in assigned_blocks:
                continue
                
            if block["matched_tags"]["low"]:
                top_by_category["low"].append(block)
                assigned_blocks.add(block["text"])
                
                if len(top_by_category["low"]) >= top_n:
                    break
        
        return top_by_category
    
    def _format_content_for_prompt(self, blocks: List[Dict[str, Any]]) -> str:
        """Formats content blocks for the LLM prompt.
        
        Args:
            blocks: List of content blocks to format.
            
        Returns:
            Formatted string containing all content blocks with their ratings and scores.
        """
        if not blocks:
            return "No matching content blocks found."
            
        formatted = []
        for i, block in enumerate(blocks, 1):
            formatted.append(f"{i}. Rating: {block['rating']:.1f}, Score: {block['score']:.2f}")
            formatted.append(f"   \"{block['text']}\"")
            formatted.append("")
            
        return "\n".join(formatted)
    
    def generate_markdown_report(self, job_id: int, include_cover_letter: bool = False,
                                print_prompt_only: bool = False) -> str:
        """Generates a markdown report for the job matches.
        
        Creates a comprehensive markdown report containing job details, requirements (tags),
        matching content blocks organized by match score, and optionally a cover letter draft.
        
        Args:
            job_id: Sequential ID of the job to analyze.
            include_cover_letter: Whether to include a cover letter draft in the report.
            print_prompt_only: Whether to print the LLM prompt instead of generating a cover letter.
            
        Returns:
            Markdown formatted string containing the complete report.
            
        Raises:
            ValueError: If job with the specified ID is not found.
        """
        # Find matching content
        matches_data = self.find_matching_content(job_id)
        job = matches_data["job"]
        matches = matches_data["matches"]
        
        if not job:
            return f"# Error: Job with ID {job_id} not found."
            
        # Start building the report
        report = [
            f"# Job Match Report: {job.get('job_title', 'Unknown Position')}",
            "",
            f"**Company:** {job.get('org_name', 'Unknown Organization')}",
            f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Job URL:** [{job.get('url', '#')}]({job.get('url', '#')})",
            "",
            f"## Job Summary",
            "",
            job.get('summary', 'No summary available'),
            "",
            "## Job Requirements (Tags)",
            ""
        ]
        
        # Add job tags by priority
        tags = job.get("tags", {})
        
        report.append("### High Priority")
        report.append("")
        for tag in tags.get("high_priority", []):
            report.append(f"- {tag}")
        if not tags.get("high_priority"):
            report.append("- None")
        report.append("")
        
        report.append("### Medium Priority")
        report.append("")
        for tag in tags.get("medium_priority", []):
            report.append(f"- {tag}")
        if not tags.get("medium_priority"):
            report.append("- None")
        report.append("")
        
        report.append("### Low Priority")
        report.append("")
        for tag in tags.get("low_priority", []):
            report.append(f"- {tag}")
        if not tags.get("low_priority"):
            report.append("- None")
        report.append("")
        
        # Add matching content blocks sorted by score
        report.append("## Matching Content Blocks")
        report.append("")
        report.append("Content blocks are sorted by match score and rating.")
        report.append("")
        
        if not matches:
            report.append("No matching content blocks found.")
        else:
            # Divide into high/medium/low match sections based on score percentiles
            if len(matches) >= 3:
                top_third = len(matches) // 3
                high_matches = matches[:top_third]
                medium_matches = matches[top_third:top_third*2]
                low_matches = matches[top_third*2:]
            else:
                high_matches = matches
                medium_matches = []
                low_matches = []
            
            # High matches
            report.append("### High Match Score")
            report.append("")
            if high_matches:
                for i, block in enumerate(high_matches, 1):
                    report.append(f"#### {i}. Score: {block['score']:.2f}, Rating: {block['rating']:.1f}")
                    report.append("")
                    report.append(f"> {block['text']}")
                    report.append("")
                    
                    # List matched tags
                    report.append("**Matched Tags:**")
                    report.append("")
                    if block['matched_tags']['high']:
                        report.append("*High Priority:* " + ", ".join(block['matched_tags']['high']))
                    if block['matched_tags']['medium']:
                        report.append("*Medium Priority:* " + ", ".join(block['matched_tags']['medium']))
                    if block['matched_tags']['low']:
                        report.append("*Low Priority:* " + ", ".join(block['matched_tags']['low']))
                    report.append("")
            else:
                report.append("No high match score content blocks found.")
                report.append("")
            
            # Medium matches
            report.append("### Medium Match Score")
            report.append("")
            if medium_matches:
                for i, block in enumerate(medium_matches, 1):
                    report.append(f"#### {i}. Score: {block['score']:.2f}, Rating: {block['rating']:.1f}")
                    report.append("")
                    report.append(f"> {block['text']}")
                    report.append("")
                    
                    # List matched tags
                    report.append("**Matched Tags:**")
                    report.append("")
                    if block['matched_tags']['high']:
                        report.append("*High Priority:* " + ", ".join(block['matched_tags']['high']))
                    if block['matched_tags']['medium']:
                        report.append("*Medium Priority:* " + ", ".join(block['matched_tags']['medium']))
                    if block['matched_tags']['low']:
                        report.append("*Low Priority:* " + ", ".join(block['matched_tags']['low']))
                    report.append("")
            else:
                report.append("No medium match score content blocks found.")
                report.append("")
            
            # Low matches
            report.append("### Low Match Score")
            report.append("")
            if low_matches:
                for i, block in enumerate(low_matches, 1):
                    report.append(f"#### {i}. Score: {block['score']:.2f}, Rating: {block['rating']:.1f}")
                    report.append("")
                    report.append(f"> {block['text']}")
                    report.append("")
                    
                    # List matched tags
                    report.append("**Matched Tags:**")
                    report.append("")
                    if block['matched_tags']['high']:
                        report.append("*High Priority:* " + ", ".join(block['matched_tags']['high']))
                    if block['matched_tags']['medium']:
                        report.append("*Medium Priority:* " + ", ".join(block['matched_tags']['medium']))
                    if block['matched_tags']['low']:
                        report.append("*Low Priority:* " + ", ".join(block['matched_tags']['low']))
                    report.append("")
            else:
                report.append("No low match score content blocks found.")
                report.append("")
        
        # Add cover letter draft if requested
        if include_cover_letter:
            report.append("## Cover Letter Draft")
            report.append("")
            
            if print_prompt_only:
                report.append("### LLM Prompt")
                report.append("")
                report.append("```")
                report.append(self.generate_cover_letter(job_id, print_prompt_only=True))
                report.append("```")
            else:
                cover_letter = self.generate_cover_letter(job_id)
                report.append(cover_letter)
        
        return "\n".join(report)
    
    def save_markdown_report(self, job_id: int, include_cover_letter: bool = False,
                           print_prompt_only: bool = False) -> None:
        """Generates and saves a markdown report for a specific job.
        
        Creates a comprehensive report for the specified job and saves it to the 
        reports directory in the external data repository.
        
        Args:
            job_id: Sequential ID of the job to analyze.
            include_cover_letter: Whether to include a cover letter draft in the report.
            print_prompt_only: Whether to print the LLM prompt instead of generating a cover letter.
            
        Raises:
            ValueError: If job with the specified ID is not found.
            OSError: If unable to create the reports directory or write the report file.
        """
        # Find matching content
        matches_data = self.find_matching_content(job_id)
        job = matches_data["job"]
        matches = matches_data["matches"]
        
        if not job:
            print(f"Error: Job with ID {job_id} not found.")
            return
        
        # Create report directory if it doesn't exist
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Generate report filename
        job_title = job.get("job_title", "Unknown").replace(" ", "_")
        org_name = job.get("org_name", "Unknown").replace(" ", "_")
        date_str = datetime.now().strftime("%Y%m%d")
        report_file = os.path.join(REPORTS_DIR, f"job_{job_id}_{org_name}_{job_title}_{date_str}.md")
        
        # Generate report content
        report_content = self.generate_markdown_report(
            job_id, 
            include_cover_letter=include_cover_letter,
            print_prompt_only=print_prompt_only
        )
        
        # Save the report
        with open(report_file, "w") as f:
            f.write(report_content)
            
        print(f"Report saved to {report_file}")
    
    def list_available_jobs(self) -> None:
        """Lists all available jobs with their sequential IDs.
        
        Prints a formatted table of all available jobs to the console, including
        their sequential IDs, organization names, and job titles.
        """
        print("\nAvailable Jobs:")
        print("ID | Organization | Job Title")
        print("-" * 50)
        
        for job in self.sequential_jobs.get("jobs", []):
            print(f"{job['id']:2d} | {job.get('org_name', 'Unknown'):<15} | {job.get('job_title', 'Unknown')}")
    
    def list_available_models(self) -> List[str]:
        """Lists all available Ollama models.
        
        Queries the Ollama API to get a list of all available models,
        printing them to the console and returning them as a list.
        
        Returns:
            List of available model names sorted alphabetically.
            
        Raises:
            RuntimeError: If unable to connect to Ollama.
            requests.RequestException: If there's an error in the HTTP request.
        """
        try:
            # Call Ollama API to list models
            response = requests.get("http://localhost:11434/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return sorted(models)  # Sort alphabetically
            else:
                print(f"Error fetching models: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []
    
    def get_job_by_id(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Gets a job by its sequential ID.
        
        Args:
            job_id: Sequential ID of the job to retrieve.
        
        Returns:
            Job data dictionary if found, None otherwise.
        """
        for job in self.sequential_jobs.get("jobs", []):
            if job.get("id") == job_id:
                return job
        return None
        
    def set_priority_weights(self, high: float = 0.5, medium: float = 0.3, low: float = 0.2) -> None:
        """Sets the scoring weights for different priority levels.
        
        Updates the global SCORING_WEIGHTS dictionary with the provided values.
        
        Args:
            high: Weight for high priority tag matches (default: 0.5).
            medium: Weight for medium priority tag matches (default: 0.3).
            low: Weight for low priority tag matches (default: 0.2).
        """
        global SCORING_WEIGHTS
        SCORING_WEIGHTS["high_priority_match"] = high
        SCORING_WEIGHTS["medium_priority_match"] = medium
        SCORING_WEIGHTS["low_priority_match"] = low
        print(f"Updated priority weights - High: {high}, Medium: {medium}, Low: {low}")
