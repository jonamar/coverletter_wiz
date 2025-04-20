#!/usr/bin/env python3
"""
HTML Utils - Utilities for HTML content extraction.

This module provides utilities for extracting and cleaning content from HTML pages,
with specific focus on extracting job descriptions from various job posting websites.
"""

from __future__ import annotations

import re
from typing import Optional
from bs4 import BeautifulSoup

def extract_main_content(html_content: str) -> str:
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
