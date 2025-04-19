"""
Ollama Utilities Module - Provides functions for interacting with the Ollama API.

This module contains utility functions for generating text using Ollama language models,
with specific functions for cover letter generation and other text generation tasks.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Any, Union

import ollama
from src.config import DEFAULT_LLM_MODEL

# Define fallback models in order of preference
FALLBACK_MODELS = ["llama3:8b", "mistral:7b", "gemma:7b"]

def check_model_availability(model: str) -> bool:
    """
    Check if a specific model is available in Ollama.
    
    Args:
        model: Name of the model to check
        
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        models = ollama.list()
        available_models = [m.get('name') for m in models.get('models', [])]
        return model in available_models
    except Exception as e:
        print(f"Error checking model availability: {str(e)}")
        return False

def generate_cover_letter(
    job_info: Dict[str, Any],
    content_blocks: List[Dict[str, Any]],
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40
) -> str:
    """
    Generate a cover letter draft using an Ollama language model.
    
    Args:
        job_info: Dictionary containing job information including title, company, summary, and tags
        content_blocks: List of top-rated content blocks relevant to the job
        model: Ollama model name to use for generation (default: from config)
        temperature: Controls randomness of generation (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        
    Returns:
        str: Generated cover letter text
        
    Raises:
        Exception: If there is an error in generating the cover letter with all available models
    """
    start_time = time.time()
    
    # Create job overview section for prompt
    job_title = job_info.get("job_title", "Unknown Position")
    org_name = job_info.get("org_name", "Unknown Company")
    job_summary = job_info.get("summary", "")
    
    job_tags = job_info.get("tags", {})
    high_priority_tags = job_tags.get("high_priority", [])
    medium_priority_tags = job_tags.get("medium_priority", [])
    
    # Prepare prompt for LLM with improved instructions
    prompt = f"""
You are a professional cover letter writer. Create a cover letter for a job application based on the following information:

Job Title: {job_title}
Company: {org_name}
Job Summary: {job_summary}

High Priority Requirements: {', '.join(high_priority_tags) if high_priority_tags else 'None'}
Medium Priority Requirements: {', '.join(medium_priority_tags) if medium_priority_tags else 'None'}

Use the following content blocks as the basis for the cover letter. These are high-quality, pre-written paragraphs that match the job requirements. Incorporate these effectively with minimal modification, focusing on flow and organization rather than rewriting:

"""
    
    # Sort content blocks by rating and score
    sorted_blocks = sorted(content_blocks[:15], key=lambda x: (x.get("rating", 0), x.get("score", 0)), reverse=True)
    
    # Add top content blocks to the prompt
    for i, block in enumerate(sorted_blocks, 1):
        block_id = block.get("id", "Unknown")
        block_rating = block.get("rating", 0)
        block_content = block.get("content", "")
        block_tags = block.get("tags", [])
        
        tags_str = ', '.join(block_tags) if block_tags else 'general'
        prompt += f"{i}. Block {block_id} (Rating: {block_rating}/10, Tags: {tags_str})\n\"{block_content}\"\n\n"
    
    prompt += """
Guidelines:
1. Write in a professional, confident tone
2. Maintain a conversational style
3. Use the provided content blocks with minimal modification - prioritize ORIGINAL CONTENT
4. Make only minor adjustments for flow, order, and transitions between blocks
5. Keep the letter concise (300-400 words)
6. Include a standard cover letter format with placeholders for contact information
7. Focus on how the candidate's experience aligns with the high priority job requirements
8. End with a call to action

Create a complete cover letter that feels cohesive and tailored to this specific job while preserving the original content blocks as much as possible.
"""
    
    # Try the primary model first, then fallbacks if needed
    models_to_try = [model] + FALLBACK_MODELS
    last_error = None
    
    for current_model in models_to_try:
        try:
            # Check if model is available
            if not check_model_availability(current_model):
                print(f"Model {current_model} is not available, trying next model...")
                continue
                
            print(f"Generating cover letter using {current_model}...")
            
            response = ollama.generate(
                model=current_model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k
                }
            )
            
            cover_letter = response.get("response", "")
            
            # Calculate and print the time taken
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Cover letter generation time with {current_model}: {elapsed_time:.2f} seconds")
            
            return cover_letter
            
        except Exception as e:
            last_error = e
            print(f"Error generating cover letter with {current_model}: {str(e)}")
            print(f"Trying fallback model...")
    
    # If we get here, all models failed
    error_msg = f"All models failed to generate cover letter. Last error: {str(last_error)}"
    print(error_msg)
    raise Exception(error_msg)
