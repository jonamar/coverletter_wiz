#!/usr/bin/env python3
"""
Match Content CLI - Command-line interface for matching content to job requirements.

This script provides a CLI for matching cover letter content blocks to job requirements
and generating reports and cover letter drafts.
"""

from __future__ import annotations

import argparse
import sys
import os
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Union, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.content_matcher import ContentMatcher
from src.config import DEFAULT_LLM_MODEL, DATA_DIR

def setup_argparse(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Sets up argument parser for the content matching CLI.
    
    Args:
        parser: Optional pre-existing ArgumentParser instance. If None, a new 
            parser will be created.
            
    Returns:
        argparse.ArgumentParser: Configured argument parser ready for parsing arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Match cover letter content to job requirements and generate reports.")
    
    # Main operation modes
    parser.add_argument("--job-id", type=int, help="Sequential ID of the job to analyze")
    parser.add_argument("--list", action="store_true", help="List all available jobs with their IDs")
    parser.add_argument("--report", action="store_true", help="Generate a markdown report in the reports directory")
    parser.add_argument("--cover-letter", action="store_true", 
                      help="Include a cover letter draft in the report or display it in the terminal")
    parser.add_argument("--print-prompt", action="store_true",
                      help="Print the LLM prompt instead of generating a cover letter")
    
    # Scoring weights
    parser.add_argument("--high-weight", type=float, default=0.5,
                      help="Weight for high priority tag matches (default: 0.5)")
    parser.add_argument("--medium-weight", type=float, default=0.3,
                      help="Weight for medium priority tag matches (default: 0.3)")
    parser.add_argument("--low-weight", type=float, default=0.2,
                      help="Weight for low priority tag matches (default: 0.2)")
    parser.add_argument("--multi-tag-bonus", type=float, default=0.1,
                      help="Bonus for each additional tag match (default: 0.1)")
    
    # File paths and LLM options
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL,
                      help="LLM model to use for cover letter generation (default: gemma3:12b)")
    parser.add_argument("--jobs-file", type=str, default=os.path.join(DATA_DIR, "json/analyzed_jobs.json"),
                      help="Path to the analyzed jobs JSON file")
    parser.add_argument("--content-file", type=str, default=os.path.join(DATA_DIR, "json/cover_letter_content.json"),
                      help="Path to the cover letter content JSON file")
    parser.add_argument("--list-models", action="store_true",
                      help="List available Ollama models and exit")
    parser.add_argument("--multi-model", action="store_true",
                      help="Run analysis with multiple LLM models")
    parser.add_argument("--models", type=str,
                      help="Comma-separated list of LLM models to use with --multi-model")
    
    return parser

def match_content(args: argparse.Namespace) -> int:
    """Matches content blocks to a specific job posting.
    
    This function implements the core content matching functionality by validating inputs,
    initializing the ContentMatcher, and performing requested operations like listing jobs,
    generating reports, or creating cover letters based on the provided arguments.
    
    Args:
        args: Command line arguments containing job selection, operation modes,
            scoring weights, and LLM model configuration.
            
    Returns:
        int: Exit code indicating success (0) or failure (non-zero).
        
    Raises:
        Exception: Various exceptions are caught internally and converted to error messages.
    """
    try:
        # Validate required arguments
        if args.list_models is False and args.list is False and args.job_id is None:
            print("Error: You must specify either --job-id, --list-jobs, or --list-models")
            return 1
            
        # Setup paths
        cover_letters_file = os.path.join(DATA_DIR, 'json/cover_letter_content.json')
        analyzed_jobs_file = os.path.join(DATA_DIR, 'json/analyzed_jobs.json')
        
        # Validate file existence
        if not os.path.exists(cover_letters_file):
            print(f"Error: Cover letters file not found at {cover_letters_file}")
            print("Please process your cover letters first using the 'process' command.")
            return 1
            
        if not os.path.exists(analyzed_jobs_file):
            print(f"Error: Analyzed jobs file not found at {analyzed_jobs_file}")
            print("Please analyze jobs first using the 'analyze' command.")
            return 1
        
        # Initialize the matcher
        try:
            # Initialize with specified model, or use default if not specified
            if args.model:
                matcher = ContentMatcher(cover_letters_file, analyzed_jobs_file, llm_model=args.model)
            else:
                matcher = ContentMatcher(cover_letters_file, analyzed_jobs_file)
        except ValueError as e:
            print(f"Error initializing ContentMatcher: {e}")
            return 1
        except Exception as e:
            print(f"Unexpected error initializing ContentMatcher: {e}")
            traceback.print_exc()
            return 1
        
        # List all analyzed jobs
        if args.list:
            try:
                matcher.list_available_jobs()
                return 0
            except Exception as e:
                print(f"Error listing available jobs: {e}")
                return 1
                
        # List available models
        if args.list_models:
            try:
                matcher.list_available_models()
                return 0
            except Exception as e:
                print(f"Error listing available models: {e}")
                if "connection refused" in str(e).lower():
                    print("\nTip: Make sure Ollama is installed and running. Download from https://ollama.com/download")
                return 1
        
        # Multi-model analysis
        if args.multi_model and args.models:
            models = args.models.split(',')
            print(f"Performing multi-model analysis using models: {', '.join(models)}")
            
            try:
                # Validate job ID
                job = matcher.get_job_by_id(args.job_id)
                if not job:
                    print(f"Error: Job with ID {args.job_id} not found.")
                    return 1
                    
                # Generate cover letter using each model
                for model in models:
                    model = model.strip()
                    print(f"\n===== Cover Letter using {model} =====")
                    
                    try:
                        # Create a new matcher with the current model
                        model_matcher = ContentMatcher(cover_letters_file, analyzed_jobs_file, llm_model=model)
                        
                        # Generate the cover letter
                        cover_letter = model_matcher.generate_cover_letter(args.job_id)
                        
                        # Check if the result contains an error message
                        if cover_letter.startswith("Error:"):
                            print(cover_letter)
                        else:
                            print(cover_letter)
                            
                    except Exception as e:
                        print(f"Error generating cover letter with model {model}: {e}")
                        continue
                        
                return 0
                
            except Exception as e:
                print(f"Error performing multi-model analysis: {e}")
                return 1
        
        # Standard matching for a single job
        job_id = args.job_id
        
        # Validate the job exists
        job = matcher.get_job_by_id(job_id)
        if not job:
            print(f"Error: Job with ID {job_id} not found.")
            return 1
            
        # Set the weights if provided
        if args.high_weight is not None or args.medium_weight is not None or args.low_weight is not None:
            high_weight = args.high_weight if args.high_weight is not None else 1.0
            medium_weight = args.medium_weight if args.medium_weight is not None else 0.6
            low_weight = args.low_weight if args.low_weight is not None else 0.3
            
            matcher.set_priority_weights(high=high_weight, medium=medium_weight, low=low_weight)
            print(f"Using custom weights - High: {high_weight}, Medium: {medium_weight}, Low: {low_weight}")
        
        print(f"Analyzing job #{job_id}: {job.get('job_title', 'Unknown')} at {job.get('org_name', 'Unknown')}")
        
        # Generate report if requested
        if args.report:
            print(f"Generating report for job {job_id}...")
            try:
                matcher.save_markdown_report(job_id, include_cover_letter=args.cover_letter, 
                                          print_prompt_only=args.print_prompt)
                return 0
            except Exception as e:
                print(f"Error generating report: {e}")
                return 1
        
        # Generate cover letter if requested
        if args.cover_letter and not args.report:
            print(f"Generating cover letter for job {job_id}...")
            
            if args.print_prompt:
                print("LLM Prompt:\n")
                print(matcher.generate_cover_letter(job_id, print_prompt_only=True))
            else:
                try:
                    cover_letter = matcher.generate_cover_letter(job_id)
                    # Check if the result contains an error message
                    if cover_letter.startswith("Error:"):
                        print(cover_letter)
                        return 1
                    else:
                        print(cover_letter)
                        return 0
                except Exception as e:
                    print(f"Error generating cover letter: {e}")
                    return 1
        
        # If no specific output is requested, show matches
        if not args.report and not args.cover_letter:
            try:
                matches_data = matcher.find_matching_content(job_id)
                matches = matches_data["matches"]
                
                if not matches:
                    print("No matching content blocks found for this job.")
                else:
                    print(f"Found {len(matches)} matching content blocks.")
                    print("Top 5 matches:")
                    
                    # Sort matches by score and rating
                    sorted_matches = sorted(matches, key=lambda x: (x["score"], x["rating"]), reverse=True)[:5]
                    
                    for i, match in enumerate(sorted_matches, 1):
                        print(f"\n{i}. Score: {match['score']:.2f}, Rating: {match['rating']}")
                        print(f"Tags: {', '.join(match['matched_tags'])}")
                        print(f"Text: {match['text'][:150]}...")
                        
                print("\nUse --report flag to generate a full markdown report.")
                print("Use --cover-letter flag to generate a cover letter draft.")
                
            except Exception as e:
                print(f"Error finding matching content: {e}")
                return 1
                
        return 0
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1

def main(args: Optional[argparse.Namespace] = None) -> None:
    """Runs the content matcher CLI with the provided arguments.
    
    This function handles the main execution flow of the content matcher CLI,
    processing command-line arguments and performing the appropriate operations
    based on the specified modes and configurations.
    
    Args:
        args: Optional pre-parsed command line arguments. If None, arguments 
            will be parsed from sys.argv.
            
    Returns:
        None
    """
    if args is None:
        parser = setup_argparse()
        args = parser.parse_args()
    
    # Update scoring weights based on arguments
    if args.high_weight != 0.5 or args.medium_weight != 0.3 or args.low_weight != 0.2 or args.multi_tag_bonus != 0.1:
        # This would update the ContentMatcher.SCORING_WEIGHTS dictionary
        print(f"Using custom scoring weights: high={args.high_weight}, medium={args.medium_weight}, "
              f"low={args.low_weight}, multi-tag-bonus={args.multi_tag_bonus}")
    
    # Initialize matcher with specified files and model
    matcher = ContentMatcher(
        jobs_file=args.jobs_file,
        content_file=args.content_file,
        llm_model=args.model
    )
    
    # Determine operation mode
    if args.list_models:
        print("Listing available Ollama models...")
        try:
            models = matcher.list_available_models()
            if models:
                print("\nAvailable Ollama models:")
                print("-" * 40)
                for model in models:
                    print(f"- {model}")
            else:
                print("No models found. Make sure Ollama is running.")
        except Exception as e:
            print(f"Error listing models: {e}")
            print("Make sure Ollama is installed and running.")
        return
    
    if args.list:
        print("Listing all available jobs...")
        matcher.list_available_jobs()
        return
    
    if args.job_id is not None:
        match_content(args)
        return
    else:
        # If no operation mode specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
