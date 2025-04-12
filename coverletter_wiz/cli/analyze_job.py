#!/usr/bin/env python3
"""
Analyze Job CLI - Command-line interface for analyzing job postings.

This script provides a CLI for analyzing job postings, extracting key information
using spaCy and a local LLM, and storing the results.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.job_analyzer import JobAnalyzer

def setup_argparse():
    """Set up argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze job postings and extract key information."
    )
    
    # Main arguments
    parser.add_argument("--url", type=str, help="URL of the job posting to analyze")
    parser.add_argument("--list", action="store_true", help="List all analyzed jobs")
    parser.add_argument("--display", type=int, help="Display a job by its ID")
    
    # Optional parameters
    parser.add_argument("--model", type=str, default="deepseek-r1:8b", 
                       help="LLM model to use for analysis")
    parser.add_argument("--file", type=str, default="data/json/analyzed_jobs.json", 
                       help="Path to jobs JSON file")
    parser.add_argument("--multi-model", action="store_true", 
                       help="Run analysis with multiple LLM models and compare results")
    parser.add_argument("--models", type=str, 
                       help="Comma-separated list of LLM models to use with --multi-model")
    
    return parser

def main():
    """Run the job analyzer CLI."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Initialize analyzer with specified file and model
    analyzer = JobAnalyzer(output_file=args.file, llm_model=args.model)
    
    # Determine operation mode
    if args.url:
        print(f"Analyzing job posting at: {args.url}")
        # Fetch and analyze the job posting
        job_text, html_content = analyzer.fetch_job_posting(args.url)
        
        if job_text:
            print(f"Successfully fetched job posting. Text length: {len(job_text)} characters")
            
            if args.multi_model:
                # If multi-model analysis is requested
                if args.models:
                    models = [model.strip() for model in args.models.split(",")]
                    print(f"Running multi-model analysis with models: {', '.join(models)}")
                    # This would run analysis with multiple models
                    print("Multi-model analysis functionality would be called here")
                else:
                    print("Error: --models must be specified with --multi-model")
            else:
                # Single model analysis
                job_data = analyzer.analyze_job_posting(args.url, job_text)
                
                if job_data:
                    # Save the job data
                    analyzer.save_job_data(job_data)
                    
                    # Display the analysis
                    analyzer.display_job_analysis(job_data)
                else:
                    print("Error: Failed to analyze job posting")
        else:
            print("Error: Failed to fetch job posting")
    
    elif args.list:
        print("Listing all analyzed jobs:")
        # This would list all jobs in the JSON file
        print("Job listing functionality would be called here")
    
    elif args.display is not None:
        print(f"Displaying job with ID: {args.display}")
        # This would display a specific job by ID
        print("Job display functionality would be called here")
    
    else:
        # If no mode specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
