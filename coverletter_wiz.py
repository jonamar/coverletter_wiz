#!/usr/bin/env python3
"""
Cover Letter Wizard - Main entry point wrapper script

This script provides a convenient entry point to run the CoverLetterWiz system.
"""
import sys
import os
from pathlib import Path

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change to the src directory
os.chdir(os.path.join(script_dir, 'src'))

# Import from the package
from src.cli.rate_content import main as rate_content_main
from src.cli.analyze_job import main as analyze_job_main
from src.cli.match_content import main as match_content_main

def setup_argparse():
    """Set up the main argument parser for the application."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cover Letter Wizard - A comprehensive system for managing cover letters and job applications."
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Content rating command
    rate_parser = subparsers.add_parser("rate", help="Rate and manage cover letter content")
    rate_parser.add_argument("--batch", action="store_true", help="Run batch rating mode for initial content evaluation")
    rate_parser.add_argument("--tournament", action="store_true", help="Run tournament mode to compare content blocks by category")
    rate_parser.add_argument("--refinement", action="store_true", help="Run category refinement mode to organize content by topic")
    rate_parser.add_argument("--legends", action="store_true", help="Run legends tournament for top-rated content (10.0+)")
    rate_parser.add_argument("--stats", action="store_true", help="Show detailed statistics about content blocks and categories")
    rate_parser.add_argument("--export", action="store_true", help="Export high-rated content blocks to markdown")
    rate_parser.add_argument("--min-rating", type=float, help="Minimum rating for exported content blocks")
    rate_parser.add_argument("--output", type=str, help="Output file for exported content blocks")
    
    # Job analysis command
    job_parser = subparsers.add_parser("job", help="Analyze job postings")
    job_parser.add_argument("--url", type=str, help="URL of the job posting to analyze")
    job_parser.add_argument("--list", action="store_true", help="List all analyzed jobs")
    job_parser.add_argument("--display", type=int, help="Display a job by its ID")
    job_parser.add_argument("--model", type=str, help="LLM model to use for analysis")
    
    # Content matching command
    match_parser = subparsers.add_parser("match", help="Match content to job requirements")
    match_parser.add_argument("--job-id", type=int, help="Sequential ID of the job to analyze")
    match_parser.add_argument("--list", action="store_true", help="List all available jobs")
    match_parser.add_argument("--report", action="store_true", help="Generate a markdown report")
    match_parser.add_argument("--cover-letter", action="store_true", help="Include a cover letter draft")
    match_parser.add_argument("--model", type=str, help="LLM model to use for cover letter generation")
    match_parser.add_argument("--list-models", action="store_true", help="List available Ollama models")
    match_parser.add_argument("--multi-model", action="store_true", help="Run analysis with multiple LLM models")
    match_parser.add_argument("--models", type=str, help="Comma-separated list of LLM models to use with --multi-model")
    
    return parser

def main():
    """Run the main entry point for the application."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return
    
    # Save current sys.argv
    old_argv = sys.argv.copy()
    
    # Route to appropriate subcommand
    if args.command == "rate":
        # Convert args to the format expected by rate_content_main
        sys.argv = [sys.argv[0]]
        if args.batch:
            sys.argv.append("--batch")
        if args.tournament:
            sys.argv.append("--tournament")
        if args.refinement:
            sys.argv.append("--refinement")
        if args.legends:
            sys.argv.append("--legends")
        if args.stats:
            sys.argv.append("--stats")
        if args.export:
            sys.argv.append("--export")
            
            # Pass additional parameters for export if provided
            if hasattr(args, 'min_rating') and args.min_rating is not None:
                sys.argv.extend(["--min-rating", str(args.min_rating)])
            if hasattr(args, 'output') and args.output is not None:
                sys.argv.extend(["--output", args.output])
        rate_content_main()
    
    elif args.command == "job":
        # Convert args to the format expected by analyze_job_main
        sys.argv = [sys.argv[0]]
        if args.url:
            sys.argv.extend(["--url", args.url])
        if args.list:
            sys.argv.append("--list")
        if args.display is not None:
            sys.argv.extend(["--display", str(args.display)])
        if args.model:
            sys.argv.extend(["--model", args.model])
        analyze_job_main()
    
    elif args.command == "match":
        # Convert args to the format expected by match_content_main
        sys.argv = [sys.argv[0]]
        if args.job_id is not None:
            sys.argv.extend(["--job-id", str(args.job_id)])
        if args.list:
            sys.argv.append("--list")
        if args.report:
            sys.argv.append("--report")
        if args.cover_letter:
            sys.argv.append("--cover-letter")
        if args.model:
            sys.argv.extend(["--model", args.model])
        if args.list_models:
            sys.argv.append("--list-models")
        if args.multi_model:
            sys.argv.append("--multi-model")
        if args.models:
            sys.argv.extend(["--models", args.models])
        match_content_main()
    
    # Restore sys.argv
    sys.argv = old_argv

if __name__ == "__main__":
    main()
