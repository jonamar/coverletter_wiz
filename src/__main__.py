#!/usr/bin/env python3
"""
Cover Letter Wizard - Main entry point for the application.

This script provides a unified CLI for accessing all functionality of the Cover Letter Wizard system.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import CLI modules
from src.cli.rate_content import main as rate_content_main
from src.cli.analyze_job import main as analyze_job_main
from src.cli.match_content import main as match_content_main

def setup_argparse():
    """Set up the main argument parser for the application."""
    parser = argparse.ArgumentParser(
        description="Cover Letter Wizard - A comprehensive system for managing cover letters and job applications."
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Content rating command
    rate_parser = subparsers.add_parser("rate", help="Rate and manage cover letter content")
    rate_parser.add_argument("--batch", action="store_true", help="Run batch rating mode")
    rate_parser.add_argument("--tournament", action="store_true", help="Run tournament mode")
    rate_parser.add_argument("--refinement", action="store_true", help="Run category refinement mode")
    rate_parser.add_argument("--legends", action="store_true", help="Run legends tournament")
    rate_parser.add_argument("--stats", action="store_true", help="Show content block statistics")
    rate_parser.add_argument("--export", action="store_true", help="Export high-rated content blocks")
    
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
    
    return parser

def main():
    """Run the main entry point for the application."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return
    
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
        match_content_main()

if __name__ == "__main__":
    main()
