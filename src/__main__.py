#!/usr/bin/env python3
"""
Cover Letter Wizard - Main entry point for the application.

This script provides a unified CLI for accessing all functionality of the Cover Letter Wizard system.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Any, Optional, NoReturn, Sequence, Union, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import CLI modules
from src.cli.rate_content import main as rate_content_main
from src.cli.analyze_job import main as analyze_job_main
from src.cli.process_text import main as process_text_main
from src.cli.export_content import main as export_content_main
from src.cli.generate_report import main as generate_report_main

def setup_argparse() -> argparse.ArgumentParser:
    """Set up the main argument parser for the application.
    
    Creates and configures the argument parser with all subcommands
    and their respective arguments for the Cover Letter Wizard CLI.
    
    Returns:
        Configured ArgumentParser object with all subparsers and arguments.
    """
    parser = argparse.ArgumentParser(
        description="Coverletter Wizard - Tools for job application content management"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process text command
    process_parser = subparsers.add_parser("process", help="Process new cover letters from the text-archive directory")
    
    # Content rating command
    rate_parser = subparsers.add_parser("rate", help="Rate and manage cover letter content blocks")
    rate_parser.add_argument("--batch", action="store_true", help="Run batch rating mode")
    rate_parser.add_argument("--tournament", action="store_true", help="Run tournament mode")
    rate_parser.add_argument("--refinement", action="store_true", help="Run category refinement mode")
    rate_parser.add_argument("--legends", action="store_true", help="Run legends tournament")
    rate_parser.add_argument("--stats", action="store_true", help="Show content block statistics")
    
    # Job analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a job posting and extract requirements")
    analyze_parser.add_argument("--url", type=str, help="URL of the job posting to analyze")
    analyze_parser.add_argument("--list", action="store_true", help="List all analyzed jobs")
    analyze_parser.add_argument("--display", type=int, help="Display a job by its ID")
    analyze_parser.add_argument("--model", type=str, help="LLM model to use for analysis")
    
    # Export content command
    export_parser = subparsers.add_parser("export", help="Export high-rated content blocks to a markdown file")
    
    # Report generation command
    report_parser = subparsers.add_parser("report", help="Generate a comprehensive job match report with content matching and cover letter")
    report_parser.add_argument("--job-id", type=str, help="ID of the job to analyze")
    report_parser.add_argument("--job-url", type=str, help="URL of the job to analyze (alternative to --job-id)")
    report_parser.add_argument("--no-cover-letter", dest="include_cover_letter", action="store_false", default=True, help="Skip cover letter generation")
    report_parser.add_argument("--llm-model", type=str, help="LLM model to use for cover letter generation")
    report_parser.add_argument("--tags", "--keywords", type=str, nargs="+", help="Additional keywords/tags to prioritize in matching")
    
    return parser

def main() -> None:
    """Run the main entry point for the application.
    
    Parses command-line arguments and routes the execution to the appropriate
    subcommand handler based on the user's input.
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return
    
    # Route to appropriate subcommand
    if args.command == "process":
        process_text_main()
    
    elif args.command == "rate":
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
        rate_content_main()
    
    elif args.command == "analyze":
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
    
    elif args.command == "export":
        export_content_main()
    
    elif args.command == "report":
        # Convert args to the format expected by generate_report_main
        sys.argv = [sys.argv[0]]
        if args.job_id:
            sys.argv.extend(["--job-id", args.job_id])
        if args.job_url:
            sys.argv.extend(["--job-url", args.job_url])
        if not args.include_cover_letter:
            sys.argv.append("--no-cover-letter")
        if args.llm_model:
            sys.argv.extend(["--llm-model", args.llm_model])
        if args.tags:
            sys.argv.extend(["--tags"] + args.tags)
        generate_report_main()

if __name__ == "__main__":
    main()
