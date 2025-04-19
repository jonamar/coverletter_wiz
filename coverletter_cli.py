#!/usr/bin/env python3
"""
Coverletter Wizard CLI - Main entry point for the coverletter_wiz tools.

This script provides a unified command-line interface for all coverletter_wiz tools.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the repository root to the Python path
REPO_DIR = Path(__file__).parent.absolute()
sys.path.append(str(REPO_DIR))

# Import CLI modules
from src.cli import analyze_job
from src.cli import export_content
from src.cli import generate_report
from src.cli import process_content
from src.cli import rate_content_unified

def main():
    """Main entry point for the coverletter_wiz CLI."""
    parser = argparse.ArgumentParser(
        description="Coverletter Wizard - Tools for job application content management",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process content command
    process_parser = subparsers.add_parser(
        "process", 
        help="Process new cover letters from the text-archive directory"
    )
    process_content.setup_argparse(process_parser)
    
    # Rate content command
    rate_parser = subparsers.add_parser(
        "rate", 
        help="Rate and manage cover letter content blocks"
    )
    rate_content_unified.setup_argparse(rate_parser)
    
    # Analyze job command
    analyze_parser = subparsers.add_parser(
        "analyze", 
        help="Analyze a job posting and extract requirements"
    )
    analyze_job.setup_argparse(analyze_parser)
    
    # Export content command
    export_parser = subparsers.add_parser(
        "export", 
        help="Export high-rated content blocks to a markdown file"
    )
    export_content.setup_argparse(export_parser)
    
    # Generate report command
    report_parser = subparsers.add_parser(
        "report", 
        help="Generate a comprehensive job match report with content matching and cover letter"
    )
    generate_report.setup_argparse(report_parser)
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is specified, show help
    if not args.command:
        parser.print_help()
        return
    
    # Run the appropriate command
    if args.command == "process":
        process_content.main(args)
    elif args.command == "rate":
        rate_content_unified.main(args)
    elif args.command == "analyze":
        analyze_job.main(args)
    elif args.command == "export":
        export_content.main(args)
    elif args.command == "report":
        generate_report.main(args)

if __name__ == "__main__":
    main()
