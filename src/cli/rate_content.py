#!/usr/bin/env python3
"""
Cover Letter Wizard - Content Rating CLI

This module provides a command-line interface for rating, organizing, and refining 
cover letter content blocks. It implements several rating modes:

1. Batch Rating: Initial rating of content blocks on a 1-10 scale
2. Tournament Mode: Compare content blocks within categories to refine ratings
3. Legends Tournament: Special tournament for top-rated content blocks (10.0+)
4. Category Refinement: Organize and improve content by topic/category
5. Export: Export high-rated content blocks to markdown format

The rating system helps identify your strongest content for use in cover letters.
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.content_processor import ContentProcessor

def setup_argparse():
    """
    Set up the argument parser for content rating.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Cover Letter Wizard - Content Rating CLI"
    )
    
    # Operation modes (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batch", action="store_true", 
                      help="Run batch rating mode to give initial ratings to content blocks")
    group.add_argument("--tournament", action="store_true", 
                      help="Run tournament mode to compare content blocks by category")
    group.add_argument("--refinement", action="store_true", 
                      help="Run category refinement mode to organize and improve content by topic")
    group.add_argument("--legends", action="store_true", 
                      help="Run legends tournament for top-rated content blocks (rating ≥ 10.0)")
    group.add_argument("--stats", action="store_true", 
                      help="Show detailed statistics about your content blocks and categories")
    group.add_argument("--export", action="store_true", 
                      help="Export high-rated content blocks to a markdown file")
    
    # Optional parameters
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Number of blocks to rate in each batch (default: 10)")
    parser.add_argument("--min-rating", type=float, default=8.0, 
                       help="Minimum rating for content blocks to export (default: 8.0)")
    parser.add_argument("--category", type=str, 
                       help="Specific category to refine in refinement mode")
    parser.add_argument("--file", type=str, default="data/json/cover_letter_content.json", 
                       help="Path to content JSON file (default: data/json/cover_letter_content.json)")
    parser.add_argument("--output", type=str, 
                       help="Path to output file for exports (default: auto-generated)")
    
    return parser

def main():
    """Run the content rating CLI."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Initialize processor with specified file
    processor = ContentProcessor(json_file=args.file)
    
    # Show statistics if requested
    if args.stats:
        print("\nContent Block Statistics:")
        print(f"Total blocks: {processor.total_blocks}")
        print(f"Rated blocks: {processor.rated_blocks} ({processor.rated_blocks/processor.total_blocks*100:.1f}%)")
        print(f"High-rated blocks: {processor.high_rated_blocks} ({processor.high_rated_blocks/processor.total_blocks*100:.1f}%)")
        # Add more statistics as needed
        return
    
    # Determine operation mode
    if args.batch:
        print("Running batch rating mode...")
        processor._run_batch_rating()
    elif args.tournament:
        print("Running tournament mode...")
        processor._run_tournament_mode()
    elif args.refinement:
        print("Running category refinement mode...")
        processor._run_category_refinement()
    elif args.legends:
        print("Running legends tournament mode...")
        processor._run_legends_tournament()
    elif args.export:
        print("Exporting high-rated content blocks...")
        output_file = args.output
        if not output_file:
            # Create a default filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"data/exports/high_rated_content_{timestamp}.md"
            
            # Ensure exports directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
        processor.export_high_rated_content(min_rating=args.min_rating, output_file=output_file)
    else:
        # If no mode specified, show help
        parser.print_help()

if __name__ == "__main__":
    main()
