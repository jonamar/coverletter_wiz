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

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any, Union, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.content_processor import ContentProcessor
from src.config import DATA_DIR

def setup_argparse() -> argparse.ArgumentParser:
    """Sets up the argument parser for content rating.
    
    Configures an ArgumentParser with various operating modes and parameters
    for the content rating system, including batch rating, tournament mode,
    category refinement, legends tournament, statistics, and content export.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser ready for parsing arguments.
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
    parser.add_argument("--file", type=str, default=os.path.join(DATA_DIR, "json/cover_letter_content.json"), 
                       help="Path to content JSON file (default: in external data directory)")
    parser.add_argument("--output", type=str, 
                       help="Path to output file for exports (default: auto-generated)")
    
    return parser

def main() -> None:
    """Runs the content rating CLI with command line arguments.
    
    This function handles the main execution flow of the content rating CLI,
    processing command-line arguments and running the selected operation mode
    (batch rating, tournament, category refinement, legends tournament, 
    statistics, or content export).
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If the content JSON file does not exist.
        ValueError: If there are errors in the content data structure.
        OSError: If output directory creation fails during export.
    """
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
