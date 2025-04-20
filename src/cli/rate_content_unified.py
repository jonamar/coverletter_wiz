#!/usr/bin/env python3
"""
Rate Content CLI - Command-line interface for rating cover letter content.

This module provides a CLI for rating, organizing, and refining cover letter content blocks.
It implements several rating modes:

1. Batch Rating: Initial rating of content blocks on a 1-10 scale
2. Tournament Mode: Compare content blocks within categories to refine ratings
3. Legends Tournament: Special tournament for top-rated content blocks (10.0+)
4. Category Refinement: Organize and improve content by topic/category
5. Statistics: Show detailed statistics about content blocks and categories
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.content_processor import ContentProcessor
from src.config import DATA_DIR

def setup_argparse(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Sets up argument parser for the content rating CLI.
    
    Configures an ArgumentParser with various operating modes and parameters
    for the content rating system, including batch rating, tournament mode,
    category refinement, and statistics.
    
    Args:
        parser: Optional pre-existing ArgumentParser instance. If None, a new 
            parser will be created.
            
    Returns:
        argparse.ArgumentParser: Configured argument parser ready for parsing arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Rate and manage cover letter content blocks."
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
    
    return parser

def main(args: Optional[argparse.Namespace] = None) -> None:
    """Runs the content rating CLI with the provided arguments.
    
    This function handles the main execution flow of the content rating CLI,
    processing command-line arguments and running the selected operation mode
    (batch rating, tournament, category refinement, legends tournament, or statistics).
    
    Args:
        args: Optional pre-parsed command line arguments. If None, arguments 
            will be parsed from sys.argv.
            
    Returns:
        None
        
    Raises:
        FileNotFoundError: If the content JSON file does not exist.
        ValueError: If there are errors in the content data structure.
    """
    if args is None:
        parser = setup_argparse()
        args = parser.parse_args()
    
    # Initialize processor with default file
    processor = ContentProcessor()
    
    # Show statistics if requested
    if args.stats:
        print("\nContent Block Statistics:")
        print(f"Total blocks: {processor.total_blocks}")
        print(f"Rated blocks: {processor.rated_blocks} ({processor.rated_blocks/processor.total_blocks*100:.1f}%)")
        print(f"High-rated blocks: {processor.high_rated_blocks} ({processor.high_rated_blocks/processor.total_blocks*100:.1f}%)")
        
        # Get categories from content blocks
        categories = processor._get_categories_from_blocks()
        category_stats = categories.get("stats", {})
        
        # Show category statistics
        print("\nCategory Statistics:")
        print(f"{'Category':<25} {'Count':<7} {'Avg Rating':<12} {'Status'}")
        print("-" * 70)
        
        # Sort categories by completion status and then by name
        sorted_stats = sorted(
            category_stats.items(),
            key=lambda x: (
                -x[1].get("completion_percentage", 0),  # Sort by completion (descending)
                x[0]  # Then by category name
            )
        )
        
        for category, stats in sorted_stats:
            category_display = category[:22] + "..." if len(category) > 25 else category
            avg_rating = stats.get("average_rating", 0)
            completion = stats.get("completion_percentage", 0) * 100
            
            status = "Incomplete"
            if completion >= 70:
                status = "Complete"
            if avg_rating >= 8.0:
                status = "Refined"
                
            print(f"{category_display:<25} {stats.get('count', 0):<7} {avg_rating:.1f}/10      {status}")
        
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
    else:
        # If no mode specified, show help
        parser = setup_argparse()
        parser.print_help()

if __name__ == "__main__":
    main()
