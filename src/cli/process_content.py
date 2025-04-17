#!/usr/bin/env python3
"""
Process Content CLI - Command-line interface for processing cover letter text files.

This script provides a CLI for processing cover letter text files from the text-archive
directory, extracting content blocks, and generating tags using spaCy.
"""

from __future__ import annotations

import argparse
import sys
import os
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.text_processor import TextProcessor
from src.config import DATA_DIR

def setup_argparse(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Sets up argument parser for the content processing CLI.
    
    Args:
        parser: Optional pre-existing ArgumentParser instance. If None, a new 
            parser will be created.
            
    Returns:
        argparse.ArgumentParser: Configured argument parser ready for parsing arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Process cover letter text files and extract content blocks."
        )
    
    # Main arguments
    parser.add_argument("--force", action="store_true", 
                       help="Force reprocessing of all files even if unchanged")
    
    # Optional parameters
    parser.add_argument("--archive-dir", type=str, 
                       default=os.path.join(DATA_DIR, "text-archive"),
                       help="Directory containing text files to process")
    parser.add_argument("--output-file", type=str, 
                       default=os.path.join(DATA_DIR, "json/processed_cover_letters.json"),
                       help="Output JSON file for processed content")
    parser.add_argument("--model", type=str, default="en_core_web_lg",
                       help="spaCy model to use for NLP processing")
    
    return parser

def main(args: Optional[argparse.Namespace] = None) -> int:
    """Runs the content processing CLI with the provided arguments.
    
    This function handles the main execution flow of the content processing CLI,
    processing command-line arguments, initializing the TextProcessor, and
    processing text files to extract content blocks with NLP analysis.
    
    Args:
        args: Optional pre-parsed command line arguments. If None, arguments 
            will be parsed from sys.argv.
            
    Returns:
        int: Exit code indicating success (0), failure (1), or
            operation cancelled (130).
            
    Raises:
        FileNotFoundError: If the archive directory does not exist.
        Exception: Various exceptions are caught internally and converted to error messages.
    """
    if args is None:
        parser = setup_argparse()
        args = parser.parse_args()
    
    try:
        # Set default paths using external data directory
        archive_dir = args.archive_dir
        output_file = args.output_file
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize text processor with en_core_web_lg model
        processor = TextProcessor(
            archive_dir=archive_dir,
            output_file=output_file,
            spacy_model=args.model
        )
        
        # Process text files
        print(f"Processing text files from {archive_dir}...")
        result = processor.process_text_files(force_reprocess=args.force)
        
        if result:
            print(f"Successfully processed text files. Output saved to {output_file}")
            print(f"Processed {result['files_processed']} files")
            print(f"Found {result['total_blocks']} content blocks")
            print(f"New files: {result['new_files']}")
            print(f"Updated files: {result['updated_files']}")
            return 0
        else:
            print("Error processing text files. See above for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
