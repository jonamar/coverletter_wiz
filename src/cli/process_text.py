#!/usr/bin/env python3
"""
Process Text - CLI module for processing cover letter text files.

This module handles processing text files from the text-archive directory,
extracting content blocks, and generating tags using spaCy.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, Any, Optional, List, Union

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.text_processor import TextProcessor
from src.config import DATA_DIR

def setup_argparse() -> argparse.ArgumentParser:
    """Sets up and configures the argument parser for the process_text CLI.
    
    Creates an ArgumentParser instance with arguments for controlling text processing
    options such as forcing reprocessing, specifying input/output paths, and
    selecting the spaCy model to use.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser ready for parsing arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process cover letter text files and extract content blocks."
    )
    
    parser.add_argument("--force", action="store_true", 
                       help="Force reprocessing of all files even if unchanged")
    parser.add_argument("--archive-dir", type=str, 
                       help="Directory containing text files to process")
    parser.add_argument("--output-file", type=str, 
                       help="Output JSON file for processed content")
    parser.add_argument("--model", type=str, default="en_core_web_lg",
                       help="spaCy model to use for NLP processing")
    
    return parser

def main() -> int:
    """Runs the text processing CLI with command line arguments.
    
    This function drives the text processing workflow by parsing command line arguments,
    initializing the TextProcessor, and processing text files to extract content 
    blocks with NLP analysis. It handles path configuration, error processing,
    and reporting of processing results.
    
    Returns:
        int: Exit code indicating success (0), failure (1), or
            operation cancelled (130).
            
    Raises:
        FileNotFoundError: If the archive directory does not exist.
        Exception: Various exceptions are caught internally and converted to error messages.
    """
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        # Set default paths using external data directory
        archive_dir = args.archive_dir or os.path.join(DATA_DIR, "text-archive")
        output_file = args.output_file or os.path.join(DATA_DIR, "json/processed_text_files.json")
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize text processor
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
