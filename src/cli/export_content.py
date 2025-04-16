#!/usr/bin/env python3
"""
Export Content CLI - Export high-rated content blocks to markdown files.

This module provides a command-line interface for exporting content blocks
with ratings above a specified threshold to a markdown file.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR

# Constants
DEFAULT_CONTENT_FILE = os.path.join(DATA_DIR, "json/cover_letter_content.json")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
DEFAULT_MIN_RATING = 8.0  # Default minimum rating threshold

def export_high_rated_content(min_rating: float = DEFAULT_MIN_RATING, 
                             output_file: str = None) -> str:
    """
    Export all content blocks with ratings above the specified threshold.
    
    Args:
        min_rating: Minimum rating threshold (default: 8.0)
        output_file: Optional output file path
        
    Returns:
        Path to the exported file
    """
    try:
        # Load content data
        with open(DEFAULT_CONTENT_FILE, 'r') as f:
            content_data = json.load(f)
        
        # Prepare markdown report
        report = []
        report.append(f"# High-Rated Content Blocks (Rating >= {min_rating})")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")
        
        # Extract high-rated content blocks
        high_rated_blocks = []
        
        for file_key, file_data in content_data.items():
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
            
            source = file_data.get("source_file", file_key)
            paragraphs = file_data.get("content", {}).get("paragraphs", [])
            
            for paragraph in paragraphs:
                blocks = paragraph.get("sentences", [])
                
                for block in blocks:
                    rating = float(block.get("rating", 0))
                    if rating >= min_rating:
                        high_rated_blocks.append({
                            "text": block.get("text", ""),
                            "rating": rating,
                            "tags": block.get("tags", []) + block.get("primary_tags", []),
                            "source": source
                        })
        
        # Group blocks by text to eliminate duplicates
        unique_blocks = {}
        for block in high_rated_blocks:
            text = block["text"]
            if text in unique_blocks:
                # Update existing block with higher rating if applicable
                if block["rating"] > unique_blocks[text]["rating"]:
                    unique_blocks[text]["rating"] = block["rating"]
                
                # Add source if not already present
                if block["source"] not in unique_blocks[text]["sources"]:
                    unique_blocks[text]["sources"].append(block["source"])
                
                # Add new tags if not already present
                for tag in block["tags"]:
                    if tag not in unique_blocks[text]["tags"]:
                        unique_blocks[text]["tags"].append(tag)
            else:
                # Create new entry for this text
                unique_blocks[text] = {
                    "text": text,
                    "rating": block["rating"],
                    "tags": block["tags"],
                    "sources": [block["source"]]
                }
        
        # Convert dictionary back to list and sort by rating
        deduplicated_blocks = list(unique_blocks.values())
        deduplicated_blocks.sort(key=lambda x: x["rating"], reverse=True)
        
        # Add blocks to report
        if not deduplicated_blocks:
            report.append(f"No content blocks found with rating >= {min_rating}.")
        else:
            report.append(f"Found {len(deduplicated_blocks)} unique high-rated content blocks (from {len(high_rated_blocks)} total blocks).")
            report.append("")
            
            for i, block in enumerate(deduplicated_blocks, 1):
                report.append(f"## {i}. Rating: {block['rating']:.1f}")
                report.append("")
                report.append(f"> {block['text']}")
                report.append("")
                report.append("**Tags:** " + ", ".join(block["tags"]))
                report.append("")
                report.append("**Sources:** " + ", ".join(block["sources"]))
                report.append("")
        
        # Create exports directory if it doesn't exist
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        
        # Determine output file path
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(EXPORTS_DIR, f"high_rated_content_{timestamp}.md")
        
        # Write report to file
        report_content = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        # Print preview
        print(f"Exported {len(deduplicated_blocks)} unique high-rated content blocks to {output_file}")
        
        return output_file
    
    except Exception as e:
        print(f"Error exporting high-rated content: {e}")
        import traceback
        traceback.print_exc()
        return None

def setup_argparse(parser=None):
    """Set up argument parser for the CLI."""
    if parser is None:
        parser = argparse.ArgumentParser(description="Export high-rated content blocks")
    
    parser.add_argument("--min-rating", type=float, default=DEFAULT_MIN_RATING,
                        help=f"Minimum rating threshold (default: {DEFAULT_MIN_RATING})")
    parser.add_argument("--output", help="Output file path")
    
    return parser

def main(args=None):
    """Main function to run the export tool."""
    if args is None:
        parser = setup_argparse()
        args = parser.parse_args()
    
    export_high_rated_content(min_rating=args.min_rating, output_file=args.output)

if __name__ == "__main__":
    main()
