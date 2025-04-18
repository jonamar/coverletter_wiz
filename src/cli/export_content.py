#!/usr/bin/env python3
"""
Export Content CLI - Export high-rated content blocks to markdown files.

This module provides a command-line interface for exporting content blocks
with ratings above a specified threshold to a markdown file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR
from src.utils.spacy_utils import analyze_content_block_similarity
from src.utils.diff_utils import display_text_differences

# Constants
DEFAULT_CONTENT_FILE = os.path.join(DATA_DIR, "json/cover_letter_content.json")
EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
DEFAULT_MIN_RATING = 8.0  # Default minimum rating threshold
DEFAULT_SIMILARITY_THRESHOLD = 0.8  # Default similarity threshold for semantic deduplication

def export_high_rated_content(min_rating: float = DEFAULT_MIN_RATING, 
                             output_file: Optional[str] = None,
                             use_semantic_dedup: bool = False,
                             similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Optional[str]:
    """Exports all content blocks with ratings above the specified threshold.
    
    This function reads content from the default content JSON file, filters for
    blocks with ratings at or above the minimum threshold, deduplicates by text
    content, and writes the results to a Markdown file.
    
    Args:
        min_rating: Minimum rating threshold (default: 8.0). Only content blocks
            with ratings greater than or equal to this value will be exported.
        output_file: Optional output file path. If None, a timestamped file will be
            created in the exports directory.
        use_semantic_dedup: Whether to use semantic similarity for deduplication.
        similarity_threshold: Similarity threshold for semantic deduplication.
            
    Returns:
        Path to the exported file if successful, None if an error occurred.
        
    Raises:
        FileNotFoundError: If the content file does not exist.
        json.JSONDecodeError: If the content file contains invalid JSON.
    """
    try:
        # Load content data
        with open(DEFAULT_CONTENT_FILE, 'r') as f:
            content_data = json.load(f)
        
        # Prepare markdown report
        report = []
        report.append(f"# High-Rated Content Blocks (Rating >= {min_rating})")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if use_semantic_dedup:
            report.append(f"Using semantic deduplication with similarity threshold: {similarity_threshold}")
        report.append("")
        
        # Extract high-rated content blocks
        high_rated_blocks: List[Dict[str, Any]] = []
        
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
                            "source": source,
                            "id": block.get("id", "")
                        })
        
        # Group blocks by text to eliminate duplicates
        unique_blocks: Dict[str, Dict[str, Union[str, float, List[str], List[Any]]]] = {}
        
        if use_semantic_dedup and len(high_rated_blocks) > 1:
            # Use semantic similarity for deduplication
            similarity_map = analyze_content_block_similarity(high_rated_blocks)
            
            # Process blocks in order of rating (highest first)
            sorted_blocks = sorted(high_rated_blocks, key=lambda x: x["rating"], reverse=True)
            processed_ids = set()
            
            for block in sorted_blocks:
                block_id = block.get("id", "")
                text = block["text"]
                
                # Skip if this block has already been processed as a duplicate
                if block_id and block_id in processed_ids:
                    continue
                
                # Add this block to unique blocks
                if text not in unique_blocks:
                    unique_blocks[text] = {
                        "text": text,
                        "rating": block["rating"],
                        "tags": block["tags"],
                        "sources": [block["source"]],
                        "id": block_id,
                        "similar_blocks": []
                    }
                    
                    # Find similar blocks
                    if text in similarity_map:
                        similar_blocks = similarity_map[text]
                        for similar in similar_blocks:
                            similar_text = similar["text"]
                            similarity = similar["similarity"]
                            
                            if similarity >= similarity_threshold:
                                # Find the original block for this text
                                similar_block = None
                                for b in high_rated_blocks:
                                    if b["text"] == similar_text:
                                        similar_block = b
                                        break
                                
                                if similar_block:
                                    similar_id = similar_block.get("id", "")
                                    if similar_id:
                                        processed_ids.add(similar_id)
                                    
                                    # Add to similar blocks
                                    unique_blocks[text]["similar_blocks"].append({
                                        "text": similar_text,
                                        "rating": similar_block["rating"],
                                        "similarity": similarity,
                                        "id": similar_id
                                    })
                                    
                                    # Add source if not already present
                                    if similar_block["source"] not in unique_blocks[text]["sources"]:
                                        unique_blocks[text]["sources"].append(similar_block["source"])
                                    
                                    # Add new tags if not already present
                                    for tag in similar_block["tags"]:
                                        if tag not in unique_blocks[text]["tags"]:
                                            unique_blocks[text]["tags"].append(tag)
        else:
            # Use exact text matching for deduplication (original method)
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
                        "sources": [block["source"]],
                        "id": block.get("id", "")
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
                report.append(f"## {i}. Rating: {block['rating']:.1f} | ID: {block.get('id', 'N/A')}")
                report.append("")
                report.append(f"> {block['text']}")
                report.append("")
                report.append("**Tags:** " + ", ".join(block["tags"]))
                report.append("")
                report.append("**Sources:** " + ", ".join(block["sources"]))
                
                # Add similar blocks if available
                if use_semantic_dedup and "similar_blocks" in block and block["similar_blocks"]:
                    report.append("")
                    report.append("**Similar Blocks:**")
                    for j, similar in enumerate(block["similar_blocks"], 1):
                        report.append("")
                        report.append(f"### Similar Block {j} (Similarity: {similar['similarity']:.1%} | Rating: {similar['rating']:.1f} | ID: {similar.get('id', 'N/A')})")
                        report.append("")
                        report.append(f"> {similar['text']}")
                        
                        # Show differences
                        diff1, diff2 = display_text_differences(block["text"], similar["text"])
                        report.append("")
                        report.append("**Differences:**")
                        report.append("```")
                        report.append(f"Original: {diff1}")
                        report.append(f"Similar:  {diff2}")
                        report.append("```")
                
                report.append("")
                report.append("---")
                report.append("")
        
        # Create exports directory if it doesn't exist
        os.makedirs(EXPORTS_DIR, exist_ok=True)
        
        # Determine output file path
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dedup_type = "semantic" if use_semantic_dedup else "exact"
            output_file = os.path.join(EXPORTS_DIR, f"high_rated_content_{dedup_type}_{timestamp}.md")
        
        # Write report to file
        report_content = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        # Print preview
        print(f"Exported {len(deduplicated_blocks)} unique high-rated content blocks to {output_file}")
        if use_semantic_dedup:
            total_similar = sum(len(block.get("similar_blocks", [])) for block in deduplicated_blocks)
            print(f"Found {total_similar} similar blocks using semantic deduplication")
        
        return output_file
    
    except Exception as e:
        print(f"Error exporting high-rated content: {e}")
        import traceback
        traceback.print_exc()
        return None

def setup_argparse(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Sets up argument parser for the export content CLI.
    
    Args:
        parser: Optional pre-existing ArgumentParser instance. If None, a new 
            parser will be created.
            
    Returns:
        argparse.ArgumentParser: Configured argument parser ready for parsing arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Export high-rated content blocks to a markdown file")
    
    parser.add_argument("--min-rating", type=float, default=DEFAULT_MIN_RATING,
                       help=f"Minimum rating threshold (default: {DEFAULT_MIN_RATING})")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (default: auto-generated in exports directory)")
    parser.add_argument("--semantic-dedup", action="store_true",
                       help="Use semantic similarity for deduplication")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                       help=f"Similarity threshold for semantic deduplication (default: {DEFAULT_SIMILARITY_THRESHOLD})")
    
    return parser

def main(args: Optional[argparse.Namespace] = None) -> None:
    """Runs the export content CLI with the provided arguments.
    
    This function handles the main execution flow of the export content CLI,
    processing command-line arguments and exporting high-rated content blocks
    based on the specified criteria.
    
    Args:
        args: Optional pre-parsed command line arguments. If None, arguments 
            will be parsed from sys.argv.
            
    Returns:
        None
    """
    if args is None:
        parser = setup_argparse()
        args = parser.parse_args()
    
    export_high_rated_content(
        min_rating=args.min_rating,
        output_file=args.output,
        use_semantic_dedup=args.semantic_dedup,
        similarity_threshold=args.similarity_threshold
    )

if __name__ == "__main__":
    main()
