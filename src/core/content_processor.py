#!/usr/bin/env python3
"""
Content Processor - Core module for processing cover letter content.

This module handles the extraction, grouping, and management of content blocks
from cover letters, replacing the functionality from the original sentence_rater.py.
"""

from __future__ import annotations

import json
import random
import math
import os
import spacy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any, Union, DefaultDict
from collections import defaultdict
from datetime import datetime
import uuid
import time

# Import config for data directory paths
from src.config import DATA_DIR

# Constants
JSON_FILE = os.path.join(DATA_DIR, "json/cover_letter_content.json")
BATCH_RATING_SCALE = 10   # 1-10 scale for batch ratings
FILTER_THRESHOLD = 2      # Ratings <= this value are filtered out
BATCH_SIZE = 10           # Number of items to show in each batch
REFINEMENT_THRESHOLD = 1  # Maximum rating difference to consider for refinement
HIGH_RATING_THRESHOLD = 7 # Minimum rating to be considered high quality
TOURNAMENT_MIN_RATING = 5 # Minimum rating to be included in tournament
TOURNAMENT_MAX_RATING = 10.0 # Maximum rating to be included in tournament
TOURNAMENT_GROUP_SIZE = 2 # Number of content blocks to compare in each tournament round
TOURNAMENT_WIN_RATING_CHANGE = 1.0  # Rating adjustment amount for tournament winners
TOURNAMENT_LOSE_RATING_CHANGE = 0.5 # Rating adjustment amount for tournament losers

# Legends tournament constants
LEGENDS_MIN_RATING = 10.0  # Minimum rating to be included in legends tournament
LEGENDS_MAX_RATING = 12.0  # Maximum possible rating in legends tournament
LEGENDS_WIN_RATING_CHANGE = 0.5  # Rating adjustment for legends winners (smaller to avoid inflation)
LEGENDS_LOSE_RATING_CHANGE = 0.0  # Legends losers don't lose points

# Category refinement constants
CATEGORY_REFINED_THRESHOLD = 8.0  # Average rating threshold to consider a category "refined"
CATEGORY_COMPLETION_THRESHOLD = 0.7  # Percentage of content blocks above HIGH_RATING_THRESHOLD to consider "complete"

class ContentProcessor:
    """Process and manage cover letter content blocks.
    
    The ContentProcessor is responsible for:
    1. Loading and parsing content blocks from raw cover letters
    2. Implementing various rating systems:
       - Batch rating for initial rating assignment
       - Tournament mode for comparing blocks within categories
       - Legends tournament for top-rated content (10.0+)
       - Category refinement for organizing by topic
    3. Tracking statistics and ratings for all content blocks
    4. Exporting high-rated content blocks for cover letter creation
    
    The class implements a comprehensive rating system that helps identify 
    the strongest content for use in matching to job requirements.
    """
    
    def __init__(self, json_file: str = JSON_FILE) -> None:
        """Initialize the ContentProcessor with the given JSON file.
        
        Args:
            json_file: Path to the JSON file containing processed cover letters.
        """
        self.json_file = json_file
        self.data = self._load_content_data()
        self.content_blocks = self._extract_content_blocks()
        self.total_blocks = len(self.content_blocks)
        self.rated_blocks = len([b for b in self.content_blocks if b.get("rating", 0) > 0])
        self.high_rated_blocks = len([b for b in self.content_blocks if b.get("rating", 0) >= HIGH_RATING_THRESHOLD])
        self.perfect_blocks: List[Dict[str, Any]] = []  # Track blocks that reach a perfect score
        self.legends_blocks: List[Dict[str, Any]] = []  # Track blocks in the legends tournament
        self.category_stats: Dict[str, Any] = {}  # Track category refinement statistics
        
        # Initialize legends blocks
        for block in self.content_blocks:
            if block.get("rating", 0) >= LEGENDS_MIN_RATING and block not in self.legends_blocks:
                self.legends_blocks.append(block)
        
    def _load_content_data(self) -> Dict[str, Any]:
        """Load content data from JSON file.
        
        The function includes robust error handling for common file issues:
        - File not found
        - Malformed JSON
        - Empty content
        - Permission issues
        
        Returns:
            Content data dictionary with file metadata and content blocks.
            
        Raises:
            json.JSONDecodeError: If the JSON file contains invalid JSON.
            PermissionError: If the file cannot be read due to permission issues.
        """
        try:
            # Check if file exists
            if not os.path.exists(self.json_file):
                print(f"Error: Content file {self.json_file} does not exist.")
                print("Creating empty content structure.")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat()}}
                
            # Check if file is readable
            if not os.access(self.json_file, os.R_OK):
                print(f"Error: No read permission for {self.json_file}.")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "permission_denied"}}
                
            # Try to read the file
            with open(self.json_file, "r") as f:
                content_data = json.load(f)
                
            # Validate content structure
            if not isinstance(content_data, dict):
                print(f"Error: Content file {self.json_file} has invalid format (root not a dictionary).")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "invalid_format"}}
                
            # Check for empty content
            if not content_data:
                print(f"Warning: Content file {self.json_file} is empty.")
                return {"metadata": {"version": "1.0", "created": datetime.now().isoformat()}}
                
            return content_data
            
        except json.JSONDecodeError as e:
            print(f"Error: Content file {self.json_file} contains invalid JSON: {e}")
            print(f"Line {e.lineno}, Column {e.colno}: {e.msg}")
            return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "invalid_json"}}
        except PermissionError:
            print(f"Error: Permission denied when trying to read {self.json_file}")
            return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": "permission_denied"}}
        except Exception as e:
            print(f"Unexpected error loading content file {self.json_file}: {e}")
            import traceback
            traceback.print_exc()
            return {"metadata": {"version": "1.0", "created": datetime.now().isoformat(), "error": str(e)}}
    
    def _extract_content_blocks(self) -> List[Dict[str, Any]]:
        """Extract unique content blocks from the JSON data.
        
        Processes the loaded JSON data to extract and deduplicate content blocks
        from all cover letters, preserving their metadata such as ratings and tags.
        
        Returns:
            A list of unique content blocks with their metadata.
        """
        unique_blocks: Dict[str, Dict[str, Any]] = {}
        
        for filename, file_data in self.data.items():
            # Skip metadata keys
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
                
            paragraphs = file_data.get("content", {}).get("paragraphs", [])
            
            for paragraph in paragraphs:
                paragraph_text = paragraph.get("text", "")
                blocks = paragraph.get("sentences", [])  # Original naming, will be renamed in future versions
                
                for block in blocks:
                    text = block.get("text", "").strip()
                    if not text:
                        continue
                        
                    # Check if this is a new block or if we should update an existing one
                    if text not in unique_blocks:
                        # Create a new entry
                        is_group = block.get("is_sentence_group", False)
                        component_sentences = block.get("component_sentences", [])
                        
                        unique_blocks[text] = {
                            "text": text,
                            "sources": [filename],
                            "rating": block.get("rating", 0),
                            "batch_rating": block.get("batch_rating", False),
                            "tags": block.get("tags", []),
                            "is_content_group": is_group,  # Renamed from is_sentence_group
                            "component_content": component_sentences,  # Renamed from component_sentences
                            "context": paragraph_text
                        }
                    else:
                        # Update existing entry
                        if filename not in unique_blocks[text]["sources"]:
                            unique_blocks[text]["sources"].append(filename)
                            
                        # Keep highest rating if multiple exist
                        if block.get("rating", 0) > unique_blocks[text].get("rating", 0):
                            unique_blocks[text]["rating"] = block.get("rating", 0)
                            unique_blocks[text]["batch_rating"] = block.get("batch_rating", False)
        
        return list(unique_blocks.values())
    
    def _save_ratings(self) -> bool:
        """Save the current ratings back to the JSON file.
        
        Updates the original JSON file with the modified ratings and metadata
        from the current content blocks.
        
        Returns:
            True if the save was successful, False otherwise.
            
        Raises:
            PermissionError: If the file cannot be written due to permission issues.
            OSError: If there are other file system related errors.
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.json_file), exist_ok=True)
            
            # Check if file is writable if it exists
            if os.path.exists(self.json_file) and not os.access(self.json_file, os.W_OK):
                print(f"Error: No write permission for {self.json_file}.")
                return False
                
            # Try to write the file
            with open(self.json_file, "w") as f:
                json.dump(self.data, f, indent=2)
                
            print(f"Ratings saved to {self.json_file}")
            return True
            
        except PermissionError:
            print(f"Error: Permission denied when trying to write to {self.json_file}")
            print("Try running with appropriate permissions or changing the output location.")
            return False
        except OSError as e:
            if "No space left on device" in str(e):
                print(f"Error: Not enough disk space to save to {self.json_file}")
            else:
                print(f"Error saving to {self.json_file}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error saving to {self.json_file}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_unrated_blocks(self) -> List[Dict[str, Any]]:
        """Get all content blocks that haven't been rated yet.
        
        Returns:
            List of content block dictionaries.
        """
        unrated = []
        
        for block in self.content_blocks:
            # If block has no rating or rating is 0, it's unrated
            if block.get("rating", 0) == 0:
                unrated.append(block)
        
        return unrated
    
    def _get_blocks_for_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of content blocks for rating.
        
        Args:
            batch_size: Number of blocks to include in the batch
            
        Returns:
            List of content block dictionaries.
        """
        # Get all unrated blocks
        unrated_blocks = self._get_unrated_blocks()
        
        # If we have fewer blocks than the batch size, return all of them
        if len(unrated_blocks) <= batch_size:
            return unrated_blocks
        
        # Otherwise, return a random batch
        return random.sample(unrated_blocks, batch_size)
    
    def _show_stats(self) -> None:
        """Show statistics about the current state of the ratings."""
        print("\nContent Block Statistics:")
        print("-" * 50)
        print(f"Total blocks: {self.total_blocks}")
        print(f"Rated blocks: {self.rated_blocks} ({self.rated_blocks/self.total_blocks*100:.1f}%)")
        print(f"High-rated blocks (>= {HIGH_RATING_THRESHOLD}): {self.high_rated_blocks} ({self.high_rated_blocks/self.total_blocks*100:.1f}%)")
        
        # Get category statistics
        categories = self._get_categories_from_blocks()
        category_stats = categories.get("stats", {})
        
        # Show top 5 most populated categories
        top_categories = sorted(category_stats.items(), key=lambda x: x[1].get("count", 0), reverse=True)[:5]
        if top_categories:
            print("\nTop 5 Categories:")
            print(f"{'Category':<25} {'Count':<8} {'Avg Rating':<12} {'Completion %':<12}")
            print("-" * 60)
            for category, stats in top_categories:
                avg_rating = stats.get("avg_rating", 0)
                completion = stats.get("completion", 0) * 100
                print(f"{category:<25} {stats.get('count', 0):<8} {avg_rating:.1f}/10{' '*8} {completion:.1f}%")
        
        # Show legend blocks
        legend_count = len(self.legends_blocks)
        if legend_count > 0:
            max_rated = max(self.legends_blocks, key=lambda s: s.get("rating", 0))
            max_rating = max_rated.get("rating", 0)
            print(f"\nLegends blocks (>= {LEGENDS_MIN_RATING}): {legend_count} (highest rating: {max_rating:.1f})")
        else:
            print(f"\nNo legends blocks (>= {LEGENDS_MIN_RATING}) yet.")
    
    def _get_categories_from_blocks(self) -> Dict[str, Any]:
        """Get a dictionary of categories mapped to content blocks that have that tag.
        Separates regular tournament blocks from legends blocks.
        
        Returns:
            Dictionary with:
            - 'regular': Dict mapping category names to lists of regular blocks
            - 'legends': Dict mapping category names to lists of legends blocks
            - 'stats': Dict with statistics for each category
        """
        categories_regular = defaultdict(list)
        categories_legends = defaultdict(list)
        category_stats = {}
        
        # Calculate total blocks with ratings for completion percentage
        high_rated_blocks = [b for b in self.content_blocks if b.get("rating", 0) >= HIGH_RATING_THRESHOLD]
        
        # Separate blocks by category/tag
        for block in self.content_blocks:
            rating = block.get("rating", 0)
            if rating >= TOURNAMENT_MIN_RATING:
                for tag in block.get("tags", []):
                    # Skip very generic tags
                    if len(tag) <= 3 or tag in ["the", "and", "for", "with"]:
                        continue
                    
                    # Add to appropriate category
                    if rating >= LEGENDS_MIN_RATING:
                        categories_legends[tag].append(block)
                    else:
                        categories_regular[tag].append(block)
        
        # Calculate statistics for each category
        all_categories = set(list(categories_regular.keys()) + list(categories_legends.keys()))
        for category in all_categories:
            regular_blocks = categories_regular.get(category, [])
            legends_blocks = categories_legends.get(category, [])
            all_blocks = regular_blocks + legends_blocks
            
            # Skip categories with very few blocks
            if len(all_blocks) < 2:
                continue
            
            # Calculate average rating
            avg_rating = sum(b.get("rating", 0) for b in all_blocks) / len(all_blocks)
            
            # Calculate how many blocks in this category are high-rated
            high_rated_in_category = [b for b in all_blocks if b.get("rating", 0) >= HIGH_RATING_THRESHOLD]
            completion = len(high_rated_in_category) / len(all_blocks) if all_blocks else 0
            
            # Determine if this category is refined
            is_refined = avg_rating >= CATEGORY_REFINED_THRESHOLD
            is_complete = completion >= CATEGORY_COMPLETION_THRESHOLD
            
            category_stats[category] = {
                "count": len(all_blocks),
                "regular_count": len(regular_blocks),
                "legends_count": len(legends_blocks),
                "avg_rating": avg_rating,
                "completion": completion,
                "is_refined": is_refined,
                "is_complete": is_complete
            }
        
        return {
            "regular": dict(categories_regular),
            "legends": dict(categories_legends),
            "stats": category_stats
        }
    
    def _show_category_status(self, show_all: bool = True) -> None:
        """Show a comprehensive view of all categories and their status.
        
        Args:
            show_all: Whether to show all categories or only those with content blocks
        """
        categories = self._get_categories_from_blocks()
        category_stats = categories.get("stats", {})
        
        if not category_stats:
            print("No categories found with rated content blocks.")
            return
        
        print("\nCategory Status:")
        print(f"{'Category':<25} {'Count':<8} {'Regular':<8} {'Legends':<8} {'Avg Rating':<12} {'Completion %':<12} {'Status':<10}")
        print("-" * 85)
        
        # Sort by count
        sorted_stats = sorted(category_stats.items(), key=lambda x: x[1].get("count", 0), reverse=True)
        
        for category, stats in sorted_stats:
            # Skip categories with very few blocks if show_all is False
            if not show_all and stats.get("count", 0) < 3:
                continue
                
            avg_rating = stats.get("avg_rating", 0)
            completion = stats.get("completion", 0) * 100
            is_refined = stats.get("is_refined", False)
            is_complete = stats.get("is_complete", False)
            
            status = ""
            if is_refined and is_complete:
                status = "✅ Done"
            elif is_refined:
                status = "🟨 Refined"
            elif is_complete:
                status = "🟦 Complete"
            else:
                status = "🟥 In progress"
            
            print(f"{category:<25} {stats.get('count', 0):<8} {stats.get('regular_count', 0):<8} {stats.get('legends_count', 0):<8} {avg_rating:.1f}/10{' '*6} {completion:.1f}%{' '*7} {status}")
    
    def _show_legends_blocks(self) -> None:
        """Show the content blocks that have reached legend status (rating >= 10)."""
        if not self.legends_blocks:
            print("No content blocks have reached legend status yet.")
            return
        
        print(f"\nLegend Content Blocks (Rating >= {LEGENDS_MIN_RATING}):")
        print("-" * 80)
        
        # Sort by rating (highest first)
        sorted_legends = sorted(self.legends_blocks, key=lambda s: s.get("rating", 0), reverse=True)
        
        for i, block in enumerate(sorted_legends, 1):
            rating = block.get("rating", 0)
            text = block.get("text", "")
            print(f"{i}. [{rating:.1f}] {text[:150]}{'...' if len(text) > 150 else ''}")
    
    def _run_legends_tournament(self) -> None:
        """
        Run a tournament specifically for content blocks that have reached legend status (rating >= 10).
        
        This method implements a special tournament for the absolute best content:
        1. Only blocks rated 10.0 or higher qualify as "legends"
        2. Uses an extended rating scale (up to 12.0) for finer differentiation
        3. Allows cross-category competition of top-tier content
        4. Prevents rating inflation by requiring tougher competition
        5. Identifies the very best content blocks across your entire library
        
        Legend blocks receive special treatment in matching algorithms and reports.
        """
        print("\nLegends Tournament - The Ultimate Battle of the Best Content")
        print("=" * 70)
        
        # Reset legends blocks list
        self.legends_blocks = []
        for block in self.content_blocks:
            if block.get("rating", 0) >= LEGENDS_MIN_RATING and block not in self.legends_blocks:
                self.legends_blocks.append(block)
        
        if not self.legends_blocks:
            print("No content blocks have reached legend status yet.")
            return
        
        # Get categories from legends blocks
        categories = self._get_categories_from_blocks()
        
        # Filter to only include categories with legend blocks
        legends_categories = {}
        for category, blocks in categories["legends"].items():
            if blocks:
                legends_categories[category] = blocks
        
        if not legends_categories:
            print("No categories with legend blocks found.")
            return
            
        # Calculate legends stats
        total_legends = len(self.legends_blocks)
        max_rated = max(self.legends_blocks, key=lambda s: s.get("rating", 0))
        max_rating = max_rated.get("rating", 0)
        
        # Sort categories by number of legend blocks (most first)
        sorted_categories = sorted(
            legends_categories.items(), 
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        while True:
            print(f"\nLegends Tournament - {total_legends} blocks, highest rating: {max_rating:.1f}")
            print(f"Select a category to run a tournament for legend blocks:")
            print(f"{'#':<4} {'Category':<25} {'Count':<7} {'Avg Rating'}")
            print("-" * 50)
            
            for i, (category, blocks) in enumerate(sorted_categories, 1):
                category_display = category[:22] + "..." if len(category) > 25 else category
                avg_rating = sum(s.get("rating", 0) for s in blocks) / len(blocks)
                print(f"{i:<4} {category_display:<25} {len(blocks):<7} {avg_rating:.1f}/12")
                
            # Let user select a category
            try:
                category_input = input("\nSelect category number, 's' for status overview, 'r' for regular tournament, or 'q' to quit: ").strip().lower()
                
                if category_input == 'q':
                    # Show comprehensive status before exiting
                    self._show_category_status()
                    self._show_legends_blocks()
                    return
                elif category_input == 's':
                    # Show comprehensive category status
                    self._show_category_status()
                    input("\nPress Enter to continue...")
                    continue
                elif category_input == 'r':
                    # Switch to regular tournament
                    print("Switching to regular tournament mode...")
                    self._run_tournament_mode()
                    return  # Exit legends tournament after regular tournament completes
                category_idx = int(category_input) - 1
                
                if 0 <= category_idx < len(sorted_categories):
                    selected_category = sorted_categories[category_idx][0]
                    result = self._run_legends_category_tournament(selected_category, sorted_categories[category_idx][1])
                    
                    # If user quit from the tournament, exit completely
                    if result == "quit":
                        # Show comprehensive status before exiting
                        self._show_category_status()
                        return
                    
                    # If returning to menu, update categories
                    if result == "menu":
                        # Recalculate categories
                        categories = self._get_categories_from_blocks()
                        
                        # Filter to only include categories with legend blocks
                        legends_categories = {}
                        for category, blocks in categories["legends"].items():
                            if blocks:
                                legends_categories[category] = blocks
                        
                        if not legends_categories:
                            print(f"No more legends found.")
                            # Show comprehensive status before exiting
                            self._show_category_status()
                            self._show_legends_blocks()
                            return
                            
                        # Recalculate legends stats
                        total_legends = len(self.legends_blocks)
                        max_rated = max(self.legends_blocks, key=lambda s: s.get("rating", 0))
                        max_rating = max_rated.get("rating", 0)
                        
                        # Re-sort categories
                        sorted_categories = sorted(
                            legends_categories.items(), 
                            key=lambda x: len(x[1]),
                            reverse=True
                        )
                else:
                    print("Invalid category number.")
            except ValueError:
                print("Please enter a valid number or command.")
    
    def _run_legends_category_tournament(self, category: str, tournament_blocks: List[Dict[str, Any]]) -> str:
        """Run a tournament for a specific category of legend content blocks.
        
        Presents the user with pairs of content blocks from the same category to compare.
        The winner's rating is increased, while losers maintain their current rating.
        
        Args:
            category: The category name to run the tournament for.
            tournament_blocks: List of content blocks in this category.
            
        Returns:
            String command to signal the next action ('menu' to return to category menu).
        """
        # Create a copy of the tournament blocks for this category
        category_blocks = tournament_blocks.copy()
        
        # Check if we have enough blocks for a tournament
        if len(category_blocks) < TOURNAMENT_GROUP_SIZE:
            print(f"Not enough content blocks in category '{category}' for a tournament.")
            print(f"Need at least {TOURNAMENT_GROUP_SIZE} blocks, but only found {len(category_blocks)}.")
            input("Press Enter to continue...")
            return "menu"  # Signal to return to category menu
            
        # Initialize tracking variables
        compared_pairs: Set[str] = set()  # Track which pairs have been compared
        rounds = 0
        max_rounds = 10  # Limit the number of rounds to avoid an endless tournament
        
        # Main tournament loop
        while rounds < max_rounds:
            rounds += 1
            
            # Shuffle the blocks to get random comparisons
            random.shuffle(category_blocks)
            
            # Split into groups for comparison
            groups = [category_blocks[i:i + TOURNAMENT_GROUP_SIZE] 
                     for i in range(0, len(category_blocks), TOURNAMENT_GROUP_SIZE)]
            
            # Only keep groups with the right number of items
            complete_groups = [g for g in groups if len(g) == TOURNAMENT_GROUP_SIZE]
            
            if not complete_groups:
                print("No more complete groups to compare.")
                break
                
            # Choose a random group that hasn't been fully compared before
            valid_groups = []
            for group in complete_groups:
                # Check if all pairs in this group have already been compared
                all_compared = True
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pair_id = self._get_pair_id(group[i], group[j])
                        if pair_id not in compared_pairs:
                            all_compared = False
                            break
                    if not all_compared:
                        break
                        
                if not all_compared:
                    valid_groups.append(group)
                    
            if not valid_groups:
                print("\nAll possible pairs have been compared in this category.")
                print("Tournament complete!")
                break
                
            # Choose a random group
            chosen_group = random.choice(valid_groups)
            
            print(f"\n--- LEGENDS TOURNAMENT - ROUND {rounds} - CATEGORY: {category.upper()} ---")
            print("Compare these content blocks and choose the strongest one.")
            print("These are LEGEND tier blocks (rating 10+) competing for even higher ratings.")
            print("Judge them on clarity, impact, and professionalism.\n")
            
            # Display the blocks
            for i, block in enumerate(chosen_group, 1):
                print(f"{i}. Rating: {block.get('rating', 0):.1f}")
                print(f"   {block['text']}")
                print()
                
            # Ask for user input
            while True:
                try:
                    choice = input("\nEnter the number of the strongest block (or 'q' to quit): ").strip()
                    
                    if choice.lower() == 'q':
                        return "menu"  # Signal to return to category menu
                        
                    choice_idx = int(choice) - 1
                    
                    if choice_idx < 0 or choice_idx >= len(chosen_group):
                        print(f"Please enter a number between 1 and {len(chosen_group)}.")
                        continue
                        
                    # Get the winning block
                    winner = chosen_group[choice_idx]
                    
                    # Update ratings
                    # For legends: winners gain points, losers keep their rating
                    new_rating = min(winner.get("rating", 0) + LEGENDS_WIN_RATING_CHANGE, LEGENDS_MAX_RATING)
                    print(f"\nBlock {choice} rating increased: {winner.get('rating', 0):.1f} -> {new_rating:.1f}")
                    
                    # Update the winner's rating
                    for block in self.content_blocks:
                        if block.get("text") == winner.get("text"):
                            block["rating"] = new_rating
                            # Check if this is a new perfect block
                            if new_rating >= LEGENDS_MAX_RATING and block not in self.perfect_blocks:
                                self.perfect_blocks.append(block)
                                print(f"\n🏆 CONGRATULATIONS! Block has reached PERFECT status! 🏆")
                                print("This content block is now considered perfect and will be specially marked in exports.")
                                
                    # Update the compared pairs set
                    for i in range(len(chosen_group)):
                        for j in range(i + 1, len(chosen_group)):
                            pair_id = self._get_pair_id(chosen_group[i], chosen_group[j])
                            compared_pairs.add(pair_id)
                            
                    # Save the updated ratings
                    self._save_ratings()
                    
                    # Ask if the user wants to continue
                    cont = input("\nContinue with another round? (y/n): ").strip().lower()
                    if cont != 'y':
                        return "menu"  # Signal to return to category menu
                        
                    break  # Break the input loop and continue with the tournament
                    
                except ValueError:
                    print("Please enter a valid number or 'q'.")
            
        print("\nTournament complete! All blocks have been compared.")
        input("Press Enter to continue...")
            
        return "menu"  # Signal to return to category menu

    def _get_pair_id(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> str:
        """Generate a unique ID for a pair of content blocks to track compared pairs.
        
        Args:
            block1: First content block.
            block2: Second content block.
            
        Returns:
            A unique string identifier for this pair of blocks.
        """
        # Use the block text as the identifier, sort them to ensure consistency
        texts = sorted([block1.get("text", ""), block2.get("text", "")])
        return f"{texts[0]}|{texts[1]}"

    def _run_tournament_mode(self) -> None:
        """
        Run the tournament mode to compare content blocks within categories.
        
        This method implements a category-based tournament system that:
        1. Organizes content blocks into topic-based categories
        2. Allows for head-to-head comparisons within each category
        3. Tracks completion and refinement status for each category
        4. Adjusts ratings based on comparison outcomes
        5. Provides detailed statistics and progress tracking
        
        The tournament mode helps identify the strongest content in each 
        category and refine ratings through direct comparisons.
        """
        print("\nRegular Tournament Mode - Compare content blocks by category")
        print("=" * 70)
        
        # Get categories from content blocks
        categories = self._get_categories_from_blocks()
        category_stats = categories.get("stats", {})
        regular_categories = categories.get("regular", {})
        
        if not regular_categories:
            print("No categories with regular-rated content blocks found.")
            print("Try rating more content blocks first to reach tournament status.")
            return
        
        # Sort categories by number of blocks (most first)
        sorted_categories = sorted(
            [(category, blocks) for category, blocks in regular_categories.items() if len(blocks) >= 2],
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        if not sorted_categories:
            print("No categories with enough blocks for tournament found.")
            return
        
        while True:
            print(f"\nTournament Mode - Select a category to run a tournament:")
            print(f"{'#':<4} {'Category':<25} {'Count':<7} {'Avg Rating':<12} {'Status'}")
            print("-" * 70)
            
            for i, (category, blocks) in enumerate(sorted_categories, 1):
                category_display = category[:22] + "..." if len(category) > 25 else category
                avg_rating = sum(b.get("rating", 0) for b in blocks) / len(blocks)
                
                # Get status from category stats
                status = ""
                if category in category_stats:
                    is_refined = category_stats[category].get("is_refined", False)
                    is_complete = category_stats[category].get("is_complete", False)
                    
                    if is_refined and is_complete:
                        status = "✅ Done"
                    elif is_refined:
                        status = "🟨 Refined"
                    elif is_complete:
                        status = "🟦 Complete"
                    else:
                        status = "🟥 In progress"
                
                print(f"{i:<4} {category_display:<25} {len(blocks):<7} {avg_rating:.1f}/10{' '*5} {status}")
            
            # Let user select a category
            try:
                category_input = input("\nSelect category number, 's' for status overview, 'l' for legends tournament, or 'q' to quit: ").strip().lower()
                
                if category_input == 'q':
                    # Show comprehensive status before exiting
                    self._show_category_status()
                    self._show_legends_blocks()
                    return
                elif category_input == 's':
                    # Show comprehensive category status
                    self._show_category_status()
                    input("\nPress Enter to continue...")
                    continue
                elif category_input == 'l':
                    # Switch to legends tournament
                    print("Switching to legends tournament mode...")
                    self._run_legends_tournament()
                    return  # Exit regular tournament after legends tournament completes
                
                category_idx = int(category_input) - 1
                
                if 0 <= category_idx < len(sorted_categories):
                    selected_category = sorted_categories[category_idx][0]
                    result = self._run_category_tournament(selected_category, sorted_categories[category_idx][1])
                    
                    # If user quit from the tournament, exit completely
                    if result == "quit":
                        # Show comprehensive status before exiting
                        self._show_category_status()
                        return
                    
                    # If returning to menu, update categories
                    if result == "menu":
                        # Recalculate categories
                        categories = self._get_categories_from_blocks()
                        regular_categories = categories.get("regular", {})
                        category_stats = categories.get("stats", {})
                        
                        # Resort categories
                        sorted_categories = sorted(
                            [(category, blocks) for category, blocks in regular_categories.items() if len(blocks) >= 2],
                            key=lambda x: len(x[1]),
                            reverse=True
                        )
                        
                        if not sorted_categories:
                            print("No more categories with enough blocks for tournament.")
                            self._show_category_status()
                            return
                else:
                    print("Invalid category number.")
            except ValueError:
                print("Please enter a valid number or command.")
                
    def _run_category_tournament(self, category: str, tournament_blocks: List[Dict[str, Any]]) -> str:
        """Run a tournament for a specific category of content blocks.
        
        Presents the user with pairs of content blocks from the same category to compare.
        Ratings are adjusted based on the outcome, with winners gaining points and losers
        potentially losing points.
        
        Args:
            category: The category name to run the tournament for.
            tournament_blocks: List of content blocks in this category.
            
        Returns:
            String command to signal the next action ('menu' to return to category menu).
        """
        # Create a copy of the tournament blocks for this category
        category_blocks = tournament_blocks.copy()
        
        # Check if we have enough blocks for a tournament
        if len(category_blocks) < TOURNAMENT_GROUP_SIZE:
            print(f"Not enough content blocks in category '{category}' for a tournament.")
            print(f"Need at least {TOURNAMENT_GROUP_SIZE} blocks, but only found {len(category_blocks)}.")
            input("Press Enter to continue...")
            return "menu"  # Signal to return to category menu
            
        # Initialize tracking variables
        compared_pairs: Set[str] = set()  # Track which pairs have been compared
        rounds = 0
        max_rounds = 10  # Limit the number of rounds to avoid an endless tournament
        
        # Main tournament loop
        while rounds < max_rounds:
            rounds += 1
            
            # Shuffle the blocks to get random comparisons
            random.shuffle(category_blocks)
            
            # Split into groups for comparison
            groups = [category_blocks[i:i + TOURNAMENT_GROUP_SIZE] 
                     for i in range(0, len(category_blocks), TOURNAMENT_GROUP_SIZE)]
            
            # Only keep groups with the right number of items
            complete_groups = [g for g in groups if len(g) == TOURNAMENT_GROUP_SIZE]
            
            if not complete_groups:
                print("No more complete groups to compare.")
                break
                
            # Choose a random group that hasn't been fully compared before
            valid_groups = []
            for group in complete_groups:
                # Check if all pairs in this group have already been compared
                all_compared = True
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        pair_id = self._get_pair_id(group[i], group[j])
                        if pair_id not in compared_pairs:
                            all_compared = False
                            break
                    if not all_compared:
                        break
                        
                if not all_compared:
                    valid_groups.append(group)
                    
            if not valid_groups:
                print("\nAll possible pairs have been compared in this category.")
                print("Tournament complete!")
                break
                
            # Choose a random group
            chosen_group = random.choice(valid_groups)
            
            print(f"\n--- TOURNAMENT - ROUND {rounds} - CATEGORY: {category.upper()} ---")
            print("Compare these content blocks and choose the strongest one.")
            print("Judge them on clarity, impact, and professionalism.\n")
            
            # Display the blocks
            for i, block in enumerate(chosen_group, 1):
                print(f"{i}. Rating: {block.get('rating', 0):.1f}")
                print(f"   {block['text']}")
                print()
                
            # Ask for user input
            while True:
                try:
                    choice = input("\nEnter the number of the strongest block (or 'q' to quit): ").strip()
                    
                    if choice.lower() == 'q':
                        return "menu"  # Signal to return to category menu
                        
                    choice_idx = int(choice) - 1
                    
                    if choice_idx < 0 or choice_idx >= len(chosen_group):
                        print(f"Please enter a number between 1 and {len(chosen_group)}.")
                        continue
                        
                    # Get the winning block
                    winner = chosen_group[choice_idx]
                    
                    # Calculate updated ratings
                    winner_old_rating = winner.get("rating", 0)
                    winner_new_rating = min(winner_old_rating + TOURNAMENT_WIN_RATING_CHANGE, TOURNAMENT_MAX_RATING)
                    
                    # Update all blocks
                    for i, block in enumerate(chosen_group):
                        if i == choice_idx:  # Winner
                            print(f"\nBlock {i+1} rating increased: {winner_old_rating:.1f} -> {winner_new_rating:.1f}")
                            
                            # Update in content_blocks
                            for content_block in self.content_blocks:
                                if content_block.get("text") == block.get("text"):
                                    content_block["rating"] = winner_new_rating
                                    
                                    # Check if this block now qualifies for legends tournament
                                    if winner_new_rating >= LEGENDS_MIN_RATING and content_block not in self.legends_blocks:
                                        self.legends_blocks.append(content_block)
                                        print(f"\n🎖️ CONGRATULATIONS! Block has reached LEGEND status! 🎖️")
                                        print("This content is now eligible for the legends tournament.")
                        else:  # Losers
                            old_rating = block.get("rating", 0)
                            new_rating = max(old_rating - TOURNAMENT_LOSE_RATING_CHANGE, 0)
                            
                            if old_rating != new_rating:
                                print(f"Block {i+1} rating decreased: {old_rating:.1f} -> {new_rating:.1f}")
                                
                                # Update in content_blocks
                                for content_block in self.content_blocks:
                                    if content_block.get("text") == block.get("text"):
                                        content_block["rating"] = new_rating
                                        
                                        # Check if this block should be removed from legends
                                        if new_rating < LEGENDS_MIN_RATING and content_block in self.legends_blocks:
                                            self.legends_blocks.remove(content_block)
                    
                    # Update the compared pairs set
                    for i in range(len(chosen_group)):
                        for j in range(i + 1, len(chosen_group)):
                            pair_id = self._get_pair_id(chosen_group[i], chosen_group[j])
                            compared_pairs.add(pair_id)
                            
                    # Save the updated ratings
                    self._save_ratings()
                    
                    # Ask if the user wants to continue
                    cont = input("\nContinue with another round? (y/n): ").strip().lower()
                    if cont != 'y':
                        return "menu"  # Signal to return to category menu
                        
                    break  # Break the input loop and continue with the tournament
                    
                except ValueError:
                    print("Please enter a valid number or 'q'.")
            
        print("\nTournament complete! All blocks have been compared.")
        input("Press Enter to continue...")
            
        return "menu"  # Signal to return to category menu

    def _run_batch_rating(self, batch_size: int = 10) -> None:
        """
        Run the batch rating phase for initial content block evaluation.
        
        This method presents unrated content blocks in manageable batches, allowing
        users to assign ratings on a 1-10 scale. It also supports editing blocks 
        directly during the rating process, and tracks progress across batches.
        
        Rating scale interpretation:
        - 1-2: Poor (will be filtered out)
        - 3-5: Fair to Average
        - 6-7: Good
        - 8-10: Excellent
        
        Args:
            batch_size: Number of blocks to include in each batch
        """
        # Get all unrated blocks
        unrated_blocks = self._get_unrated_blocks()
        
        if not unrated_blocks:
            print("\nNo new content blocks to rate.")
            return
        
        print("\n" + "=" * 70)
        print("BATCH RATING PHASE - Give Initial Ratings to Content Blocks")
        print("=" * 70)
        print("Please rate each content block on a scale of 1-10:")
        print("1-2 = Poor (will be filtered out)")
        print("3-5 = Fair to Average")
        print("6-7 = Good")
        print("8-10 = Excellent")
        print("Or enter 'e' to edit the block before rating")
        
        # Calculate number of batches
        num_batches = math.ceil(len(unrated_blocks) / batch_size)
        
        # Create batches
        batches = []
        for i in range(0, len(unrated_blocks), batch_size):
            batches.append(unrated_blocks[i:i+batch_size])
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            print(f"\n{'=' * 70}")
            print(f"BATCH {batch_idx + 1} OF {num_batches}")
            print(f"{'=' * 70}")
            
            batch_ratings = []
            
            block_idx = 0
            while block_idx < len(batch):
                block = batch[block_idx]
                print(f"\n{'-' * 70}")
                print(f"{block_idx + 1}. {block['text']}")
                
                while True:
                    try:
                        rating_input = input("Rating (1-10, 'e' to edit, 's' to skip, 'q' to quit): ").lower().strip()
                        
                        if rating_input == 'q':
                            self._save_ratings()
                            print("Rating process saved and exited.")
                            return
                        elif rating_input == 's':
                            block_idx += 1
                            break
                        elif rating_input == 'e':
                            # Handle editing
                            edited_block = self._edit_block(block)
                            if edited_block:
                                # Mark original block as poor quality
                                block["rating"] = 1.0
                                block["last_rated"] = datetime.now().isoformat()
                                block["batch_rating"] = True
                                block["edited"] = True
                                
                                # Add the edited block to the data
                                self._add_edited_block(edited_block)
                                self._save_ratings()
                                
                                # Now rate the new block
                                print(f"\nNew content block: {edited_block['text']}")
                                # Continue with the normal flow but with the edited block
                                # We'll stay at the same index but replace the block
                                batch[block_idx] = edited_block
                                break
                            else:
                                # If edit was canceled, continue with the original block
                                continue
                        
                        rating = float(rating_input)
                        if 1 <= rating <= 10:
                            block["rating"] = rating
                            block["last_rated"] = datetime.now().isoformat()
                            block["batch_rating"] = True
                            batch_ratings.append(rating)
                            block_idx += 1
                            break
                        else:
                            print("Please enter a number between 1 and 10.")
                    except ValueError:
                        print("Please enter a valid number, 'e' to edit, 's' to skip, or 'q' to quit.")
            
            # Save after each batch
            self._save_ratings()
            
            # Show batch summary
            if batch_ratings:
                avg_rating = sum(batch_ratings) / len(batch_ratings)
                print(f"\n{'-' * 70}")
                print(f"Batch {batch_idx + 1} complete!")
                print(f"Average rating: {avg_rating:.1f}")
                
                if batch_idx < len(batches) - 1:
                    input("Press Enter to continue to the next batch...")
        
        print("\nBatch rating phase completed!")
        
        # Show top rated blocks
        top_blocks = sorted(self.content_blocks, key=lambda b: b.get("rating", 0), reverse=True)[:5]
        print("\nTOP RATED CONTENT BLOCKS:")
        for i, block in enumerate(top_blocks, 1):
            print(f"{i}. [{block.get('rating', 0):.1f}/10] {block['text']}")

    def _edit_block(self, block: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allow the user to edit a content block.
        
        Args:
            block: The content block dictionary to edit
            
        Returns:
            Dict: The edited content block dictionary, or None if canceled
        """
        print("\n" + "=" * 70)
        print("EDIT MODE")
        print("=" * 70)
        
        print("Original text:")
        print(block.get("text", ""))
        new_text = input("New version (leave empty to cancel): ").strip()
        
        if not new_text:
            print("Edit canceled.")
            return None
            
        # Create a new content block dictionary
        edited_block = {
            "text": new_text,
            "tags": block.get("tags", []).copy(),
            "rating": block.get("rating", 0),  # Preserve the original rating
            "is_block_group": False,  # Always create as a single block
            "component_blocks": [],
            "edited_from": block.get("text", ""),
            "edit_date": datetime.now().isoformat()
        }
        
        return edited_block
    
    def _add_edited_block(self, edited_block: Dict[str, Any]) -> None:
        """
        Add an edited content block to the appropriate location in the JSON data.
        
        Args:
            edited_block: The edited content block dictionary
        """
        # Generate a unique ID for the edited block
        edited_id = str(uuid.uuid4())
        
        # First, add to the content_blocks list for the current rating session
        self.content_blocks.append(edited_block)
        
        # Find a suitable location in the JSON data
        # We'll add it to a special "edited_blocks" section in the JSON
        if "edited_blocks" not in self.data:
            self.data["edited_blocks"] = {}
        
        self.data["edited_blocks"][edited_id] = edited_block
        
        print(f"Added edited content block with ID: {edited_id}")

    def _run_category_refinement(self) -> None:
        """
        Run a refinement tool to organize and improve content blocks by category.
        
        This mode allows users to:
        1. View completion status for each category
        2. Focus on categories that need more work
        3. Track refinement progress across the entire content library
        4. Seamlessly switch between refinement and tournament modes
        5. Systematically improve content organization by topic
        
        Category refinement helps ensure comprehensive coverage of all 
        important topics in your cover letter content.
        """
        print("\n" + "=" * 70)
        print("CATEGORY REFINEMENT MODE - Organize and Improve Content by Category")
        print("=" * 70)
        
        # Get categories
        categories = self._get_categories_from_blocks()
        
        if not categories["regular"] and not categories["legends"]:
            print(f"No categories with content blocks found.")
            return
            
        # Calculate tournament completion
        total_categories = len(categories["stats"])
        refined_categories = sum(1 for stats in categories["stats"].values() if stats["is_refined"])
        overall_completion = refined_categories / total_categories if total_categories > 0 else 0
        
        # Sort categories by refined status and then by average rating
        sorted_categories = sorted(
            categories["stats"].items(), 
            key=lambda x: (x[1]["is_refined"], -x[1]["avg_rating"])
        )
        
        # Main refinement loop
        show_refined = True  # Whether to show refined categories
        
        while True:
            print(f"\nCategory Refinement Mode - {overall_completion:.0%} complete")
            print("=" * 70)
            
            # Filter categories based on refinement status
            display_categories = []
            for category, stats in sorted_categories:
                if show_refined or not stats["is_refined"]:
                    display_categories.append((category, stats))
            
            print(f"\nSelect a category to refine:")
            print(f"{'#':<4} {'Category':<25} {'Count':<7} {'Rating':<11} {'Status'}")
            print("-" * 70)
            
            for i, (category, stats) in enumerate(display_categories, 1):
                # Truncate long category names
                category_display = category
                if len(category) > 22:
                    category_display = category[:19] + "..."
                
                status = "✅ Refined" if stats["is_refined"] else "🔄 Needs work"
                print(f"{i:<4} {category_display:<25} {stats['regular_count']:<7} {stats['avg_rating']:.1f}/10{' ':<6} {status}")
                
            # Let user select a category
            try:
                category_input = input("\nSelect category number, 'r' to toggle refined, 's' for status overview, 'l' for legends, or 'q' to quit: ").strip().lower()
                
                if category_input == 'q':
                    # Show comprehensive status before exiting
                    self._show_category_status()
                    return
                elif category_input == 'r':
                    # Toggle showing refined categories
                    show_refined = not show_refined
                    continue
                elif category_input == 's':
                    # Show comprehensive category status
                    self._show_category_status()
                    input("\nPress Enter to continue...")
                    continue
                elif category_input == 'l':
                    # Switch to legends tournament
                    print("Switching to legends tournament mode...")
                    self._run_legends_tournament()
                    
                    # Refresh categories after legends tournament
                    categories = self._get_categories_from_blocks()
                    
                    # Recalculate tournament completion
                    total_categories = len(categories["stats"])
                    refined_categories = sum(1 for stats in categories["stats"].values() if stats["is_refined"])
                    overall_completion = refined_categories / total_categories if total_categories > 0 else 0
                    
                    # Re-sort categories
                    sorted_categories = sorted(
                        categories["stats"].items(), 
                        key=lambda x: (x[1]["is_refined"], -x[1]["avg_rating"])
                    )
                    continue
                    
                category_idx = int(category_input) - 1
                
                if 0 <= category_idx < len(display_categories):
                    selected_category = display_categories[category_idx][0]
                    result = self._run_category_tournament(selected_category, 
                                                         categories["regular"].get(selected_category, []), 
                                                         categories["legends"].get(selected_category, []))
                    
                    # If user quit from the tournament, exit completely
                    if result == "quit":
                        # Show comprehensive status before exiting
                        self._show_category_status()
                        return
                    
                    # If returning to menu, update categories
                    if result == "menu":
                        # Recalculate categories
                        categories = self._get_categories_from_blocks()
                        
                        if not categories["regular"] and not categories["legends"]:
                            print(f"No categories with content blocks found.")
                            self._show_category_status()
                            return
                            
                        # Recalculate tournament completion
                        total_categories = len(categories["stats"])
                        refined_categories = sum(1 for stats in categories["stats"].values() if stats["is_refined"])
                        overall_completion = refined_categories / total_categories if total_categories > 0 else 0
                        
                        # Re-sort categories
                        sorted_categories = sorted(
                            categories["stats"].items(), 
                            key=lambda x: (x[1]["is_refined"], -x[1]["avg_rating"])
                        )
                else:
                    print("Invalid category number.")
            except ValueError:
                print("Please enter a valid number or command.")
                
        print("\nCategory refinement completed!")
        
    def export_high_rated_content(self, min_rating: float = 8.0, output_file: str = None) -> None:
        """
        Export high-rated content blocks to a file or display them.
        
        This method generates a comprehensive markdown export that:
        1. Includes all content blocks above the specified rating threshold
        2. Organizes content by overall rating and by category
        3. Provides detailed statistics and metadata
        4. Creates a reference document for cover letter writing
        
        Args:
            min_rating: Minimum rating for content to be exported (default: 8.0)
            output_file: Path to the output file (if None, content will be displayed)
        """
        # Get all blocks with rating >= min_rating
        high_rated_blocks = [block for block in self.content_blocks if block.get("rating", 0) >= min_rating]
        
        if not high_rated_blocks:
            print(f"No content blocks with rating >= {min_rating} found.")
            return
        
        # Sort blocks by rating (highest first)
        high_rated_blocks.sort(key=lambda x: x.get("rating", 0), reverse=True)
        
        # Get categories for the blocks
        categories = {}
        for block in high_rated_blocks:
            for tag in block.get("tags", []):
                if tag not in categories:
                    categories[tag] = []
                categories[tag].append(block)
                
        # Create markdown output
        output = f"# High-Rated Content Blocks (Rating >= {min_rating})\n\n"
        output += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add overall summary
        output += f"## Summary\n\n"
        output += f"- Total blocks: {len(high_rated_blocks)}\n"
        output += f"- Average rating: {sum(block.get('rating', 0) for block in high_rated_blocks) / len(high_rated_blocks):.1f}/10\n"
        output += f"- Categories: {len(categories)}\n\n"
        
        # Add full list sorted by rating
        output += f"## All High-Rated Blocks\n\n"
        for i, block in enumerate(high_rated_blocks, 1):
            output += f"### {i}. [{block.get('rating', 0):.1f}/10] {block.get('text', '')[:80]}...\n\n"
            output += f"**Full text:** {block.get('text', '')}\n\n"
            output += f"**Tags:** {', '.join(block.get('tags', []))}\n\n"
            output += f"**Last rated:** {block.get('last_rated', 'Unknown')}\n\n"
            output += "---\n\n"
        
        # Add category-based lists
        output += f"## Content by Category\n\n"
        for category, blocks in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
            output += f"### {category} ({len(blocks)} blocks)\n\n"
            
            # Sort blocks within category by rating
            blocks.sort(key=lambda x: x.get("rating", 0), reverse=True)
            
            for i, block in enumerate(blocks, 1):
                output += f"{i}. [{block.get('rating', 0):.1f}/10] {block.get('text', '')}\n\n"
            
            output += "---\n\n"
        
        # Either write to file or print to console
        if output_file:
            with open(output_file, 'w') as f:
                f.write(output)
            print(f"Exported {len(high_rated_blocks)} high-rated blocks to {output_file}")
        else:
            print(output)
