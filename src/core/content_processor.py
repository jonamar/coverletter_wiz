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
import hashlib

# Import config for data directory paths
from src.config import DATA_DIR

# Import DataManager for centralized data access
from src.core.data_manager import DataManager

# Constants
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
    
    def __init__(self, json_file=None) -> None:
        """Initialize the ContentProcessor.
        
        Args:
            json_file: Path to the JSON file containing processed cover letters.
                       If None, uses the default file from DataManager.
        """
        # Initialize data manager for centralized data access
        self.data_manager = DataManager(content_file=json_file) if json_file else DataManager()
        
        # Get data from the data manager
        self.data = self.data_manager.data
        self.content_blocks = self.data_manager.get_content_blocks()
        
        # Calculate statistics
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
    
    def _save_ratings(self) -> bool:
        """Save the current ratings to the content file.
        
        Updates the ratings for all content blocks in the data manager.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Get the data manager
            from src.core.data_manager import DataManager
            data_manager = DataManager()
            
            # For test compatibility, check if update_ratings method exists and use it
            if hasattr(data_manager, 'update_ratings') and callable(getattr(data_manager, 'update_ratings')):
                result = data_manager.update_ratings(self.content_blocks)
                if not result:
                    print("Error saving ratings: DataManager.update_ratings returned False")
                    return False
                return True
            
            # Track changes
            changes_made = False
            
            # Update ratings for each block
            for block in self.content_blocks:
                block_id = block.get("id")
                text = block.get("text", "")
                
                if not (block_id or text):
                    continue
                
                # Find the block in the data manager
                found = False
                
                # First try to find by ID if available
                if block_id:
                    stored_block = data_manager.get_block_by_id(block_id)
                    if stored_block:
                        # Update the stored block
                        if stored_block.get("rating") != block.get("rating"):
                            stored_block["rating"] = block.get("rating", 0)
                            stored_block["batch_rating"] = block.get("batch_rating", False)
                            changes_made = True
                        found = True
                
                # If not found by ID, try to find by text
                if not found and text:
                    for filename, file_data in data_manager.data.items():
                        if not isinstance(file_data, dict) or "content" not in file_data:
                            continue
                            
                        for paragraph in file_data.get("content", {}).get("paragraphs", []):
                            for sentence in paragraph.get("sentences", []):
                                if sentence.get("text") == text:
                                    # Update the stored block
                                    if sentence.get("rating") != block.get("rating"):
                                        sentence["rating"] = block.get("rating", 0)
                                        sentence["batch_rating"] = block.get("batch_rating", False)
                                        changes_made = True
                                    
                                    # If the block has an ID but the stored one doesn't, add it
                                    if block_id and not sentence.get("id"):
                                        sentence["id"] = block_id
                                        changes_made = True
                                        
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
            
            # Save if changes were made
            if changes_made:
                return data_manager.save_data()
            
            # No changes detected
            print("No changes to ratings detected.")
            return True
            
        except (IOError, PermissionError) as e:
            print(f"Error saving ratings - Permission or IO error: {e}")
            traceback.print_exc()
            return False
        except OSError as e:
            print(f"Error saving ratings - OS error (possibly disk space): {e}")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"Error saving ratings: {e}")
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
    
    def _get_pair_id(self, block1: Dict[str, Any], block2: Dict[str, Any]) -> str:
        """Generate a unique ID for a pair of content blocks.
        
        Uses block IDs if available, otherwise falls back to text content.
        Ensures consistent ordering regardless of which block is passed first.
        
        Args:
            block1: First content block
            block2: Second content block
            
        Returns:
            str: A unique identifier for this pair of blocks
        """
        # Try to use block IDs if available
        id1 = block1.get("id")
        id2 = block2.get("id")
        
        if id1 and id2:
            # Sort IDs to ensure consistent ordering
            sorted_ids = sorted([id1, id2])
            return f"{sorted_ids[0]}:{sorted_ids[1]}"
        
        # Fall back to text-based pair ID
        text1 = block1.get("text", "")
        text2 = block2.get("text", "")
        
        # Create hashes of the text content
        hash1 = hashlib.md5(text1.encode()).hexdigest()[:8]
        hash2 = hashlib.md5(text2.encode()).hexdigest()[:8]
        
        # Sort hashes to ensure consistent ordering
        sorted_hashes = sorted([hash1, hash2])
        return f"{sorted_hashes[0]}:{sorted_hashes[1]}"

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
                
                print(f"{i:<4} {category_display:<25} {len(blocks):<7} {avg_rating:.1f}/10{' ':<6} {status}")
            
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
            String command to signal the next action ('menu' to return to category menu)
        """
        # Create a copy of the tournament blocks for this category
        category_blocks = tournament_blocks.copy()
        
        # Check if we have enough blocks for a tournament
        if len(category_blocks) < 2:
            print(f"Not enough content blocks in category '{category}' for a tournament.")
            print(f"Need at least 2 blocks, but only found {len(category_blocks)}.")
            input("Press Enter to continue...")
            return "menu"  # Signal to return to category menu
            
        # Initialize tracking variables
        compared_pairs: Set[str] = set()  # Track which pairs have been compared
        rounds = 0
        max_rounds = 10  # Limit the number of rounds to avoid an endless tournament
        
        # Get similarity-based pairs for this category
        similarity_pairs = self._get_similar_content_pairs(category_blocks)
        
        # If no similar pairs found, fall back to random pairing
        if not similarity_pairs:
            print("No similar content pairs found. Using random pairing.")
            # Main tournament loop with random pairing
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
                
                result = self._run_tournament_round(category, chosen_group, compared_pairs, rounds)
                if result == "menu":
                    return "menu"
        else:
            # Main tournament loop with similarity-based pairing
            while rounds < max_rounds and similarity_pairs:
                rounds += 1
                
                # Get the most similar pair that hasn't been compared yet
                current_pair = None
                for pair in similarity_pairs:
                    pair_id = self._get_pair_id(pair[0], pair[1])
                    if pair_id not in compared_pairs:
                        current_pair = pair
                        break
                
                if not current_pair:
                    print("\nAll similar pairs have been compared in this category.")
                    print("Tournament complete!")
                    break
                
                # Extract the blocks and similarity
                block1, block2, similarity = current_pair
                
                print(f"\n--- SIMILARITY TOURNAMENT - ROUND {rounds} - CATEGORY: {category.upper()} ---")
                print(f"These content blocks are {similarity:.1%} similar. Compare them and choose the stronger one.")
                print("Judge them on clarity, impact, and professionalism.\n")
                
                # Display the blocks
                print(f"1. ID: {block1.get('id', 'N/A')} | Rating: {block1.get('rating', 0):.1f}")
                print(f"   {block1['text']}")
                print()
                print(f"2. ID: {block2.get('id', 'N/A')} | Rating: {block2.get('rating', 0):.1f}")
                print(f"   {block2['text']}")
                print()
                
                # Show differences if similarity is high
                if similarity > 0.8:
                    from src.utils.diff_utils import print_text_differences
                    print("\nDetailed differences (red = unique to #1, green = unique to #2):")
                    print_text_differences(block1['text'], block2['text'])
                    print()
                
                # Ask for user input
                while True:
                    try:
                        choice = input("\nEnter the number of the strongest block (or 'q' to quit): ").strip()
                        
                        if choice.lower() == 'q':
                            return "menu"  # Signal to return to category menu
                            
                        choice_idx = int(choice) - 1
                        
                        if choice_idx < 0 or choice_idx >= 2:
                            print("Please enter 1 or 2.")
                            continue
                        
                        # Get the winning and losing blocks
                        winner = [block1, block2][choice_idx]
                        loser = [block1, block2][1 - choice_idx]
                        
                        # Calculate updated ratings
                        winner_old_rating = winner.get("rating", 0)
                        winner_new_rating = min(winner_old_rating + TOURNAMENT_WIN_RATING_CHANGE, TOURNAMENT_MAX_RATING)
                        
                        loser_old_rating = loser.get("rating", 0)
                        loser_new_rating = max(loser_old_rating - TOURNAMENT_LOSE_RATING_CHANGE, 0)
                        
                        # Update winner
                        print(f"\nBlock {choice_idx+1} rating increased: {winner_old_rating:.1f} -> {winner_new_rating:.1f}")
                        
                        # Update in content_blocks
                        for content_block in self.content_blocks:
                            if content_block.get("id") == winner.get("id") or content_block.get("text") == winner.get("text"):
                                content_block["rating"] = winner_new_rating
                                
                                # Check if this block now qualifies for legends tournament
                                if winner_new_rating >= LEGENDS_MIN_RATING and content_block not in self.legends_blocks:
                                    self.legends_blocks.append(content_block)
                                    print(f"\n🎖️ CONGRATULATIONS! Block has reached LEGEND status! 🎖️")
                                    print("This content is now eligible for the legends tournament.")
                        
                        # Update loser
                        if loser_old_rating != loser_new_rating:
                            print(f"Block {2-choice_idx} rating decreased: {loser_old_rating:.1f} -> {loser_new_rating:.1f}")
                            
                            # Update in content_blocks
                            for content_block in self.content_blocks:
                                if content_block.get("id") == loser.get("id") or content_block.get("text") == loser.get("text"):
                                    content_block["rating"] = loser_new_rating
                                    
                                    # Check if this block should be removed from legends
                                    if loser_new_rating < LEGENDS_MIN_RATING and content_block in self.legends_blocks:
                                        self.legends_blocks.remove(content_block)
                        
                        # Update the compared pairs set
                        pair_id = self._get_pair_id(block1, block2)
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
        
    def _get_similar_content_pairs(self, category_blocks: List[Dict[str, Any]], 
                                  similarity_threshold: float = 0.7) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
        """Get pairs of content blocks sorted by similarity.
        
        Args:
            category_blocks: List of content blocks in a category
            similarity_threshold: Minimum similarity threshold (0.0-1.0)
            
        Returns:
            List of tuples (block1, block2, similarity_score) sorted by similarity (highest first)
        """
        from src.utils.spacy_utils import analyze_content_block_similarity
        
        # Get similarity map for all blocks
        similarity_map = analyze_content_block_similarity(category_blocks)
        
        # Convert to pairs format
        similarity_pairs = []
        compared_pairs = set()
        
        for block1 in category_blocks:
            text1 = block1.get("text", "")
            if not text1 or text1 not in similarity_map:
                continue
                
            # Get similar blocks for this block
            similar_blocks = similarity_map[text1]
            
            for similar_block_info in similar_blocks:
                similarity = similar_block_info.get("similarity", 0)
                text2 = similar_block_info.get("text", "")
                
                # Skip if below threshold
                if similarity < similarity_threshold:
                    continue
                    
                # Find the actual block object for text2
                block2 = None
                for b in category_blocks:
                    if b.get("text", "") == text2:
                        block2 = b
                        break
                
                if not block2:
                    continue
                    
                # Skip if we've already compared this pair
                pair_id = self._get_pair_id(block1, block2)
                if pair_id in compared_pairs:
                    continue
                    
                # Add to pairs
                similarity_pairs.append((block1, block2, similarity))
                compared_pairs.add(pair_id)
        
        # Sort by similarity (highest first)
        return sorted(similarity_pairs, key=lambda x: x[2], reverse=True)

    def _run_tournament_round(self, category: str, chosen_group: List[Dict[str, Any]], 
                             compared_pairs: Set[str], rounds: int) -> str:
        """Run a single round of the tournament with the chosen group.
        
        Args:
            category: The category name
            chosen_group: The group of blocks to compare
            compared_pairs: Set of already compared pairs
            rounds: Current round number
            
        Returns:
            String command to signal the next action ('menu' to return to category menu)
        """
        print(f"\n--- TOURNAMENT - ROUND {rounds} - CATEGORY: {category.upper()} ---")
        print("Compare these content blocks and choose the strongest one.")
        print("Judge them on clarity, impact, and professionalism.\n")
        
        # Display the blocks
        for i, block in enumerate(chosen_group, 1):
            print(f"{i}. ID: {block.get('id', 'N/A')} | Rating: {block.get('rating', 0):.1f}")
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
                            if content_block.get("id") == block.get("id") or content_block.get("text") == block.get("text"):
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
                                if content_block.get("id") == block.get("id") or content_block.get("text") == block.get("text"):
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

    def _import_from_processed_file(self) -> None:
        """Import new content from text files.
        
        This method is maintained for backward compatibility, but now
        directly refreshes content blocks from the data manager since
        the TextProcessor writes directly to the canonical file.
        """
        # Refresh content blocks from the data manager
        self.content_blocks = self.data_manager.get_content_blocks()
        
        # Recalculate statistics
        self.total_blocks = len(self.content_blocks)
        self.rated_blocks = len([b for b in self.content_blocks if b.get("rating", 0) > 0])
        self.high_rated_blocks = len([b for b in self.content_blocks if b.get("rating", 0) >= HIGH_RATING_THRESHOLD])
        
        print("Content blocks refreshed from the canonical data source.")

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
