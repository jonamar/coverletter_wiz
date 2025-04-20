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
    1. Loading and parsing content blocks from the content database
    2. Implementing various rating systems:
       - Batch rating for initial rating assignment
       - Tournament mode for comparing blocks within categories
    3. Tracking statistics and ratings for all content blocks
    4. Exporting high-rated content blocks for cover letter creation
    
    The class implements a comprehensive rating system that helps identify 
    the strongest content for use in matching to job requirements.
    
    Attributes:
        data_manager (DataManager): Centralized data access manager
        data (Dict[str, Any]): Complete content data structure
        content_blocks (List[Dict[str, Any]]): List of all content blocks
        total_blocks (int): Total number of content blocks
        rated_blocks (int): Number of blocks with ratings > 0
        high_rated_blocks (int): Number of blocks with ratings >= HIGH_RATING_THRESHOLD
        perfect_blocks (List[Dict[str, Any]]): Blocks with perfect scores
        legends_blocks (List[Dict[str, Any]]): Blocks in the legends tournament
        category_stats (Dict[str, Any]): Category refinement statistics
    """
    
    def __init__(self) -> None:
        """Initialize the ContentProcessor.
        
        Loads content blocks from the data manager and calculates initial statistics.
        No parameters are required as the DataManager handles data source configuration.
        
        Raises:
            RuntimeError: If there's a critical error initializing the data manager
        """
        # Initialize data manager for centralized data access
        self.data_manager = DataManager()
        
        # Get data from the data manager
        self.data = self.data_manager.data
        self.content_blocks = self.data_manager.get_content_blocks()
        
        # Calculate statistics
        self.total_blocks: int = len(self.content_blocks)
        self.rated_blocks: int = len([b for b in self.content_blocks if b.get("rating", 0) > 0])
        self.high_rated_blocks: int = len([b for b in self.content_blocks if b.get("rating", 0) >= HIGH_RATING_THRESHOLD])
        self.perfect_blocks: List[Dict[str, Any]] = []  # Track blocks that reach a perfect score
        self.legends_blocks: List[Dict[str, Any]] = []  # Track blocks in the legends tournament
        self.category_stats: Dict[str, Dict[str, Any]] = {}  # Track category refinement statistics
        
        # Initialize legends blocks
        for block in self.content_blocks:
            if block.get("rating", 0) >= LEGENDS_MIN_RATING and block not in self.legends_blocks:
                self.legends_blocks.append(block)
    
    def _save_ratings(self) -> bool:
        """Save the current ratings to the content database.
        
        Updates the ratings for all content blocks in the data manager.
        This method is called automatically after rating operations to
        ensure data persistence.
        
        Returns:
            bool: True if the save operation was successful, False otherwise.
            
        Raises:
            OSError: May be raised if there are file system issues (handled internally)
        """
        try:
            # Check if there are any changes to save
            if not self.data_manager.update_ratings(self.content_blocks):
                print("No changes to ratings detected.")
                return True
                
            # Save the updated data
            result = self.data_manager.save_data()
            if result:
                print("Ratings saved successfully.")
            else:
                print("Error: Failed to save ratings.")
            return result
        except Exception as e:
            print(f"Error saving ratings: {str(e)}")
            return False

    def _get_unrated_blocks(self) -> List[Dict[str, Any]]:
        """Get all unrated content blocks.
        
        Retrieves content blocks that have not yet been rated (rating = 0).
        
        Returns:
            List[Dict[str, Any]]: A list of unrated content blocks.
        """
        return [block for block in self.content_blocks if block.get("rating", 0) == 0]
    
    def _get_rated_blocks(self) -> List[Dict[str, Any]]:
        """Get all rated content blocks.
        
        Retrieves content blocks that have been assigned a non-zero rating.
        
        Returns:
            List[Dict[str, Any]]: A list of rated content blocks with non-zero ratings.
        """
        return [block for block in self.content_blocks if block.get("rating", 0) > 0]
    
    def _get_blocks_by_rating_range(self, min_rating: float = 0.0, max_rating: float = 10.0) -> List[Dict[str, Any]]:
        """Get content blocks within a specific rating range.
        
        This method efficiently filters content blocks based on their rating values,
        allowing for more specific targeting of blocks for different operations.
        
        Args:
            min_rating: The minimum rating value (inclusive), defaults to 0.0
            max_rating: The maximum rating value (inclusive), defaults to 10.0
            
        Returns:
            List[Dict[str, Any]]: Content blocks with ratings in the specified range
        """
        return [
            block for block in self.content_blocks 
            if min_rating <= block.get("rating", 0) <= max_rating
        ]
    
    def _get_blocks_for_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a batch of content blocks for rating.
        
        Retrieves a batch of unrated content blocks for the batch rating process.
        If there are fewer unrated blocks than the batch size, returns all available blocks.
        
        Args:
            batch_size: The number of content blocks to include in the batch
            
        Returns:
            List[Dict[str, Any]]: A list of unrated content blocks up to the batch size
        """
        # Get all unrated blocks
        unrated_blocks = self._get_unrated_blocks()
        
        # Return all unrated blocks if fewer than batch size
        if len(unrated_blocks) <= batch_size:
            return unrated_blocks
            
        # Otherwise, return a batch of the specified size
        return unrated_blocks[:batch_size]
    
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

    def _run_tournament_mode(self) -> bool:
        """Run the tournament mode for comparing and refining content blocks.
        
        Tournament mode presents pairs of content blocks for side-by-side comparison,
        allowing for more precise relative rating adjustments. Winners receive a rating
        boost while losers have their ratings slightly reduced.
        
        Returns:
            bool: True if at least one tournament round was completed, False otherwise
            
        Note:
            This method handles user interaction via the console and automatically
            saves ratings after each tournament round.
        """
        # Get blocks with ratings in the tournament range
        tournament_blocks = self._get_blocks_by_rating_range(
            min_rating=TOURNAMENT_MIN_RATING, 
            max_rating=TOURNAMENT_MAX_RATING
        )
        
        # Filter out blocks with too few ratings
        tournament_blocks = [b for b in tournament_blocks if b.get("rating", 0) >= TOURNAMENT_MIN_RATING]
        
        # If not enough blocks, show message and return
        if len(tournament_blocks) < TOURNAMENT_GROUP_SIZE:
            print(f"Not enough blocks with ratings >= {TOURNAMENT_MIN_RATING} for tournament mode.")
            print(f"Need at least {TOURNAMENT_GROUP_SIZE} blocks, but only found {len(tournament_blocks)}.")
            return False
        
        print("\n" + "=" * 70)
        print("TOURNAMENT MODE - Compare Content Blocks Side by Side")
        print("=" * 70)
        print(f"You'll be shown {TOURNAMENT_GROUP_SIZE} blocks at a time.")
        print("Choose the best one to increase its rating.")
        
        # Shuffle blocks to ensure random comparisons
        random.shuffle(tournament_blocks)
        
        # Track if any tournaments were completed
        tournaments_completed = False
        
        # Process blocks in groups
        for i in range(0, len(tournament_blocks), TOURNAMENT_GROUP_SIZE):
            # Get the current group
            group = tournament_blocks[i:i+TOURNAMENT_GROUP_SIZE]
            
            # If we don't have enough blocks for a full group, break
            if len(group) < TOURNAMENT_GROUP_SIZE:
                break
                
            print("\n" + "-" * 70)
            print(f"ROUND {i//TOURNAMENT_GROUP_SIZE + 1}")
            print("-" * 70)
            
            # Display blocks
            for j, block in enumerate(group):
                print(f"\n{j+1}. [{block.get('rating', 0):.1f}] {block.get('text', '')}")
                if "tags" in block and block["tags"]:
                    print(f"   Tags: {', '.join(block['tags'])}")
            
            # Get user choice
            while True:
                try:
                    choice_input = input(f"\nWhich is best? (1-{len(group)}, 'e' to edit, 's' to skip): ").lower().strip()
                    
                    if choice_input == 's':
                        print("Skipping this round.")
                        break
                    elif choice_input == 'e':
                        # Handle editing
                        edit_idx = int(input(f"Which block to edit? (1-{len(group)}): ")) - 1
                        if 0 <= edit_idx < len(group):
                            self._edit_block(group[edit_idx])
                        else:
                            print("Invalid block number.")
                        continue
                    
                    choice = int(choice_input) - 1
                    
                    if 0 <= choice < len(group):
                        # Update ratings
                        winner = group[choice]
                        
                        # Increase winner's rating
                        winner["rating"] = min(winner.get("rating", 0) + TOURNAMENT_WIN_RATING_CHANGE, TOURNAMENT_MAX_RATING)
                        
                        # Decrease other blocks' ratings
                        for j, block in enumerate(group):
                            if j != choice:
                                block["rating"] = max(block.get("rating", 0) - TOURNAMENT_LOSE_RATING_CHANGE, TOURNAMENT_MIN_RATING)
                        
                        tournaments_completed = True
                        print(f"Block {choice+1} selected as winner!")
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(group)}.")
                except ValueError:
                    print(f"Please enter a valid number, 'e' to edit, or 's' to skip.")
            
            # Save after each round
            if tournaments_completed:
                self._save_ratings()
                
            # Ask if user wants to continue
            if i + TOURNAMENT_GROUP_SIZE < len(tournament_blocks):
                continue_input = input("\nContinue to next round? (y/n): ")
                if continue_input.lower() != "y":
                    break
        
        if tournaments_completed:
            print("\nTournament mode completed!")
        
        return tournaments_completed

    def _run_batch_rating(self) -> bool:
        """Run the batch rating process for unrated content blocks.
        
        Presents unrated content blocks in batches of BATCH_SIZE for rating.
        Users can rate each block on a scale of 1-10 or skip blocks.
        
        Returns:
            bool: True if at least one block was rated, False otherwise
            
        Note:
            This method handles user interaction via the console and automatically
            saves ratings after each batch is processed.
        """
        # Get unrated blocks
        unrated_blocks = self._get_unrated_blocks()
        
        # Create batches of BATCH_SIZE
        batches = []
        for i in range(0, len(unrated_blocks), BATCH_SIZE):
            batches.append(unrated_blocks[i:i+BATCH_SIZE])
        
        # If no unrated blocks, show message and return
        if not batches:
            print("No unrated content blocks found.")
            return False
        
        # Track if any blocks were rated
        blocks_rated = False
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            print(f"\n===== Batch {batch_idx + 1}/{len(batches)} =====")
            
            # Process each block in the batch
            for block_idx, block in enumerate(batch):
                print(f"\n----- Block {block_idx + 1}/{len(batch)} -----")
                print(f"Text: {block.get('text', 'No text available')}")
                
                # Show tags if available
                if "tags" in block and block["tags"]:
                    print(f"Tags: {', '.join(block['tags'])}")
                
                # Get rating from user
                while True:
                    try:
                        rating_input = input(f"\nRate this block (1-{BATCH_RATING_SCALE}, s to skip): ")
                        
                        # Skip this block
                        if rating_input.lower() == "s":
                            print("Skipping this block.")
                            break
                            
                        # Edit this block
                        elif rating_input.lower() == "e":
                            self._edit_block(block)
                            continue
                            
                        # Parse rating
                        rating = float(rating_input)
                        
                        # Validate rating
                        if 1 <= rating <= BATCH_RATING_SCALE:
                            block["rating"] = rating
                            block["batch_rating"] = True
                            blocks_rated = True
                            print(f"Block rated: {rating}")
                            break
                        else:
                            print(f"Please enter a rating between 1 and {BATCH_RATING_SCALE}.")
                    except ValueError:
                        print("Please enter a valid number or 's' to skip.")
            
            # Save after each batch
            if blocks_rated:
                self._save_ratings()
                
            # Ask if user wants to continue to next batch
            if batch_idx < len(batches) - 1:
                continue_input = input("\nContinue to next batch? (y/n): ")
                if continue_input.lower() != "y":
                    break
        
        # Update statistics
        self.rated_blocks = len([b for b in self.content_blocks if b.get("rating", 0) > 0])
        self.high_rated_blocks = len([b for b in self.content_blocks if b.get("rating", 0) >= HIGH_RATING_THRESHOLD])
        
        return blocks_rated

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
