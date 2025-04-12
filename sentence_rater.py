#!/usr/bin/env python3
"""
Sentence Rater - A CLI tool to compare and rate unique sentences and sentence groups from cover letters.

This tool extracts unique sentences and sentence groups from the processed_cover_letters.json file,
allows for batch rating of content, and saves the ratings back to the file.
"""

import json
import random
import argparse
import math
import os
import spacy
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
from datetime import datetime
import uuid
import time

# Constants
JSON_FILE = "processed_cover_letters.json"
BATCH_RATING_SCALE = 10   # 1-10 scale for batch ratings
FILTER_THRESHOLD = 2      # Ratings <= this value are filtered out
BATCH_SIZE = 10           # Number of items to show in each batch
REFINEMENT_THRESHOLD = 1  # Maximum rating difference to consider for refinement
HIGH_RATING_THRESHOLD = 7 # Minimum rating to be considered high quality
TOURNAMENT_MIN_RATING = 5 # Minimum rating to be included in tournament
TOURNAMENT_MAX_RATING = 10.0 # Maximum rating to be included in tournament
TOURNAMENT_GROUP_SIZE = 2 # Number of sentences to compare in each tournament round
TOURNAMENT_WIN_RATING_CHANGE = 1.0  # Rating adjustment amount for tournament winners
TOURNAMENT_LOSE_RATING_CHANGE = 0.5 # Rating adjustment amount for tournament losers

# Legends tournament constants
LEGENDS_MIN_RATING = 10.0  # Minimum rating to be included in legends tournament
LEGENDS_MAX_RATING = 12.0  # Maximum possible rating in legends tournament
LEGENDS_WIN_RATING_CHANGE = 0.5  # Rating adjustment for legends winners (smaller to avoid inflation)
LEGENDS_LOSE_RATING_CHANGE = 0.0  # Legends losers don't lose points

# Category refinement constants
CATEGORY_REFINED_THRESHOLD = 8.0  # Average rating threshold to consider a category "refined"
CATEGORY_COMPLETION_THRESHOLD = 0.7  # Percentage of sentences above HIGH_RATING_THRESHOLD to consider "complete"

class SentenceRater:
    """
    A class for rating sentences and sentence groups from cover letters using a batch rating approach.
    
    This class extracts unique sentences and sentence groups from the processed_cover_letters.json file,
    allows for batch rating of content, and saves the ratings back to the file.
    """
    
    def __init__(self, json_file: str = JSON_FILE):
        """
        Initialize the SentenceRater with the given JSON file.
        
        Args:
            json_file (str): Path to the JSON file containing processed cover letters
        """
        self.json_file = json_file
        self.data = self._load_data()
        self.sentences = self._extract_sentences()
        self.total_sentences = len(self.sentences)
        self.rated_sentences = len([s for s in self.sentences if s.get("rating", 0) > 0])
        self.high_rated_sentences = len([s for s in self.sentences if s.get("rating", 0) >= HIGH_RATING_THRESHOLD])
        self.perfect_sentences = []  # Track sentences that reach a perfect score
        self.legends_sentences = []  # Track sentences in the legends tournament
        self.category_stats = {}     # Track category refinement statistics
        
    def _load_data(self) -> Dict:
        """Load the JSON file containing processed cover letters."""
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {self.json_file} not found.")
            return {}
        
    def _extract_sentences(self) -> List[Dict]:
        """
        Extract unique sentences and sentence groups from the JSON data.
        
        Returns:
            List[Dict]: A list of unique sentences and sentence groups with their metadata
        """
        unique_sentences = {}
        
        for filename, file_data in self.data.items():
            # Skip metadata keys
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
                
            paragraphs = file_data.get("content", {}).get("paragraphs", [])
            
            for paragraph in paragraphs:
                paragraph_text = paragraph.get("text", "")
                sentences = paragraph.get("sentences", [])
                
                for sentence in sentences:
                    text = sentence.get("text", "").strip()
                    if not text:
                        continue
                        
                    # Check if this is a new sentence or if we should update an existing one
                    if text not in unique_sentences:
                        # Create a new entry
                        is_group = sentence.get("is_sentence_group", False)
                        component_sentences = sentence.get("component_sentences", [])
                        
                        unique_sentences[text] = {
                            "text": text,
                            "sources": [filename],
                            "rating": sentence.get("rating", 0),
                            "batch_rating": sentence.get("batch_rating", False),
                            "tags": sentence.get("tags", []),
                            "is_sentence_group": is_group,
                            "component_sentences": component_sentences,
                            "context": paragraph_text
                        }
                    else:
                        # Update existing entry
                        if filename not in unique_sentences[text]["sources"]:
                            unique_sentences[text]["sources"].append(filename)
                            
                        # Keep highest rating if multiple exist
                        if sentence.get("rating", 0) > unique_sentences[text].get("rating", 0):
                            unique_sentences[text]["rating"] = sentence.get("rating", 0)
                            unique_sentences[text]["batch_rating"] = sentence.get("batch_rating", False)
        
        return list(unique_sentences.values())
    
    def _save_ratings(self) -> None:
        """Save ratings back to the JSON file."""
        for filename, file_data in self.data.items():
            # Skip metadata keys
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
                
            paragraphs = file_data.get("content", {}).get("paragraphs", [])
            
            for paragraph in paragraphs:
                sentences = paragraph.get("sentences", [])
                
                for i, sentence in enumerate(sentences):
                    text = sentence.get("text", "").strip()
                    
                    # Find matching sentence in our processed list
                    matching_sentence = next((s for s in self.sentences if s["text"] == text), None)
                    
                    if matching_sentence:
                        # Update the rating and batch_rating flag
                        sentences[i]["rating"] = matching_sentence["rating"]
                        sentences[i]["batch_rating"] = matching_sentence.get("batch_rating", False)
                        sentences[i]["last_rated"] = datetime.now().isoformat() if matching_sentence.get("batch_rating") else sentences[i].get("last_rated", "")
        
        # Save the updated data
        with open(self.json_file, 'w') as f:
            json.dump(self.data, f, indent=2)
            
        print(f"Ratings saved to {self.json_file}")
    
    def _get_unrated_sentences(self) -> List[Dict]:
        """
        Get all sentences that haven't been rated yet.
        
        Returns:
            List[Dict]: List of sentence dictionaries
        """
        unrated = []
        
        for sentence in self.sentences:
            # If sentence has no rating or rating is 0, it's unrated
            if sentence.get("rating", 0) == 0:
                unrated.append(sentence)
        
        return unrated
    
    def _get_sentences_for_batch(self, batch_size: int) -> List[Dict]:
        """
        Get a batch of sentences for rating.
        
        Args:
            batch_size (int): Number of sentences to include in the batch
            
        Returns:
            List[Dict]: List of sentence dictionaries
        """
        # Get all unrated sentences
        unrated_sentences = self._get_unrated_sentences()
        
        # If we have fewer sentences than the batch size, return all of them
        if len(unrated_sentences) <= batch_size:
            return unrated_sentences
        
        # Otherwise, return a random batch
        return random.sample(unrated_sentences, batch_size)
    
    def _run_batch_rating(self) -> None:
        """Run the batch rating phase."""
        # Get all unrated sentences
        unrated_sentences = self._get_unrated_sentences()
        
        if not unrated_sentences:
            print("\nNo new sentences to rate.")
            return
        
        print("\n" + "=" * 40)
        print("BATCH RATING PHASE")
        print("=" * 40)
        print("Please rate each sentence on a scale of 1-10:")
        print("1-2 = Poor (will be filtered out)")
        print("3-5 = Fair to Average")
        print("6-7 = Good")
        print("8-10 = Excellent")
        print("Or enter 'e' to edit the sentence before rating")
        
        # Calculate number of batches
        num_batches = math.ceil(len(unrated_sentences) / BATCH_SIZE)
        
        # Create batches
        batches = []
        for i in range(0, len(unrated_sentences), BATCH_SIZE):
            batches.append(unrated_sentences[i:i+BATCH_SIZE])
        
        # Process each batch
        for batch_idx, batch in enumerate(batches):
            print(f"\n{'=' * 40}")
            print(f"BATCH {batch_idx + 1} OF {num_batches}")
            print(f"{'=' * 40}")
            
            batch_ratings = []
            
            sentence_idx = 0
            while sentence_idx < len(batch):
                sentence = batch[sentence_idx]
                print(f"\n{'-' * 40}")
                print(f"{sentence_idx + 1}. {sentence['text']}")
                
                while True:
                    try:
                        rating_input = input("Rating (1-10, 'e' to edit, 's' to skip, 'q' to quit): ").lower().strip()
                        
                        if rating_input == 'q':
                            self._save_ratings()
                            print("Rating process saved and exited.")
                            exit(0)
                        elif rating_input == 's':
                            sentence_idx += 1
                            break
                        elif rating_input == 'e':
                            # Handle editing
                            edited_sentence = self._edit_sentence(sentence)
                            if edited_sentence:
                                # Mark original sentence as poor quality
                                sentence["rating"] = 1.0
                                sentence["last_rated"] = datetime.now().isoformat()
                                sentence["batch_rating"] = True
                                sentence["edited"] = True
                                
                                # Add the edited sentence to the data
                                self._add_edited_sentence(edited_sentence)
                                self._save_ratings()
                                
                                # Now rate the new sentence
                                print(f"\nNew sentence: {edited_sentence['text']}")
                                # Continue with the normal flow but with the edited sentence
                                # We'll stay at the same index but replace the sentence
                                batch[sentence_idx] = edited_sentence
                                break
                            else:
                                # If edit was canceled, continue with the original sentence
                                continue
                        
                        rating = float(rating_input)
                        if 1 <= rating <= 10:
                            sentence["rating"] = rating
                            sentence["last_rated"] = datetime.now().isoformat()
                            sentence["batch_rating"] = True
                            batch_ratings.append(rating)
                            sentence_idx += 1
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
                print(f"\n{'-' * 40}")
                print(f"Batch {batch_idx + 1} complete!")
                print(f"Average rating: {avg_rating:.1f}")
                
                if batch_idx < len(batches) - 1:
                    input("Press Enter to continue to the next batch...")
        
        print("\nBatch rating phase completed!")
        
        # Show top rated sentences
        top_sentences = sorted(self.sentences, key=lambda s: s.get("rating", 0), reverse=True)[:5]
        print("\nTOP RATED SENTENCES:")
        for i, sentence in enumerate(top_sentences, 1):
            print(f"{i}. [{sentence.get('rating', 0):.1f}/10] {sentence['text']}")
    
    def _edit_sentence(self, sentence: Dict) -> Dict:
        """
        Allow the user to edit a sentence or sentence group.
        
        Args:
            sentence: The sentence dictionary to edit
            
        Returns:
            Dict: The edited sentence dictionary, or None if canceled
        """
        print("\n" + "=" * 40)
        print("EDIT MODE")
        print("=" * 40)
        
        print("Original text:")
        print(sentence.get("text", ""))
        new_text = input("New version (leave empty to cancel): ").strip()
        
        if not new_text:
            print("Edit canceled.")
            return None
            
        # Create a new sentence dictionary
        edited_sentence = {
            "text": new_text,
            "tags": sentence.get("tags", []).copy(),
            "is_sentence_group": False,  # Always create as a single sentence
            "component_sentences": [],
            "edited_from": sentence.get("text", ""),
            "edit_date": datetime.now().isoformat()
        }
        
        return edited_sentence
    
    def _add_edited_sentence(self, edited_sentence: Dict) -> None:
        """
        Add an edited sentence to the appropriate location in the JSON data.
        
        Args:
            edited_sentence: The edited sentence dictionary
        """
        # Generate a unique ID for the edited sentence
        edited_id = str(uuid.uuid4())
        
        # First, add to the sentences list for the current rating session
        self.sentences.append(edited_sentence)
        
        # Find a suitable location in the JSON data
        # We'll add it to a special "edited_sentences" section in the JSON
        if "edited_sentences" not in self.data:
            self.data["edited_sentences"] = {}
        
        self.data["edited_sentences"][edited_id] = edited_sentence
        
        print(f"Added edited sentence with ID: {edited_id}")

    def _show_stats(self) -> None:
        """Show statistics about the current state of the ratings."""
        filtered_count = len([s for s in self.sentences if s.get("rating", 0) <= FILTER_THRESHOLD])
        total_count = len(self.sentences)
        active_count = total_count - filtered_count
        
        # Count unrated sentences
        unrated_count = len(self._get_unrated_sentences())
        
        print("\n" + "=" * 40)
        print("STATS")
        print("=" * 40)
        print(f"Total unique sentences: {total_count}")
        print(f"Filtered out (rated ≤ {FILTER_THRESHOLD}): {filtered_count}")
        print(f"Active sentences: {active_count}")
        print(f"Unrated sentences: {unrated_count}")
        
        # Show rating distribution
        ratings = [s.get("rating", 0) for s in self.sentences if s.get("rating", 0) > 0]
        
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            min_rating = min(ratings)
            max_rating = max(ratings)
            
            print(f"\nRating statistics:")
            print(f"Average: {avg_rating:.1f}/10")
            print(f"Range: {min_rating:.1f} - {max_rating:.1f}")
            
            # Show distribution
            print("\nRating distribution:")
            for i in range(1, 11):
                count = sum(1 for r in ratings if i-0.5 <= r < i+0.5)
                percentage = (count / len(ratings)) * 100 if ratings else 0
                bar = "#" * int(percentage / 5)
                print(f"{i:2d}: {bar} {count} ({percentage:.1f}%)")
        
        # Show top rated sentences
        top_sentences = sorted(self.sentences, key=lambda s: s.get("rating", 0), reverse=True)[:5]
        print("\nTOP RATED SENTENCES:")
        for i, sentence in enumerate(top_sentences, 1):
            print(f"{i}. [{sentence.get('rating', 0):.1f}/10] {sentence['text']}")
        
        print("=" * 40)
    
    def _show_category_status(self, show_all: bool = True) -> None:
        """
        Show a comprehensive view of all categories and their status.
        
        Args:
            show_all: Whether to show all categories or only those with sentences
        """
        # Get categories and sentences
        categories = self._get_categories_from_sentences()
        
        # Combine regular and legends categories to get a complete picture
        all_categories = set()
        for category in list(categories["regular"].keys()) + list(categories["legends"].keys()):
            all_categories.add(category)
        
        if not all_categories:
            print("No categories found with rated sentences.")
            return
        
        # Calculate overall stats
        total_categories = len(all_categories)
        refined_categories = sum(1 for cat, stats in categories["stats"].items() if stats["is_refined"])
        categories_with_legends = sum(1 for cat in categories["legends"] if categories["legends"][cat])
        
        # Create a list of category stats for display
        category_stats = []
        for category in all_categories:
            regular_count = len(categories["regular"].get(category, []))
            legends_count = len(categories["legends"].get(category, []))
            total_count = regular_count + legends_count
            
            # Skip empty categories if not showing all
            if not show_all and total_count == 0:
                continue
                
            # Get category stats
            stats = categories["stats"].get(category, {})
            avg_rating = stats.get("avg_rating", 0)
            is_refined = stats.get("is_refined", False)
            
            # Determine status
            if legends_count > 0:
                if regular_count == 0:
                    status = "Legendary"  # All sentences are legends
                else:
                    status = "Advancing"  # Mix of regular and legends
            elif is_refined:
                status = "Refined"  # No legends but refined
            else:
                status = "In Progress"  # Regular work needed
                
            category_stats.append({
                "category": category,
                "regular_count": regular_count,
                "legends_count": legends_count,
                "total_count": total_count,
                "avg_rating": avg_rating,
                "status": status,
                "is_refined": is_refined
            })
        
        # Sort by status priority and then by total count (descending)
        status_priority = {"In Progress": 0, "Refined": 1, "Advancing": 2, "Legendary": 3}
        category_stats.sort(key=lambda x: (status_priority[x["status"]], -x["total_count"]))
        
        # Display the status table
        print("\n" + "=" * 80)
        print("CATEGORY STATUS OVERVIEW")
        print("=" * 80)
        print(f"Total Categories: {total_categories} | Refined: {refined_categories} | With Legends: {categories_with_legends}")
        print("-" * 80)
        print(f"{'Category':<25} {'Regular':<8} {'Legends':<8} {'Total':<6} {'Avg':<5} {'Status':<12}")
        print("-" * 80)
        
        for stats in category_stats:
            # Truncate long category names
            category = stats["category"]
            if len(category) > 22:
                category = category[:19] + "..."
                
            # Color-code the status
            status = stats["status"]
            status_color = ""
            if status == "Legendary":
                status_color = "\033[92m"  # Green
            elif status == "Advancing":
                status_color = "\033[94m"  # Blue
            elif status == "Refined":
                status_color = "\033[93m"  # Yellow
            else:
                status_color = "\033[37m"  # White
            
            end_color = "\033[0m"
            
            print(f"{category:<25} {stats['regular_count']:<8} {stats['legends_count']:<8} "
                  f"{stats['total_count']:<6} {stats['avg_rating']:.1f}  {status_color}{status:<12}{end_color}")
        
        print("-" * 80)
        print("Status Legend:")
        print("  In Progress - Regular sentences that need more refinement")
        print("  Refined     - Regular sentences with high average rating")
        print("  Advancing   - Mix of regular and legend sentences")
        print("  Legendary   - All sentences have reached legend status")
        print("=" * 80)
    
    def _get_categories_from_sentences(self) -> Dict[str, Dict]:
        """
        Get a dictionary of categories mapped to sentences that have that tag.
        Separates regular tournament sentences from legends sentences.
        
        Returns:
            Dictionary with:
            - 'regular': Dict mapping category names to lists of regular sentences
            - 'legends': Dict mapping category names to lists of legends sentences
            - 'stats': Dict with statistics for each category
        """
        regular_categories = {}
        legends_categories = {}
        category_stats = {}
        
        # First pass: collect all sentences by category
        for sentence in self.sentences:
            # Skip sentences with ratings below the threshold
            rating = sentence.get("rating", 0)
            if rating < TOURNAMENT_MIN_RATING:
                continue
                
            # Use primary_tags if available, otherwise use regular tags
            tags_to_use = sentence.get("primary_tags", sentence.get("tags", []))
            
            # If no primary tags but we have tag_scores, use the top 2 by confidence
            if not tags_to_use and "tag_scores" in sentence:
                # Sort tags by confidence score
                sorted_tags = sorted(sentence["tag_scores"].items(), key=lambda x: x[1], reverse=True)
                # Take top 2 tags
                tags_to_use = [tag for tag, score in sorted_tags[:2]]
            
            # Add sentence to each of its categories
            for tag in tags_to_use:
                # Determine if this is a legends sentence
                is_legend = rating >= LEGENDS_MIN_RATING
                
                # Add to appropriate category dictionary
                target_dict = legends_categories if is_legend else regular_categories
                
                if tag not in target_dict:
                    target_dict[tag] = []
                target_dict[tag].append(sentence)
                
                # Track this sentence for legends tournament if applicable
                if is_legend and sentence not in self.legends_sentences:
                    self.legends_sentences.append(sentence)
        
        # Second pass: calculate statistics for each category
        all_categories = set(list(regular_categories.keys()) + list(legends_categories.keys()))
        
        for category in all_categories:
            regular_sents = regular_categories.get(category, [])
            legends_sents = legends_categories.get(category, [])
            all_sents = regular_sents + legends_sents
            
            if not all_sents:
                continue
                
            # Calculate statistics
            avg_rating = sum(s.get("rating", 0) for s in all_sents) / len(all_sents)
            high_rated = sum(1 for s in all_sents if s.get("rating", 0) >= HIGH_RATING_THRESHOLD)
            completion_ratio = high_rated / len(all_sents) if all_sents else 0
            
            # Determine if category is refined
            is_refined = avg_rating >= CATEGORY_REFINED_THRESHOLD or completion_ratio >= CATEGORY_COMPLETION_THRESHOLD
            
            category_stats[category] = {
                "avg_rating": avg_rating,
                "total_sentences": len(all_sents),
                "regular_count": len(regular_sents),
                "legends_count": len(legends_sents),
                "high_rated_count": high_rated,
                "completion_ratio": completion_ratio,
                "is_refined": is_refined
            }
        
        # Store category stats for use elsewhere
        self.category_stats = category_stats
        
        return {
            "regular": regular_categories,
            "legends": legends_categories,
            "stats": category_stats
        }
    
    def _run_tournament_mode(self) -> None:
        """Run the Apples to Apples tournament mode to compare similar sentences."""
        print("\n" + "=" * 40)
        print("APPLES TO APPLES TOURNAMENT MODE")
        print("=" * 40)
        print(f"Compare sentences rated {TOURNAMENT_MIN_RATING}+ to refine their ratings.")
        print(f"Regular tournament: Winner gets +{TOURNAMENT_WIN_RATING_CHANGE} points, loser loses {TOURNAMENT_LOSE_RATING_CHANGE} points.")
        print(f"Legends tournament: Winner gets +{LEGENDS_WIN_RATING_CHANGE} points, losers don't lose points.")
        print(f"Sentences with rating {LEGENDS_MIN_RATING}+ are considered 'Legends' and can reach up to {LEGENDS_MAX_RATING}.")
        print("=" * 40)
        
        # Get categories and sentences
        categories = self._get_categories_from_sentences()
        
        if not categories["regular"] and not categories["legends"]:
            print(f"No categories with sentences rated {TOURNAMENT_MIN_RATING}+ found.")
            return
            
        # Calculate overall tournament completion
        total_categories = len(categories["stats"])
        refined_categories = sum(1 for stats in categories["stats"].values() if stats["is_refined"])
        overall_completion = refined_categories / total_categories if total_categories > 0 else 0
        
        # Sort categories by refinement status and average rating (prioritize unrefined and lower-rated)
        sorted_categories = sorted(
            categories["stats"].items(), 
            key=lambda x: (x[1]["is_refined"], x[1]["avg_rating"])
        )
        
        # Track whether to show refined categories
        show_refined = False
            
        while True:
            # Show tournament completion status
            print(f"\nTournament completion: {overall_completion:.0%} ({refined_categories}/{total_categories} categories refined)")
            
            # Filter categories based on refinement status
            display_categories = []
            for category, stats in sorted_categories:
                if not stats["is_refined"] or show_refined:
                    display_categories.append((category, stats))
            
            # Show available categories
            print(f"\nAvailable categories (sorted by refinement need):")
            if not show_refined:
                print(f"Showing only categories that need work. Use 'r' to toggle refined categories.")
            else:
                print(f"Showing all categories including refined ones. Use 'r' to hide refined categories.")
                
            print(f"{'#':<4} {'Category':<25} {'Total':<7} {'Avg Rating':<12} {'Status'}")
            print("-" * 70)
            
            for i, (category, stats) in enumerate(display_categories, 1):
                # Truncate long category names
                category_display = category
                if len(category) > 22:
                    category_display = category[:19] + "..."
                
                status = "✓ Refined" if stats["is_refined"] else "Needs work"
                print(f"{i:<4} {category_display:<25} {stats['regular_count']:<7} {stats['avg_rating']:.1f}/10{' ':<6} {status}")
                
            # Let user select a category
            try:
                category_input = input("\nSelect category number, 'r' to toggle refined, 's' for status overview, 'l' for legends, or 'q' to quit: ").strip().lower()
                
                if category_input == 'q':
                    # Show comprehensive status before exiting
                    self._show_category_status()
                    self._show_perfect_sentences()
                    self._show_legends_sentences()
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
                    self._run_legends_tournament()
                    
                    # Refresh categories after legends tournament
                    categories = self._get_categories_from_sentences()
                    
                    # Recalculate tournament completion
                    total_categories = len(categories["stats"])
                    refined_categories = sum(1 for stats in categories["stats"].values() if stats["is_refined"])
                    overall_completion = refined_categories / total_categories if total_categories > 0 else 0
                    
                    # Re-sort categories
                    sorted_categories = sorted(
                        categories["stats"].items(), 
                        key=lambda x: (x[1]["is_refined"], x[1]["avg_rating"])
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
                        categories = self._get_categories_from_sentences()
                        
                        if not categories["regular"] and not categories["legends"]:
                            print(f"No categories with sentences rated {TOURNAMENT_MIN_RATING}+ found.")
                            self._show_category_status()
                            self._show_perfect_sentences()
                            self._show_legends_sentences()
                            return
                            
                        # Recalculate tournament completion
                        total_categories = len(categories["stats"])
                        refined_categories = sum(1 for stats in categories["stats"].values() if stats["is_refined"])
                        overall_completion = refined_categories / total_categories if total_categories > 0 else 0
                        
                        # Re-sort categories
                        sorted_categories = sorted(
                            categories["stats"].items(), 
                            key=lambda x: (x[1]["is_refined"], x[1]["avg_rating"])
                        )
                else:
                    print("Invalid category number.")
            except ValueError:
                print("Please enter a valid number or command.")
                
        print("\nTournament mode completed!")
        self._show_perfect_sentences()
        self._show_legends_sentences()
    
    def _run_category_tournament(self, category: str, regular_sentences: List[Dict], legends_sentences: List[Dict]) -> None:
        """
        Run a tournament for a specific category.
        
        Args:
            category: The category to run the tournament for
            regular_sentences: List of regular sentences in this category
            legends_sentences: List of legends sentences in this category
        """
        print(f"\nStarting tournament for category: {category}")
        print(f"Number of regular sentences: {len(regular_sentences)}")
        if legends_sentences:
            print(f"Note: {len(legends_sentences)} legend sentences in this category are only available in legends tournament.")
        
        # Only use regular sentences in the tournament
        tournament_sentences = regular_sentences.copy()
        
        # Track pairs we've already compared
        compared_pairs = set()
        
        round_num = 1
        max_attempts = 50  # Maximum attempts to find a new comparison
        
        while tournament_sentences and len(tournament_sentences) >= 2:
            print(f"\nRound {round_num}")
            
            # Create a group of sentences to compare
            comparison_group = []
            attempts = 0
            
            # Try to find sentences we haven't compared yet
            while len(comparison_group) < TOURNAMENT_GROUP_SIZE and attempts < max_attempts:
                if len(comparison_group) == 0:
                    idx = random.randint(0, len(tournament_sentences) - 1)
                    comparison_group.append(tournament_sentences[idx])
                else:
                    # For the second sentence, try to find one we haven't compared with the first
                    found_new_pair = False
                    for _ in range(max_attempts):
                        idx = random.randint(0, len(tournament_sentences) - 1)
                        candidate = tournament_sentences[idx]
                        
                        # Check if we've already compared these two sentences
                        pair_id = self._get_pair_id(comparison_group[0], candidate)
                        if pair_id not in compared_pairs and comparison_group[0] != candidate:
                            comparison_group.append(candidate)
                            found_new_pair = True
                            break
                            
                    if not found_new_pair:
                        # If we couldn't find a new pair, just pick a random one
                        while len(comparison_group) < TOURNAMENT_GROUP_SIZE:
                            idx = random.randint(0, len(tournament_sentences) - 1)
                            candidate = tournament_sentences[idx]
                            if candidate not in comparison_group:
                                comparison_group.append(candidate)
                                break
                
                attempts += 1
            
            # Display sentences for comparison
            for i, sentence in enumerate(comparison_group):
                rating = sentence.get("rating", 0)
                print(f"\n{i+1}. [{rating:.1f}] {sentence['text']}")
            
            # Get user selection
            while True:
                try:
                    selection = input("\nBest sentence (1, 2, 'd' to vote both down, 's' to skip, 'e' to edit, 'q' to quit): ").strip().lower()
                    
                    if selection == 'q':
                        self._save_ratings()
                        print("Ratings saved to", JSON_FILE)
                        print("\nExiting tournament mode.")
                        self._show_perfect_sentences()
                        self._show_legends_sentences()
                        return "quit"  # Signal to completely exit the tournament
                    elif selection == 's':
                        break
                    elif selection == 'd':
                        # Vote both items down
                        for sentence in comparison_group:
                            rating = sentence.get("rating", 0)
                            is_legend = rating >= LEGENDS_MIN_RATING
                            
                            # Legends sentences don't lose points
                            if not is_legend:
                                sentence["rating"] = max(0.0, rating - TOURNAMENT_LOSE_RATING_CHANGE)
                                sentence["last_tournament"] = datetime.now().isoformat()
                    
                        # Save after each comparison
                        self._save_ratings()
                        print("Ratings saved to", JSON_FILE)
                        
                        # Add this pair to compared pairs
                        if len(comparison_group) == 2:
                            pair_id = self._get_pair_id(comparison_group[0], comparison_group[1])
                            compared_pairs.add(pair_id)
                        
                        print("\nBoth sentences rated down!")
                        break
                    elif selection == 'e':
                        # Edit mode
                        edit_selection = input("Which sentence to edit (1 or 2)? ").strip()
                        try:
                            edit_idx = int(edit_selection) - 1
                            if 0 <= edit_idx < len(comparison_group):
                                sentence_to_edit = comparison_group[edit_idx]
                                edited_text = self._edit_sentence_text(sentence_to_edit['text'])
                                if edited_text and edited_text != sentence_to_edit['text']:
                                    # Update the sentence text
                                    sentence_to_edit['text'] = edited_text
                                    sentence_to_edit['edited'] = True
                                    sentence_to_edit['last_edited'] = datetime.now().isoformat()
                                    self._save_ratings()
                                    print("Sentence updated and saved.")
                                    
                                    # Redisplay the sentences after editing
                                    for i, sentence in enumerate(comparison_group):
                                        rating = sentence.get("rating", 0)
                                        print(f"\n{i+1}. [{rating:.1f}] {sentence['text']}")
                                else:
                                    print("No changes made to the sentence.")
                            else:
                                print("Invalid selection. Please enter 1 or 2.")
                        except ValueError:
                            print("Please enter a valid number.")
                        continue  # Continue the loop to get a valid selection
                        
                    selection = int(selection) - 1
                    
                    if 0 <= selection < len(comparison_group):
                        # Update ratings
                        winner = comparison_group[selection]
                        winner_rating = winner.get("rating", 0)
                        winner_is_legend = winner_rating >= LEGENDS_MIN_RATING
                        
                        # Apply different rating changes based on whether this is a legend
                        if winner_is_legend:
                            # Legends get smaller rating increases to avoid inflation
                            new_rating = min(LEGENDS_MAX_RATING, winner_rating + LEGENDS_WIN_RATING_CHANGE)
                        else:
                            # Regular sentences get normal rating increases
                            new_rating = min(TOURNAMENT_MAX_RATING, winner_rating + TOURNAMENT_WIN_RATING_CHANGE)
                            
                            # Check if this sentence reached legend status
                            if new_rating >= LEGENDS_MIN_RATING and winner not in self.legends_sentences:
                                self.legends_sentences.append(winner)
                                print(f"\n🏆 NEW LEGEND! This sentence has reached a rating of {new_rating}!")
                        
                        winner["rating"] = new_rating
                        winner["last_tournament"] = datetime.now().isoformat()
                        
                        # Check if this sentence reached a perfect score
                        if new_rating >= TOURNAMENT_MAX_RATING and winner not in self.perfect_sentences:
                            self.perfect_sentences.append(winner)
                            # Keep only the last 3 perfect sentences
                            if len(self.perfect_sentences) > 3:
                                self.perfect_sentences.pop(0)
                        
                        # Decrease rating for others
                        for i, sentence in enumerate(comparison_group):
                            if i != selection:
                                loser_rating = sentence.get("rating", 0)
                                loser_is_legend = loser_rating >= LEGENDS_MIN_RATING
                                
                                if loser_is_legend:
                                    # Legends don't lose points
                                    new_loser_rating = loser_rating
                                else:
                                    # Regular sentences lose points normally
                                    new_loser_rating = max(0.0, loser_rating - TOURNAMENT_LOSE_RATING_CHANGE)
                                    
                                sentence["rating"] = new_loser_rating
                                sentence["last_tournament"] = datetime.now().isoformat()
                                
                        # Save after each comparison
                        self._save_ratings()
                        print("Ratings saved to", JSON_FILE)
                        
                        # Add this pair to compared pairs
                        if len(comparison_group) == 2:
                            pair_id = self._get_pair_id(comparison_group[0], comparison_group[1])
                            compared_pairs.add(pair_id)
                        
                        # Show the winner
                        print(f"\nRatings updated! Winner: {winner['text'][:100]}{'...' if len(winner['text']) > 100 else ''}")
                        break
                    else:
                        print("Invalid selection. Please enter 1, 2, 'd', 's', 'e', or 'q'.")
                except ValueError:
                    print("Please enter a valid number or command.")
            
            # Remove sentences that fall below the threshold or reach perfect score
            tournament_sentences = [s for s in tournament_sentences if 
                                   (s.get("rating", 0) >= TOURNAMENT_MIN_RATING and 
                                    s.get("rating", 0) < LEGENDS_MIN_RATING)]
            
            # Check if we should continue
            if round_num % 5 == 0 or len(tournament_sentences) < 2:
                # Offer three options
                print("\nWhat would you like to do next?")
                print("'c' to continue in this category")
                print("'r' to return to the category picker")
                print("'q' to quit")
                
                while True:
                    next_action = input("Enter your choice: ").strip().lower()
                    if next_action == 'c':
                        if len(tournament_sentences) < 2:
                            print("Not enough sentences left in this category. Returning to category menu.")
                            return "menu"
                        break
                    elif next_action == 'r':
                        return "menu"
                    elif next_action == 'q':
                        self._save_ratings()
                        print("Ratings saved to", JSON_FILE)
                        self._show_perfect_sentences()
                        self._show_legends_sentences()
                        return "quit"
                    else:
                        print("Invalid choice. Please enter 'c', 'r', or 'q'.")
                    
            round_num += 1
            
        if len(tournament_sentences) < 2:
            print(f"\nNot enough sentences left in {category} to continue tournament.")
        else:
            print(f"\n{category} tournament completed!")
            
        return "menu"  # Signal to return to category menu
        
    def _get_pair_id(self, sentence1: Dict, sentence2: Dict) -> str:
        """
        Generate a unique ID for a pair of sentences to track compared pairs.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            
        Returns:
            A unique string ID for this pair
        """
        # Use the text as the identifier, sorted to ensure the same pair always gets the same ID
        id1 = sentence1.get('text', '')
        id2 = sentence2.get('text', '')
        
        # Sort to ensure (A,B) and (B,A) generate the same ID
        if id1 > id2:
            id1, id2 = id2, id1
            
        return f"{id1}|{id2}"

    def _edit_sentence_text(self, text: str) -> str:
        """
        Allow the user to edit a sentence.
        
        Args:
            text: The original sentence text
            
        Returns:
            The edited sentence text, or None if canceled
        """
        print("\n" + "=" * 40)
        print("EDIT MODE")
        print("=" * 40)
        
        print("Original text:")
        print(text)
        new_text = input("New version (leave empty to cancel): ").strip()
        
        if not new_text:
            print("Edit canceled.")
            return None
            
        return new_text
    
    def _show_perfect_sentences(self):
        """Show the last three sentences that reached a perfect score."""
        if self.perfect_sentences:
            print("\n" + "=" * 40)
            print("PERFECT SCORE SENTENCES")
            print("=" * 40)
            for i, sentence in enumerate(reversed(self.perfect_sentences), 1):
                print(f"{i}. {sentence['text']}")
            print("=" * 40)
    
    def _show_legends_sentences(self):
        """Show the sentences that have reached legend status (rating >= 10)."""
        if self.legends_sentences:
            print("\n" + "=" * 40)
            print("LEGENDS SENTENCES")
            print("=" * 40)
            
            # Sort by rating (highest first)
            sorted_legends = sorted(self.legends_sentences, key=lambda s: s.get("rating", 0), reverse=True)
            
            for i, sentence in enumerate(sorted_legends, 1):
                rating = sentence.get("rating", 0)
                print(f"{i}. [{rating:.1f}/12] {sentence['text']}")
            
            print("=" * 40)
            print(f"Total legends: {len(self.legends_sentences)}")
            print("=" * 40)

    def _run_legends_tournament(self) -> None:
        """Run a tournament specifically for sentences that have reached legend status (rating >= 10)."""
        print("\n" + "=" * 40)
        print("LEGENDS TOURNAMENT MODE")
        print("=" * 40)
        print(f"Compare your best sentences (rated {LEGENDS_MIN_RATING}+)")
        print(f"Winner gets +{LEGENDS_WIN_RATING_CHANGE} points (up to max {LEGENDS_MAX_RATING})")
        print(f"Losers don't lose any points in legends tournament")
        print("=" * 40)
        
        # Get categories and sentences
        categories = self._get_categories_from_sentences()
        
        # Filter to only include categories with legend sentences
        legends_categories = {}
        for category, sentences in categories["legends"].items():
            if sentences:
                legends_categories[category] = sentences
        
        if not legends_categories:
            print(f"No legends found. Sentences must reach a rating of {LEGENDS_MIN_RATING}+ to qualify.")
            print(f"Continue using regular tournament mode to promote sentences to legend status.")
            return
            
        # Calculate overall legends stats
        total_legends = len(self.legends_sentences)
        max_rated = max(self.legends_sentences, key=lambda s: s.get("rating", 0))
        max_rating = max_rated.get("rating", 0)
        
        # Sort categories by number of legend sentences (descending)
        sorted_categories = sorted(
            legends_categories.items(), 
            key=lambda x: len(x[1]),
            reverse=True
        )
            
        while True:
            # Show legends stats
            print(f"\nLegends stats: {total_legends} total legends, highest rating: {max_rating:.1f}/12")
            
            # Show available categories
            print(f"\nAvailable legend categories:")
            print(f"{'#':<4} {'Category':<25} {'Legends':<7} {'Avg Rating':<12}")
            print("-" * 70)
            
            for i, (category, sentences) in enumerate(sorted_categories, 1):
                # Truncate long category names
                category_display = category
                if len(category) > 22:
                    category_display = category[:19] + "..."
                
                avg_rating = sum(s.get("rating", 0) for s in sentences) / len(sentences)
                print(f"{i:<4} {category_display:<25} {len(sentences):<7} {avg_rating:.1f}/12")
                
            # Let user select a category
            try:
                category_input = input("\nSelect category number, 's' for status overview, 'r' for regular tournament, or 'q' to quit: ").strip().lower()
                
                if category_input == 'q':
                    # Show comprehensive status before exiting
                    self._show_category_status()
                    self._show_legends_sentences()
                    return
                elif category_input == 's':
                    # Show comprehensive category status
                    self._show_category_status()
                    input("\nPress Enter to continue...")
                    continue
                elif category_input == 'r':
                    # Switch to regular tournament
                    self._run_tournament_mode()
                    
                    # Refresh categories after regular tournament
                    categories = self._get_categories_from_sentences()
                    
                    # Filter to only include categories with legend sentences
                    legends_categories = {}
                    for category, sentences in categories["legends"].items():
                        if sentences:
                            legends_categories[category] = sentences
                    
                    if not legends_categories:
                        print(f"No legends found after tournament.")
                        return
                        
                    # Recalculate legends stats
                    total_legends = len(self.legends_sentences)
                    if total_legends > 0:
                        max_rated = max(self.legends_sentences, key=lambda s: s.get("rating", 0))
                        max_rating = max_rated.get("rating", 0)
                    else:
                        max_rating = 0
                    
                    # Re-sort categories
                    sorted_categories = sorted(
                        legends_categories.items(), 
                        key=lambda x: len(x[1]),
                        reverse=True
                    )
                    continue
                    
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
                        categories = self._get_categories_from_sentences()
                        
                        # Filter to only include categories with legend sentences
                        legends_categories = {}
                        for category, sentences in categories["legends"].items():
                            if sentences:
                                legends_categories[category] = sentences
                        
                        if not legends_categories:
                            print(f"No more legends found.")
                            # Show comprehensive status before exiting
                            self._show_category_status()
                            self._show_legends_sentences()
                            return
                            
                        # Recalculate legends stats
                        total_legends = len(self.legends_sentences)
                        max_rated = max(self.legends_sentences, key=lambda s: s.get("rating", 0))
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
{{ ... }}
