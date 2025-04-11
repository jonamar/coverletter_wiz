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
    
    def run(self) -> None:
        """Run the sentence rating process."""
        print(f"Found {len(self.sentences)} unique sentences.")
        
        # Check for unrated sentences
        unrated_sentences = self._get_unrated_sentences()
        if unrated_sentences:
            print(f"Found {len(unrated_sentences)} unrated sentences.")
        
        # Show current status
        if not unrated_sentences:
            print("All sentences have been rated.")
            self._show_stats()
            
            # After rating is complete, offer tournament mode
            tournament_mode = input("Would you like to enter tournament mode to compare similar sentences? (y/n): ").lower().strip()
            if tournament_mode == 'y':
                self._run_tournament_mode()
            else:
                restart = input("Would you like to restart the rating process? (y/n): ").lower().strip()
                if restart == 'y':
                    for sentence in self.sentences:
                        sentence["rating"] = 0
                        sentence["batch_rating"] = False
                    self._save_ratings()
                else:
                    print("Showing final statistics:")
                    self._show_stats()
                    return
        
        # Run batch rating phase
        self._run_batch_rating()
        
        # After batch rating is complete, offer tournament mode
        tournament_mode = input("Would you like to enter tournament mode to compare similar sentences? (y/n): ").lower().strip()
        if tournament_mode == 'y':
            self._run_tournament_mode()
        
        # Show final stats
        self._show_stats()
        
    def _get_categories_from_sentences(self) -> Dict[str, List[Dict]]:
        """
        Get a dictionary of categories mapped to sentences that have that tag.
        Only includes sentences with ratings above TOURNAMENT_MIN_RATING.
        
        Returns:
            Dictionary mapping category names to lists of sentences
        """
        categories = {}
        
        for sentence in self.sentences:
            # Skip sentences with ratings below the threshold
            if sentence.get("rating", 0) < TOURNAMENT_MIN_RATING:
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
                if tag not in categories:
                    categories[tag] = []
                categories[tag].append(sentence)
        
        return categories
    
    def _run_tournament_mode(self) -> None:
        """Run the Apples to Apples tournament mode to compare similar sentences."""
        print("\n" + "=" * 40)
        print("APPLES TO APPLES TOURNAMENT MODE")
        print("=" * 40)
        print(f"Compare sentences rated {TOURNAMENT_MIN_RATING}+ to refine their ratings.")
        print(f"Winner gets +{TOURNAMENT_WIN_RATING_CHANGE} points, loser loses {TOURNAMENT_LOSE_RATING_CHANGE} points.")
        print(f"Sentences dropping below {TOURNAMENT_MIN_RATING} or reaching {TOURNAMENT_MAX_RATING} are removed from the tournament.")
        print("=" * 40)
        
        # Get categories and sentences
        categories = self._get_categories_from_sentences()
        
        if not categories:
            print(f"No categories with sentences rated {TOURNAMENT_MIN_RATING}+ found.")
            return
            
        # Calculate average rating for each category and sort by number of sentences
        category_stats = []
        for category, sentences in categories.items():
            avg_rating = sum(s.get("rating", 0) for s in sentences) / len(sentences)
            category_stats.append((category, len(sentences), avg_rating))
        
        # Sort by number of sentences (descending)
        category_stats.sort(key=lambda x: x[1], reverse=True)
            
        while True:
            # Show available categories
            print(f"\nAvailable categories (showing only sentences rated {TOURNAMENT_MIN_RATING}+ and below {TOURNAMENT_MAX_RATING}):")
            for i, (category, count, avg_rating) in enumerate(category_stats, 1):
                print(f"{i}. {category} ({count} sentences, avg: {avg_rating:.1f})")
                
            # Let user select a category
            try:
                category_idx = input("\nSelect category number (or 'q' to quit): ").strip().lower()
                
                if category_idx == 'q':
                    self._show_perfect_sentences()
                    return
                    
                category_idx = int(category_idx) - 1
                
                if 0 <= category_idx < len(category_stats):
                    selected_category = category_stats[category_idx][0]
                    result = self._run_category_tournament(selected_category, categories[selected_category])
                    
                    # If user quit from the tournament, exit completely
                    if result == "quit":
                        return
                    
                    # If returning to menu, update categories
                    if result == "menu":
                        # Recalculate categories
                        categories = self._get_categories_from_sentences()
                        
                        if not categories:
                            print(f"No categories with sentences rated {TOURNAMENT_MIN_RATING}+ and below {TOURNAMENT_MAX_RATING} found.")
                            self._show_perfect_sentences()
                            return
                            
                        # Recalculate category stats
                        category_stats = []
                        for category, sentences in categories.items():
                            avg_rating = sum(s.get("rating", 0) for s in sentences) / len(sentences)
                            category_stats.append((category, len(sentences), avg_rating))
                        
                        # Sort by number of sentences (descending)
                        category_stats.sort(key=lambda x: x[1], reverse=True)
                else:
                    print("Invalid category number.")
            except ValueError:
                print("Please enter a valid number or 'q'.")
                
        print("\nTournament mode completed!")
        self._show_perfect_sentences()
    
    def _run_category_tournament(self, category: str, sentences: List[Dict]) -> None:
        """
        Run a tournament for a specific category.
        
        Args:
            category: The category to run the tournament for
            sentences: List of sentences in this category
        """
        print(f"\nStarting tournament for category: {category}")
        print(f"Number of sentences: {len(sentences)}")
        
        # Make a copy of the sentences to avoid modifying the original list
        tournament_sentences = sentences.copy()
        
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
                print(f"\n{i+1}. [{sentence.get('rating', 0):.1f}] {sentence['text']}")
            
            # Get user selection
            while True:
                try:
                    selection = input("\nBest sentence (1, 2, 's' to skip, 'e' to edit, 'q' to quit): ").strip().lower()
                    
                    if selection == 'q':
                        self._save_ratings()
                        print("Ratings saved to", JSON_FILE)
                        print("\nExiting tournament mode.")
                        self._show_perfect_sentences()
                        return "quit"  # Signal to completely exit the tournament
                    elif selection == 's':
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
                                        print(f"\n{i+1}. [{sentence.get('rating', 0):.1f}] {sentence['text']}")
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
                        new_rating = min(TOURNAMENT_MAX_RATING, winner.get("rating", 0) + TOURNAMENT_WIN_RATING_CHANGE)
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
                                sentence["rating"] = max(0.0, sentence.get("rating", 0) - TOURNAMENT_LOSE_RATING_CHANGE)
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
                        print("Invalid selection. Please enter 1, 2, 's', 'e', or 'q'.")
                except ValueError:
                    print("Please enter a valid number or command.")
            
            # Remove sentences that fall below the threshold or reach perfect score
            tournament_sentences = [s for s in tournament_sentences if 
                                   (s.get("rating", 0) >= TOURNAMENT_MIN_RATING and 
                                    s.get("rating", 0) < TOURNAMENT_MAX_RATING)]
            
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


def main():
    """Main function to run the sentence rater CLI tool."""
    parser = argparse.ArgumentParser(description="Rate sentences from cover letters")
    parser.add_argument("--file", default=JSON_FILE, help=f"Path to the JSON file (default: {JSON_FILE})")
    parser.add_argument("--tournament", action="store_true", help="Start directly in tournament mode")
    args = parser.parse_args()
    
    rater = SentenceRater(args.file)
    
    if args.tournament:
        # Skip batch rating if tournament flag is set
        if rater._get_unrated_sentences():
            print("Warning: There are unrated sentences. It's recommended to rate them first.")
            proceed = input("Do you want to proceed directly to tournament mode? (y/n): ").lower().strip()
            if proceed != 'y':
                rater.run()
                return
        
        rater._run_tournament_mode()
        rater._show_stats()
    else:
        rater.run()

if __name__ == "__main__":
    main()
