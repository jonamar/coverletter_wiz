#!/usr/bin/env python3
"""
Sentence Rater - A CLI tool to compare and rate unique sentences from cover letters.

This tool extracts unique sentences from the processed_cover_letters.json file,
allows for batch rating of sentences, and saves the ratings back to the file.
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

# Constants
JSON_FILE = "processed_cover_letters.json"
BATCH_RATING_SCALE = 10   # 1-10 scale for batch ratings
FILTER_THRESHOLD = 2      # Ratings <= this value are filtered out
BATCH_SIZE = 10           # Number of sentences to show in each batch
REFINEMENT_THRESHOLD = 1  # Maximum rating difference to consider sentences for refinement
HIGH_RATING_THRESHOLD = 7 # Minimum rating to be considered high quality

class SentenceRater:
    """
    A class for rating sentences from cover letters using a batch rating approach.
    
    This class extracts unique sentences from the processed_cover_letters.json file,
    allows for batch rating of sentences, and saves the ratings back to the file.
    """
    
    def __init__(self, json_file: str):
        """
        Initialize the SentenceRater with a JSON file.
        
        Args:
            json_file: Path to the processed_cover_letters.json file
        """
        self.json_file = json_file
        self.data = self._load_data()
        self.unique_sentences = self._extract_unique_sentences()
        self.batch_ratings_done = self.data.get("batch_ratings_done", False)
        self.refinement_done = self.data.get("refinement_done", False)
        self.filtered_sentences = self.data.get("filtered_sentences", set())
        self.last_processed_date = self.data.get("last_processed_date", "")
        
        # Convert filtered_sentences from list to set if it exists as a list
        if isinstance(self.filtered_sentences, list):
            self.filtered_sentences = set(self.filtered_sentences)
        
        # Clean up old ELO-related fields
        self._cleanup_elo_fields()
    
    def _load_data(self) -> Dict:
        """
        Load data from the JSON file.
        
        Returns:
            Dict: The JSON data
        """
        try:
            with open(self.json_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {self.json_file} not found.")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: {self.json_file} is not a valid JSON file.")
            exit(1)
    
    def _cleanup_elo_fields(self) -> None:
        """Remove ELO-specific fields from the data."""
        elo_fields = ["matches", "wins", "losses", "comparisons_done", "tier_stability", "initial_ratings_done"]
        
        # Remove top-level fields
        for field in elo_fields:
            if field in self.data:
                del self.data[field]
        
        # Remove fields from sentence objects
        for filename, file_data in self.data.items():
            # Skip non-file entries like metadata
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
            
            paragraphs = file_data["content"].get("paragraphs", [])
            
            for paragraph in paragraphs:
                sentences = paragraph.get("sentences", [])
                
                for sentence in sentences:
                    for field in ["matches", "wins", "losses"]:
                        if field in sentence:
                            del sentence[field]
                    
                    # Convert any existing rating to 1-10 scale if it's in ELO range
                    if "rating" in sentence and sentence["rating"] > 100:  # Likely an ELO rating
                        # Map from ELO range (typically 1000-1400) to 1-10 scale
                        # This is an approximation
                        elo = sentence["rating"]
                        if elo < 1100:
                            new_rating = 1
                        elif elo > 1400:
                            new_rating = 10
                        else:
                            # Linear mapping from 1100-1400 to 1-10
                            new_rating = 1 + (elo - 1100) * 9 / 300
                        
                        sentence["rating"] = round(new_rating, 1)
    
    def _extract_unique_sentences(self) -> Dict[str, List[Tuple]]:
        """
        Extract all unique sentences from the data.
        
        Returns:
            Dict: A dictionary mapping sentence text to a list of tuples (file, paragraph_index, sentence_index)
                  that point to all instances of that sentence in the data
        """
        unique_sentences = {}
        
        for filename, file_data in self.data.items():
            # Skip non-file entries like metadata
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
            
            paragraphs = file_data["content"].get("paragraphs", [])
            
            for p_idx, paragraph in enumerate(paragraphs):
                sentences = paragraph.get("sentences", [])
                
                for s_idx, sentence in enumerate(sentences):
                    text = sentence.get("text", "").strip()
                    
                    if not text:
                        continue
                    
                    # If this sentence text already exists and has a rating, keep track of it
                    if text in unique_sentences:
                        # Check if the existing sentence has a rating
                        existing_loc = unique_sentences[text][0]
                        existing_sentence = self._get_sentence_by_location(existing_loc)
                        new_sentence = sentence
                        
                        # If new sentence has a rating but existing doesn't, replace it
                        if ("rating" in new_sentence and 
                            ("rating" not in existing_sentence or existing_sentence["rating"] == 0)):
                            unique_sentences[text] = []
                    
                    if text not in unique_sentences:
                        unique_sentences[text] = []
                    
                    unique_sentences[text].append((filename, p_idx, s_idx))
        
        return unique_sentences
    
    def _get_sentence_by_location(self, location: Tuple[str, int, int]) -> Dict:
        """
        Get a sentence object by its location.
        
        Args:
            location: A tuple (filename, paragraph_index, sentence_index)
        
        Returns:
            Dict: The sentence object
        """
        filename, p_idx, s_idx = location
        return self.data[filename]["content"]["paragraphs"][p_idx]["sentences"][s_idx]
    
    def _save_data(self) -> None:
        """Save the updated data back to the JSON file."""
        # Update metadata in the data
        self.data["batch_ratings_done"] = self.batch_ratings_done
        self.data["refinement_done"] = self.refinement_done
        self.data["filtered_sentences"] = list(self.filtered_sentences)
        
        # Update last processed date
        from datetime import datetime
        self.data["last_processed_date"] = datetime.now().isoformat()
        self.last_processed_date = self.data["last_processed_date"]
        
        with open(self.json_file, "w") as f:
            json.dump(self.data, f, indent=4)
        
        print(f"Data saved to {self.json_file}")
    
    def _get_sentence_rating(self, text: str) -> float:
        """
        Get the rating of a sentence.
        
        Args:
            text: The text of the sentence
            
        Returns:
            float: The rating of the sentence
        """
        locations = self.unique_sentences[text]
        sentence = self._get_sentence_by_location(locations[0])
        return sentence.get("rating", 0)
    
    def _get_all_sentence_ratings(self) -> List[Tuple[str, float]]:
        """
        Get all sentences with their ratings.
        
        Returns:
            List: List of tuples (sentence_text, rating)
        """
        sentence_ratings = []
        
        for text, locations in self.unique_sentences.items():
            # Skip filtered sentences
            if text in self.filtered_sentences:
                continue
                
            sentence = self._get_sentence_by_location(locations[0])
            rating = sentence.get("rating", 0)
            sentence_ratings.append((text, rating))
        
        return sentence_ratings
    
    def _get_top_rated_sentences(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get the top N rated sentences.
        
        Args:
            n: Number of top sentences to return
        
        Returns:
            List: List of tuples (sentence_text, rating)
        """
        sentence_ratings = self._get_all_sentence_ratings()
        
        # Sort by rating in descending order
        sentence_ratings.sort(key=lambda x: x[1], reverse=True)
        
        return sentence_ratings[:n]
    
    def _set_batch_rating(self, text: str, rating: float) -> None:
        """
        Set the batch rating for a sentence.
        
        Args:
            text: The text of the sentence
            rating: The batch rating (1-10)
        """
        # Update all instances of the sentence
        for location in self.unique_sentences[text]:
            sentence = self._get_sentence_by_location(location)
            sentence["rating"] = rating
            sentence["batch_rating"] = True
            
            # Add timestamp for when this rating was made
            from datetime import datetime
            sentence["last_rated"] = datetime.now().isoformat()
        
        # If rating is below threshold, add to filtered sentences
        if rating <= FILTER_THRESHOLD:
            self.filtered_sentences.add(text)
    
    def _get_unrated_sentences(self) -> List[str]:
        """
        Get all sentences that haven't been rated yet.
        
        Returns:
            List: List of sentence texts
        """
        unrated = []
        
        for text, locations in self.unique_sentences.items():
            sentence = self._get_sentence_by_location(locations[0])
            
            # If sentence has no rating or rating is 0, it's unrated
            if "rating" not in sentence or sentence["rating"] == 0:
                unrated.append(text)
            # If rating is below threshold, add to filtered
            elif sentence["rating"] <= FILTER_THRESHOLD:
                self.filtered_sentences.add(text)
        
        return unrated
    
    def _get_sentences_for_batch(self, batch_size: int) -> List[str]:
        """
        Get a batch of sentences for rating.
        
        Args:
            batch_size: Number of sentences to include in the batch
            
        Returns:
            List: List of sentence texts
        """
        # Get all unrated sentences
        unrated_sentences = self._get_unrated_sentences()
        
        # If we have fewer sentences than the batch size, return all of them
        if len(unrated_sentences) <= batch_size:
            return unrated_sentences
        
        # Otherwise, return a random batch
        return random.sample(unrated_sentences, batch_size)
    
    def _get_sentences_for_refinement(self) -> List[List[str]]:
        """
        Get groups of sentences that need refinement (similar ratings).
        
        Returns:
            List: List of lists, where each inner list contains sentences with similar ratings
        """
        # Get all sentences with their ratings
        sentence_ratings = self._get_all_sentence_ratings()
        
        # Group sentences by their rounded rating
        rating_groups = defaultdict(list)
        for text, rating in sentence_ratings:
            # Only include high-rated sentences
            if rating >= HIGH_RATING_THRESHOLD:
                # Round to nearest 0.5
                rounded_rating = round(rating * 2) / 2
                rating_groups[rounded_rating].append(text)
        
        # Include all groups, even those with just one sentence
        refinement_groups = list(rating_groups.values())
        
        # Sort groups by rating (highest first)
        refinement_groups.sort(key=lambda group: self._get_sentence_rating(group[0]), reverse=True)
        
        return refinement_groups
    
    def _run_batch_rating(self) -> None:
        """Run the batch rating phase."""
        # Get all unrated sentences
        unrated_sentences = self._get_unrated_sentences()
        
        if not unrated_sentences:
            print("\nNo new sentences to rate.")
            self.batch_ratings_done = True
            self._save_data()
            return
        
        print("\n" + "=" * 40)
        print("BATCH RATING PHASE")
        print("=" * 40)
        print("Please rate each sentence on a scale of 1-10:")
        print("1-2 = Poor (will be filtered out)")
        print("3-5 = Fair to Average")
        print("6-7 = Good")
        print("8-10 = Excellent")
        
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
            
            for sentence_idx, text in enumerate(batch):
                print(f"\n{'-' * 40}")
                print(f"{sentence_idx + 1}. {text}")
                
                while True:
                    try:
                        rating_input = input("Rating (1-10, 's' to skip, 'q' to quit): ").lower().strip()
                        
                        if rating_input == 'q':
                            self._save_data()
                            print("Rating process saved and exited.")
                            exit(0)
                        elif rating_input == 's':
                            break
                        
                        rating = float(rating_input)
                        if 1 <= rating <= 10:
                            self._set_batch_rating(text, rating)
                            batch_ratings.append(rating)
                            break
                        else:
                            print("Please enter a number between 1 and 10.")
                    except ValueError:
                        print("Please enter a valid number or 's' to skip.")
            
            # Save after each batch
            self._save_data()
            
            # Show batch summary
            if batch_ratings:
                avg_rating = sum(batch_ratings) / len(batch_ratings)
                print(f"\n{'-' * 40}")
                print(f"Batch {batch_idx + 1} complete!")
                print(f"Average rating: {avg_rating:.1f}")
                print(f"Rated {len(batch_ratings)} out of {len(batch)} sentences in this batch.")
                
                if batch_idx < len(batches) - 1:
                    input("Press Enter to continue to the next batch...")
        
        self.batch_ratings_done = True
        self._save_data()
        
        print("\nBatch rating phase completed!")
        
        # Show top rated sentences
        top_sentences = self._get_top_rated_sentences(5)
        print("\nTOP RATED SENTENCES:")
        for i, (text, rating) in enumerate(top_sentences, 1):
            print(f"{i}. [{rating:.1f}/10] {text}")
    
    def _run_refinement(self) -> None:
        """Run the refinement phase for similarly rated sentences."""
        refinement_groups = self._get_sentences_for_refinement()
        
        if not refinement_groups:
            print("\nNo sentence groups need refinement.")
            self.refinement_done = True
            self._save_data()
            return
        
        print("\n" + "=" * 40)
        print("REFINEMENT PHASE")
        print("=" * 40)
        print("Please compare these similarly rated sentences to refine their rankings.")
        
        for group_idx, group in enumerate(refinement_groups):
            # Get the approximate rating of this group
            group_rating = round(self._get_sentence_rating(group[0]))
            
            print(f"\n{'=' * 40}")
            print(f"GROUP {group_idx + 1} OF {len(refinement_groups)} (Around {group_rating}/10)")
            print(f"{'=' * 40}")
            print("Rate these similar sentences more precisely (1-10, decimals allowed):")
            
            for sentence_idx, text in enumerate(group):
                current_rating = self._get_sentence_rating(text)
                print(f"\n{'-' * 40}")
                print(f"{sentence_idx + 1}. Current rating: {current_rating:.1f}")
                print(f"{text}")
                
                while True:
                    try:
                        rating_input = input("New rating (1-10, 's' to skip, 'q' to quit): ").lower().strip()
                        
                        if rating_input == 'q':
                            self._save_data()
                            print("Rating process saved and exited.")
                            exit(0)
                        elif rating_input == 's':
                            break
                        
                        rating = float(rating_input)
                        if 1 <= rating <= 10:
                            self._set_batch_rating(text, rating)
                            break
                        else:
                            print("Please enter a number between 1 and 10.")
                    except ValueError:
                        print("Please enter a valid number or 's' to skip.")
            
            # Save after each group
            self._save_data()
            
            if group_idx < len(refinement_groups) - 1:
                continue_input = input("\nPress Enter to continue to the next group, or 'q' to quit: ").lower().strip()
                if continue_input == 'q':
                    break
        
        self.refinement_done = True
        self._save_data()
        
        print("\nRefinement phase completed!")
        
        # Show top rated sentences
        top_sentences = self._get_top_rated_sentences(5)
        print("\nFINAL TOP RATED SENTENCES:")
        for i, (text, rating) in enumerate(top_sentences, 1):
            print(f"{i}. [{rating:.1f}/10] {text}")
    
    def _show_stats(self) -> None:
        """Show statistics about the current state of the ratings."""
        filtered_count = len(self.filtered_sentences)
        total_count = len(self.unique_sentences)
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
        
        if self.last_processed_date:
            print(f"Last processed: {self.last_processed_date}")
        
        # Show rating distribution
        ratings = [self._get_sentence_rating(text) for text in self.unique_sentences.keys() 
                  if text not in self.filtered_sentences and self._get_sentence_rating(text) > 0]
        
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
        top_sentences = self._get_top_rated_sentences(5)
        print("\nTOP RATED SENTENCES:")
        for i, (text, rating) in enumerate(top_sentences, 1):
            print(f"{i}. [{rating:.1f}/10] {text}")
        
        print("=" * 40)
    
    def run(self) -> None:
        """Run the sentence rating process."""
        print(f"Found {len(self.unique_sentences)} unique sentences.")
        
        # Check for unrated sentences
        unrated_sentences = self._get_unrated_sentences()
        if unrated_sentences:
            print(f"Found {len(unrated_sentences)} unrated sentences.")
        
        # Show current status
        if self.refinement_done and not unrated_sentences:
            print("All sentences have been rated and refined.")
            self._show_stats()
            
            restart = input("Would you like to restart the rating process? (y/n): ").lower().strip()
            if restart == 'y':
                self.batch_ratings_done = False
                self.refinement_done = False
                self.filtered_sentences = set()
                self._save_data()
            else:
                print("Showing final statistics:")
                self._show_stats()
                return
        
        # Run batch rating phase if not done yet or if there are new unrated sentences
        if not self.batch_ratings_done or unrated_sentences:
            self.batch_ratings_done = False  # Reset if there are new sentences
            self._run_batch_rating()
        
        # Run refinement phase if not done yet
        if not self.refinement_done and self.batch_ratings_done:
            self._run_refinement()
        
        # Show final stats
        self._show_stats()

def main():
    """Main function to run the sentence rater CLI tool."""
    parser = argparse.ArgumentParser(description="Rate sentences from cover letters")
    parser.add_argument("--file", default=JSON_FILE, help=f"Path to the JSON file (default: {JSON_FILE})")
    parser.add_argument("--force-refinement", action="store_true", help="Force the refinement phase to run even if it was previously completed")
    args = parser.parse_args()
    
    rater = SentenceRater(args.file)
    
    # If force-refinement flag is set, reset refinement_done flag
    if args.force_refinement:
        rater.refinement_done = False
        print("Forcing refinement phase to run...")
    
    rater.run()

if __name__ == "__main__":
    main()
