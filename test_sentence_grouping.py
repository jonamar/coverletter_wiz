#!/usr/bin/env python3
"""
Test Sentence Grouping Parameters

This script tests different configurations of sentence grouping parameters
on a cover letter to help find the optimal settings.
"""

import json
import os
from pathlib import Path
import spacy
import yaml
from typing import List, Dict, Any
import random
from datetime import datetime

# Import the functions we need to test
from spacy_utils import nlp

# Define a copy of the discourse markers for testing
DEFAULT_DISCOURSE_MARKERS = {
    "additionally", "furthermore", "moreover", "however", "nevertheless", 
    "therefore", "thus", "consequently", "as a result", "in conclusion", 
    "finally", "for example", "for instance", "specifically", "in particular",
    "in contrast", "on the other hand", "similarly", "likewise",
    "this", "these", "that", "those", "it", "they", "their", "them"
}

def load_cover_letter(file_path: str) -> str:
    """Load a cover letter from a text file."""
    with open(file_path, 'r') as f:
        return f.read()

def load_categories(yaml_file: str) -> Dict[str, List[str]]:
    """Load categories from a YAML file."""
    if not os.path.exists(yaml_file):
        print(f"Error: Categories file '{yaml_file}' not found.")
        return {}
    
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)

def test_sentence_grouping(text: str, config: Dict[str, Any]) -> List[Dict]:
    """
    Test sentence grouping with different parameters.
    
    Args:
        text: The text to process
        config: Configuration parameters for sentence grouping
    
    Returns:
        List of sentence groups
    """
    # Use the discourse markers from the config or the default ones
    discourse_markers = set(config.get("discourse_markers", DEFAULT_DISCOURSE_MARKERS))
    
    # Process the text with the current configuration
    paragraphs = text.split('\n\n')
    all_groups = []
    
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        # Call identify_sentence_groups with the specified parameters
        max_group_size = config.get("max_group_size", 3)
        similarity_threshold = config.get("similarity_threshold", 0.85)
        
        # We need to modify the identify_sentence_groups function to accept similarity_threshold
        # For now, we'll create a modified version here
        doc = nlp(paragraph)
        sentences = list(doc.sents)
        
        if len(sentences) <= 1:
            # Return the single sentence as its own group
            group = {
                "text": paragraph.strip(),
                "sentences": [{"text": paragraph.strip(), "index": 0}],
                "is_sentence_group": False,
                "component_sentences": []
            }
            all_groups.append(group)
            continue
        
        # Create initial groups (start with each sentence as its own group)
        groups = [{
            "index": i,
            "text": sent.text.strip(),
            "sentences": [{"text": sent.text.strip(), "index": i}],
            "connections": set(),
            "is_sentence_group": False
        } for i, sent in enumerate(sentences)]
        
        # Identify connections between sentences
        for i in range(len(sentences) - 1):
            current = sentences[i]
            next_sent = sentences[i + 1]
            
            # Check for discourse markers at the beginning of the next sentence
            next_starts_with_marker = False
            next_text_lower = next_sent.text.lower().strip()
            
            for marker in discourse_markers:
                if next_text_lower.startswith(marker):
                    next_starts_with_marker = True
                    groups[i]["connections"].add(i + 1)
                    groups[i + 1]["connections"].add(i)
                    break
            
            # Check for coreference (simple version - checking for shared entities)
            if config.get("use_entity_matching", True):
                current_ents = {e.text.lower() for e in current.ents}
                next_ents = {e.text.lower() for e in next_sent.ents}
                if current_ents and next_ents and current_ents.intersection(next_ents):
                    groups[i]["connections"].add(i + 1)
                    groups[i + 1]["connections"].add(i)
            
            # Check for pronoun references (simplistic approach)
            if config.get("use_pronoun_detection", True):
                if any(token.lower_ in ["it", "this", "these", "they", "their", "them"] for token in next_sent[:3]):
                    groups[i]["connections"].add(i + 1)
                    groups[i + 1]["connections"].add(i)
            
            # Check for semantic similarity between sentences
            if config.get("use_similarity", True):
                # Only calculate similarity if both sentences have vectors
                if current.has_vector and next_sent.has_vector:
                    sim_score = current.similarity(next_sent)
                    if sim_score > similarity_threshold:
                        groups[i]["connections"].add(i + 1)
                        groups[i + 1]["connections"].add(i)
        
        # Merge connected sentences into groups, but limit group size
        merged_groups = []
        processed = set()
        
        for i, group in enumerate(groups):
            if i in processed:
                continue
            
            if not group["connections"]:
                # No connections, keep as a single sentence
                group_dict = {
                    "text": group["text"],
                    "sentences": group["sentences"],
                    "is_sentence_group": False,
                    "component_sentences": []
                }
                merged_groups.append(group_dict)
                processed.add(i)
            else:
                # Start a new connected group with size limit
                connected = {i}
                processed.add(i)
                
                # Only consider direct connections (no transitive connections)
                # and limit to max_group_size
                direct_connections = sorted(group["connections"] - processed)
                
                # Take only enough connections to reach max_group_size
                for next_idx in direct_connections:
                    if len(connected) >= max_group_size:
                        break
                    connected.add(next_idx)
                    processed.add(next_idx)
                
                # Sort the indices to maintain order
                connected = sorted(connected)
                
                # Create the merged group
                if len(connected) > 1:
                    merged_text = " ".join(groups[idx]["text"] for idx in connected)
                    merged_sentences = []
                    component_sentences = []
                    
                    for idx in connected:
                        merged_sentences.extend(groups[idx]["sentences"])
                        component_sentences.append(groups[idx]["text"])
                    
                    group_dict = {
                        "text": merged_text,
                        "sentences": merged_sentences,
                        "is_sentence_group": True,
                        "component_sentences": component_sentences
                    }
                    merged_groups.append(group_dict)
                else:
                    # Only one sentence in the group
                    group_dict = {
                        "text": groups[connected[0]]["text"],
                        "sentences": groups[connected[0]]["sentences"],
                        "is_sentence_group": False,
                        "component_sentences": []
                    }
                    merged_groups.append(group_dict)
        
        all_groups.extend(merged_groups)
    
    return all_groups

def get_sample_groups(groups: List[Dict], sample_size: int = 3) -> List[Dict]:
    """Get a sample of sentence groups, prioritizing actual groups."""
    # First, separate groups and single sentences
    actual_groups = [g for g in groups if g["is_sentence_group"]]
    single_sentences = [g for g in groups if not g["is_sentence_group"]]
    
    # If we don't have enough groups, supplement with single sentences
    if len(actual_groups) < sample_size:
        samples = actual_groups.copy()
        remaining = sample_size - len(actual_groups)
        if single_sentences and remaining > 0:
            samples.extend(random.sample(single_sentences, min(remaining, len(single_sentences))))
    else:
        samples = random.sample(actual_groups, sample_size)
    
    return samples

def format_group_info_markdown(group: Dict, index: int) -> str:
    """Format information about a sentence group in markdown."""
    md = f"\n### Group {index + 1}\n\n"
    md += f"**Is Group:** {group['is_sentence_group']}\n\n"
    
    if group["is_sentence_group"]:
        md += f"**Number of Sentences:** {len(group['component_sentences'])}\n\n"
        md += "**Component Sentences:**\n\n"
        for i, sentence in enumerate(group["component_sentences"]):
            md += f"{i+1}. {sentence}\n\n"
    else:
        md += f"**Single Sentence:** {group['text']}\n\n"
    
    md += "---\n"
    return md

def print_group_info(group: Dict, index: int):
    """Print information about a sentence group."""
    print(f"\nGroup {index + 1}:")
    print(f"Is Group: {group['is_sentence_group']}")
    if group["is_sentence_group"]:
        print(f"Number of Sentences: {len(group['component_sentences'])}")
        print("Component Sentences:")
        for i, sentence in enumerate(group["component_sentences"]):
            print(f"  {i+1}. {sentence}")
    else:
        print(f"Single Sentence: {group['text']}")
    print("-" * 80)

def main():
    """Run the sentence grouping test with different configurations."""
    # Load the longest cover letter
    archive_dir = "text-archive"
    cover_letter_files = list(Path(archive_dir).glob("*.txt"))
    
    # Find the longest cover letter
    longest_file = None
    max_length = 0
    
    for file_path in cover_letter_files:
        with open(file_path, 'r') as f:
            content = f.read()
            if len(content) > max_length:
                max_length = len(content)
                longest_file = file_path
    
    if not longest_file:
        print("No cover letter files found.")
        return
    
    print(f"Testing with cover letter: {longest_file.name}")
    cover_letter_text = load_cover_letter(longest_file)
    
    # Define different configurations to test
    configs = [
        {
            "name": "Default Configuration",
            "max_group_size": 3,
            "similarity_threshold": 0.85,
            "use_entity_matching": True,
            "use_pronoun_detection": True,
            "use_similarity": True,
            "discourse_markers": ["additionally", "furthermore", "moreover", "however", "nevertheless", 
                                 "therefore", "thus", "consequently", "as a result", "in conclusion", 
                                 "finally", "for example", "for instance", "specifically", "in particular",
                                 "in contrast", "on the other hand", "similarly", "likewise"]
        },
        {
            "name": "Conservative Grouping",
            "max_group_size": 2,
            "similarity_threshold": 0.90,
            "use_entity_matching": True,
            "use_pronoun_detection": True,
            "use_similarity": True,
            "discourse_markers": ["additionally", "furthermore", "moreover", "however", "nevertheless", 
                                 "therefore", "thus", "consequently", "as a result"]
        },
        {
            "name": "Aggressive Grouping",
            "max_group_size": 4,
            "similarity_threshold": 0.75,
            "use_entity_matching": True,
            "use_pronoun_detection": True,
            "use_similarity": True,
            "discourse_markers": ["additionally", "furthermore", "moreover", "however", "nevertheless", 
                                 "therefore", "thus", "consequently", "as a result", "in conclusion", 
                                 "finally", "for example", "for instance", "specifically", "in particular",
                                 "in contrast", "on the other hand", "similarly", "likewise", "also",
                                 "and", "but", "so", "because", "since", "although", "though"]
        },
        {
            "name": "No Similarity Matching",
            "max_group_size": 3,
            "use_entity_matching": True,
            "use_pronoun_detection": True,
            "use_similarity": False,
            "discourse_markers": ["additionally", "furthermore", "moreover", "however", "nevertheless", 
                                 "therefore", "thus", "consequently", "as a result", "in conclusion"]
        },
        {
            "name": "Only Discourse Markers",
            "max_group_size": 3,
            "use_entity_matching": False,
            "use_pronoun_detection": False,
            "use_similarity": False,
            "discourse_markers": ["additionally", "furthermore", "moreover", "however", "nevertheless", 
                                 "therefore", "thus", "consequently", "as a result", "in conclusion", 
                                 "finally", "for example", "for instance", "specifically", "in particular"]
        }
    ]
    
    # Create markdown output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    md_filename = f"sentence_grouping_test_{timestamp}.md"
    
    with open(md_filename, 'w') as md_file:
        md_file.write(f"# Sentence Grouping Test Results\n\n")
        md_file.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        md_file.write(f"**Cover Letter:** {longest_file.name}\n\n")
        md_file.write("This document contains the results of testing different sentence grouping configurations.\n\n")
        
        # Test each configuration
        for config in configs:
            print("\n" + "=" * 80)
            print(f"Testing Configuration: {config['name']}")
            print("=" * 80)
            
            md_file.write(f"## {config['name']}\n\n")
            
            # Write configuration details to markdown
            md_file.write("### Configuration Details\n\n")
            md_file.write("```python\n")
            for key, value in config.items():
                if key == "discourse_markers":
                    md_file.write(f"{key}: [")
                    if len(value) > 5:
                        md_file.write(f"{', '.join(value[:5])}, ... ({len(value)} total)]")
                    else:
                        md_file.write(f"{', '.join(value)}]")
                else:
                    md_file.write(f"{key}: {value}")
                md_file.write("\n")
            md_file.write("```\n\n")
            
            # Process the cover letter with the current configuration
            groups = test_sentence_grouping(cover_letter_text, config)
            
            # Count the number of groups vs. single sentences
            num_groups = sum(1 for g in groups if g["is_sentence_group"])
            num_singles = sum(1 for g in groups if not g["is_sentence_group"])
            
            print(f"Total Items: {len(groups)}")
            print(f"Sentence Groups: {num_groups}")
            print(f"Single Sentences: {num_singles}")
            
            md_file.write("### Statistics\n\n")
            md_file.write(f"- **Total Items:** {len(groups)}\n")
            md_file.write(f"- **Sentence Groups:** {num_groups}\n")
            md_file.write(f"- **Single Sentences:** {num_singles}\n")
            md_file.write(f"- **Group Percentage:** {num_groups/len(groups)*100:.1f}%\n\n")
            
            # Get sample groups
            samples = get_sample_groups(groups)
            
            # Print sample groups
            print("\nSample Groups:")
            md_file.write("### Sample Groups\n")
            
            for i, group in enumerate(samples):
                print_group_info(group, i)
                md_file.write(format_group_info_markdown(group, i))
            
            md_file.write("\n\n")
    
    print(f"\nResults saved to: {md_filename}")

if __name__ == "__main__":
    main()
