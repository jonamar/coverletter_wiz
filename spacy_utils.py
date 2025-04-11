#!/usr/bin/env python3
"""
SpaCy Utilities - Shared functions for NLP processing using spaCy

This module provides common functionality for:
1. Sentence grouping based on linguistic connections
2. Tag generation using semantic similarity
3. Helper functions for NLP tasks

These utilities replace LLM-based processing with more efficient spaCy-based methods.
"""

import spacy
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Set, Any, Optional

# Load spaCy model with word vectors for semantic similarity
# Note: This requires downloading the model with: python -m spacy download en_core_web_md
try:
    nlp = spacy.load("en_core_web_md")
    # Configure sentence segmentation to be more robust with special characters
    nlp.add_pipe("sentencizer", before="parser")
except OSError:
    # Fallback to small model if medium is not available
    print("Warning: en_core_web_md not found. Using en_core_web_sm instead.")
    print("For better results, run: python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("sentencizer", before="parser")

# Define discourse markers and connective phrases
DISCOURSE_MARKERS = {
    # Standard discourse markers
    "additionally", "furthermore", "moreover", "however", "nevertheless", 
    "therefore", "thus", "consequently", "as a result", "in conclusion", 
    "finally", "for example", "for instance", "specifically", "in particular",
    "in contrast", "on the other hand", "similarly", "likewise",
    
    # Pronouns that might indicate connection to previous sentence
    "this", "these", "that", "those", "it", "they", "their", "them",
    
    # Additional connective phrases
    "because of this", "due to this", "in addition", "as such", "in fact",
    "indeed", "namely", "in other words", "to clarify", "to illustrate",
    "accordingly", "given that", "in this way", "to that end"
}

def identify_sentence_groups(text: str, max_group_size: int = 3) -> List[Dict]:
    """
    Process a paragraph and identify logically connected sentence groups.
    
    Args:
        text: Paragraph text to process
        max_group_size: Maximum number of sentences in a group (default: 3)
        
    Returns:
        List of sentence groups with their component sentences
    """
    # Process the paragraph with spaCy
    # Pre-process text to handle special characters that might break sentence segmentation
    # Replace problematic patterns with temporary placeholders
    text_for_processing = text
    
    # Store replacements to restore later
    replacements = []
    
    # Handle square brackets which can cause incorrect sentence breaks
    import re
    bracket_pattern = r'\[([^\]]+)\]'
    
    def replace_brackets(match):
        content = match.group(1)
        placeholder = f"BRACKET_CONTENT_{len(replacements)}"
        replacements.append((placeholder, f"[{content}]"))
        return placeholder
    
    text_for_processing = re.sub(bracket_pattern, replace_brackets, text_for_processing)
    
    # Process with spaCy
    doc = nlp(text_for_processing)
    sentences = list(doc.sents)
    
    # Restore original text for each sentence
    restored_sentences = []
    for sent in sentences:
        sent_text = sent.text
        for placeholder, original in replacements:
            sent_text = sent_text.replace(placeholder, original)
        # Create a custom span with the restored text
        restored_sentences.append(sent_text)
    
    if len(restored_sentences) <= 1:
        # Return the single sentence as its own group
        return [{
            "text": text.strip(),
            "sentences": [{"text": text.strip(), "index": 0}],
            "is_sentence_group": False,
            "component_sentences": []
        }]
    
    # Create initial groups (start with each sentence as its own group)
    groups = [{
        "index": i,
        "text": sent.strip(),
        "sentences": [{"text": sent.strip(), "index": i}],
        "connections": set(),
        "is_sentence_group": False
    } for i, sent in enumerate(restored_sentences)]
    
    # Identify connections between sentences
    for i in range(len(restored_sentences) - 1):
        current_text = restored_sentences[i]
        next_text = restored_sentences[i + 1]
        
        # Process with spaCy for linguistic analysis
        current_doc = nlp(current_text)
        next_doc = nlp(next_text)
        
        # Check for discourse markers at the beginning of the next sentence
        next_starts_with_marker = False
        next_text_lower = next_text.lower().strip()
        
        for marker in DISCOURSE_MARKERS:
            if next_text_lower.startswith(marker):
                next_starts_with_marker = True
                groups[i]["connections"].add(i + 1)
                groups[i + 1]["connections"].add(i)
                break
        
        # Check for coreference (simple version - checking for shared entities)
        current_ents = {e.text.lower() for e in current_doc.ents}
        next_ents = {e.text.lower() for e in next_doc.ents}
        if current_ents and next_ents and current_ents.intersection(next_ents):
            groups[i]["connections"].add(i + 1)
            groups[i + 1]["connections"].add(i)
        
        # Check for pronoun references (simplistic approach)
        if any(token.lower_ in ["it", "this", "these", "they", "their", "them"] for token in next_doc[:3]):
            groups[i]["connections"].add(i + 1)
            groups[i + 1]["connections"].add(i)
        
        # Check for semantic similarity between sentences
        # Only calculate similarity if both sentences have vectors
        if current_doc.has_vector and next_doc.has_vector:
            sim_score = current_doc.similarity(next_doc)
            if sim_score > 0.85:  # Maintain the 0.85 threshold which works well
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
    
    return merged_groups

def load_category_expansions(expansion_file="category_expansions.yaml"):
    """
    Load category keyword expansions from YAML file.
    
    Args:
        expansion_file: Path to the YAML file containing category expansions
        
    Returns:
        Dictionary mapping category keywords to lists of related terms
    """
    try:
        with open(expansion_file, "r") as file:
            expansions = yaml.safe_load(file)
        return expansions or {}
    except Exception as e:
        print(f"Warning: Could not load category expansions: {e}")
        return {}

def get_expanded_categories(categories, expansion_file="category_expansions.yaml"):
    """
    Get categories with their expanded terms for matching.
    
    Args:
        categories: Original categories dictionary
        expansion_file: Path to the YAML file containing category expansions
        
    Returns:
        Tuple of (expanded_categories, term_to_category_mapping)
    """
    # Load original categories
    original_categories = categories.copy()
    
    # Load expansions
    expansions = load_category_expansions(expansion_file)
    
    # Create mapping to track which original category each expansion belongs to
    expanded_categories = {}
    term_to_category = {}
    
    # Process each category
    for category, terms in original_categories.items():
        expanded_categories[category] = terms.copy()
        
        # Map each term to its category
        for term in terms:
            term_to_category[term] = category
            
            # Add expansions if available
            if term in expansions:
                for expanded_term in expansions[term]:
                    if expanded_term not in expanded_categories[category]:
                        expanded_categories[category].append(expanded_term)
                        term_to_category[expanded_term] = term
    
    return expanded_categories, term_to_category

def create_tag_embeddings(categories: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Create embeddings for all tags in the categories.
    
    Args:
        categories: Dictionary of category names to lists of tags
        
    Returns:
        Dictionary mapping tag names to their embeddings
    """
    tag_embeddings = {}
    
    # Flatten all tags
    all_tags = []
    for category, tags in categories.items():
        all_tags.extend(tags)
    
    # Create embeddings for each tag
    for tag in all_tags:
        # Process the tag with spaCy to get its vector
        tag_doc = nlp(tag)
        tag_embeddings[tag] = tag_doc.vector
    
    return tag_embeddings

def assign_tags_with_spacy(text: str, categories: Dict[str, List[str]], max_tags: int = 5) -> List[Dict]:
    """
    Assign tags to text using spaCy's semantic similarity and keyword matching.
    
    Args:
        text: Text to analyze
        categories: Dictionary of category names to lists of tags
        max_tags: Maximum number of tags to return
        
    Returns:
        List of dictionaries with tag names and confidence scores
    """
    # Get expanded categories and term mapping
    expanded_categories, term_to_category = get_expanded_categories(categories)
    
    # Flatten all tags including expansions
    all_tags = []
    for category, tags in expanded_categories.items():
        all_tags.extend(tags)
    
    # Process the text with spaCy
    doc = nlp(text)
    
    # 1. Direct keyword matching (highest priority)
    keyword_matches = {}
    for tag in all_tags:
        # Check if tag appears directly in text (case insensitive)
        if tag.lower() in text.lower():
            keyword_matches[tag] = 1.0  # Direct match gets highest score
    
    # 2. Extract key phrases and check similarity to tags
    phrase_matches = {}
    key_phrases = [chunk.text for chunk in doc.noun_chunks]
    
    for phrase in key_phrases:
        phrase_doc = nlp(phrase)
        # Skip phrases that are too short or have no vectors
        if len(phrase_doc) <= 1 or not phrase_doc.has_vector:
            continue
            
        for tag in all_tags:
            tag_doc = nlp(tag)
            # Skip tags that are too short or have no vectors
            if len(tag_doc) <= 1 or not tag_doc.has_vector:
                continue
                
            similarity = phrase_doc.similarity(tag_doc)
            if similarity > 0.65:  # Slightly lower threshold to capture more matches
                if tag in phrase_matches:
                    phrase_matches[tag] = max(phrase_matches[tag], similarity)
                else:
                    phrase_matches[tag] = similarity
    
    # 3. Calculate overall document similarity to tags
    doc_similarities = {}
    # Skip document similarity if the doc has no vector
    if doc.has_vector:
        for tag in all_tags:
            tag_doc = nlp(tag)
            # Skip tags that are too short or have no vectors
            if len(tag_doc) <= 1 or not tag_doc.has_vector:
                continue
                
            doc_similarities[tag] = doc.similarity(tag_doc)
    
    # Combine all scores with weights
    combined_scores = {}
    for tag in all_tags:
        # Prioritize direct matches, then phrase matches, then doc similarity
        score = (
            keyword_matches.get(tag, 0) * 3 +  # Weight direct matches higher
            phrase_matches.get(tag, 0) * 2 +   # Weight phrase matches medium
            doc_similarities.get(tag, 0)       # Weight overall similarity lower
        )
        combined_scores[tag] = score
    
    # Get top tags
    top_tags = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Filter to only include tags with a score
    scored_tags = [(tag, score) for tag, score in top_tags if score > 0]
    
    # Map expanded terms back to their original categories
    final_tags = []
    seen_categories = set()
    
    for tag, score in scored_tags:
        original_tag = term_to_category.get(tag, tag)
        if original_tag not in seen_categories:
            # Normalize score to 0-1 range
            normalized_score = min(1.0, score / 6.0)  # 6.0 is the max possible score (1*3 + 1*2 + 1)
            final_tags.append({"name": original_tag, "confidence": normalized_score})
            seen_categories.add(original_tag)
            
    return final_tags[:max_tags]

def prioritize_tags_for_job(job_text: str, categories: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Analyze a job posting and prioritize tags into high, medium, and low priority.
    
    Args:
        job_text: Job posting text
        categories: Dictionary of category names to lists of tags
        
    Returns:
        Dictionary with high_priority, medium_priority, and low_priority tag lists
    """
    # Flatten all tags
    all_tags = []
    for category, tags in categories.items():
        all_tags.extend(tags)
    
    # Process the job posting
    doc = nlp(job_text)
    
    # Calculate scores for each tag
    tag_scores = {}
    for tag in all_tags:
        # 1. Direct keyword matching (highest priority)
        direct_match_score = 0
        if tag.lower() in job_text.lower():
            # Count occurrences (more occurrences = higher importance)
            occurrences = job_text.lower().count(tag.lower())
            direct_match_score = min(occurrences * 0.2, 1.0)  # Cap at 1.0
        
        # 2. Semantic similarity
        tag_doc = nlp(tag)
        similarity_score = doc.similarity(tag_doc)
        
        # 3. Location importance (tags in the first 1/3 of the document are often more important)
        first_third_text = job_text[:len(job_text)//3]
        location_score = 0.5 if tag.lower() in first_third_text.lower() else 0
        
        # Combine scores with weights
        combined_score = (
            direct_match_score * 0.5 +
            similarity_score * 0.3 +
            location_score * 0.2
        )
        
        tag_scores[tag] = combined_score
    
    # Sort tags by score
    sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Split into priority levels - top 30% are high, middle 40% are medium, bottom 30% are low
    # Ensure we get exactly 10 tags total
    high_count = 3  # Get 3-4 high priority tags
    medium_count = 4  # Get 3-4 medium priority tags
    low_count = 3  # Get 2-3 low priority tags
    
    all_prioritized_tags = sorted_tags[:high_count + medium_count + low_count]
    
    # If we don't have enough tags, adjust the counts
    if len(all_prioritized_tags) < high_count + medium_count + low_count:
        available_count = len(all_prioritized_tags)
        high_count = max(1, available_count // 3)
        low_count = max(1, available_count // 3)
        medium_count = available_count - high_count - low_count
    
    high_priority = [tag for tag, score in all_prioritized_tags[:high_count]]
    medium_priority = [tag for tag, score in all_prioritized_tags[high_count:high_count+medium_count]]
    low_priority = [tag for tag, score in all_prioritized_tags[high_count+medium_count:]]
    
    return {
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "low_priority": low_priority
    }
