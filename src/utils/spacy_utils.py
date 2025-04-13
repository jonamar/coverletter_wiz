#!/usr/bin/env python3
"""
spaCy Utilities - Helper functions for NLP processing using spaCy.

This module provides utilities for processing text using spaCy,
including tag extraction, similarity calculation, and content analysis.
"""

import os
import re
from typing import List, Dict, Set, Tuple, Optional
import spacy
import warnings

# Try to load the large model first, then fall back to medium or small if needed
nlp = None
try:
    nlp = spacy.load("en_core_web_lg")
    print("Loaded spaCy model: en_core_web_lg")
except OSError:
    try:
        nlp = spacy.load("en_core_web_md")
        print("Loaded spaCy model: en_core_web_md (fallback)")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
            print("Loaded spaCy model: en_core_web_sm (fallback)")
        except OSError:
            print("Warning: Could not load any spaCy model. Some features may not work correctly.")

# Suppress the specific warning about empty vectors
warnings.filterwarnings("ignore", message=".*W008.*")

def normalize_text(text):
    """
    Normalize text for spaCy processing:
    - Replace underscores with spaces
    - Trim extra whitespace
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    # Replace underscores with spaces for better semantic matching
    normalized = text.replace('_', ' ')
    # Trim extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

def has_vector(doc):
    """Check if a spaCy doc has a valid vector."""
    if not doc or not hasattr(doc, "vector") or not doc.vector.size:
        return False
    
    # Check if vector is all zeros
    return doc.vector.any()

def safe_similarity(doc1, doc2, default_score=0.0):
    """
    Calculate similarity between two spaCy docs safely.
    
    Args:
        doc1: First spaCy doc
        doc2: Second spaCy doc
        default_score: Default score to return if similarity can't be calculated
        
    Returns:
        Similarity score or default value
    """
    if not doc1 or not doc2 or not has_vector(doc1) or not has_vector(doc2):
        return default_score
    
    try:
        return doc1.similarity(doc2)
    except:
        return default_score

def extract_tags_from_text(text: str) -> List[str]:
    """
    Extract potential tags from text using spaCy's named entity recognition.
    
    Args:
        text: Text to extract tags from
        
    Returns:
        List of extracted tags
    """
    if not nlp:
        return []
        
    doc = nlp(text)
    
    # Extract named entities (technical terms, skills, etc.)
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "GPE", "LAW", "WORK_OF_ART"]]
    
    # Extract noun chunks that might be skills or technologies
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
    
    # Extract specific skill keywords
    skill_patterns = [
        r'\b(agile|scrum|kanban|waterfall)\b',
        r'\b(aws|azure|gcp|cloud)\b',
        r'\b(python|java|javascript|typescript|ruby|php|c\+\+|c#|html|css)\b',
        r'\b(sql|postgresql|mysql|mongodb|nosql|database)\b',
        r'\b(api|rest|graphql|oauth)\b',
        r'\b(git|github|gitlab|bitbucket|version control)\b',
        r'\b(docker|kubernetes|container|k8s)\b',
        r'\b(ci/cd|jenkins|github actions|gitlab ci)\b',
        r'\b(leadership|management|mentoring|coaching)\b',
        r'\b(testing|tdd|bdd|unittest|pytest|jest)\b'
    ]
    
    skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, text.lower())
        skills.extend(matches)
    
    # Combine all potential tags
    all_tags = entities + noun_chunks + skills
    
    # Remove duplicates and normalize
    unique_tags = set()
    normalized_tags = []
    
    for tag in all_tags:
        tag = tag.strip()
        tag_lower = tag.lower()
        
        if tag_lower and tag_lower not in unique_tags and len(tag) > 2:
            unique_tags.add(tag_lower)
            normalized_tags.append(tag_lower)
    
    return normalized_tags

def prioritize_tags_for_job(job_text: str, categories: Dict, manual_keywords: Dict = None) -> Dict[str, List[str]]:
    """
    Analyze job text and prioritize tags based on importance using semantic matching.
    
    This function uses spaCy to:
    1. Start with categories from the YAML file as the source of truth
    2. Use semantic matching to find which categories are most relevant to the job text
    3. Incorporate manual keywords if provided
    
    Args:
        job_text: Job posting text
        categories: Categories from YAML file
        manual_keywords: Optional dictionary of manual keywords by priority level
        
    Returns:
        Dict with high, medium, and low priority tags
    """
    if not nlp:
        return {"high_priority": [], "medium_priority": [], "low_priority": []}
    
    # Process the job text with spaCy
    job_doc = nlp(normalize_text(job_text))
    
    # Calculate semantic similarity between job text and each category
    category_scores = {}
    
    # First, score the categories themselves (not the expanded tags)
    for category in categories.keys():
        # Convert underscores to spaces for better semantic matching
        category_text = normalize_text(category)
        category_doc = nlp(category_text)
        
        # Calculate similarity between job text and category
        similarity = safe_similarity(job_doc, category_doc, default_score=0.0)
        category_scores[category] = similarity * 2.0  # Give higher weight to category concepts
    
    # Now also score individual tags within each category
    tag_scores = {}
    tag_to_category = {}
    
    for category, tags in categories.items():
        for tag in tags:
            # Normalize tag text
            tag_text = normalize_text(tag)
            tag_doc = nlp(tag_text)
            # Calculate similarity between job text and tag
            similarity = safe_similarity(job_doc, tag_doc, default_score=0.0)
            
            # Store the tag's score and its parent category
            tag_scores[tag] = similarity
            tag_to_category[tag] = category
    
    # Combine scores: if a tag has a high score and its category has a high score,
    # it's more likely to be relevant
    combined_scores = {}
    for tag, score in tag_scores.items():
        category = tag_to_category[tag]
        category_score = category_scores.get(category, 0)
        
        # Weighted combination of tag and category scores
        combined_scores[tag] = (score * 0.8) + (category_score * 0.4)
    
    # Sort tags by combined score
    sorted_tags = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Determine priority thresholds based on score distribution
    if sorted_tags:
        scores = [score for _, score in sorted_tags]
        max_score = max(scores)
        high_threshold = max_score * 0.75
        medium_threshold = max_score * 0.5
        
        # Assign tags to priority levels based on thresholds
        high_priority = [tag for tag, score in sorted_tags if score >= high_threshold]
        medium_priority = [tag for tag, score in sorted_tags if high_threshold > score >= medium_threshold]
        low_priority = [tag for tag, score in sorted_tags if medium_threshold > score > 0.3]  # Minimum threshold
        
        # Limit the number of tags in each priority level
        high_priority = high_priority[:7]  # Allow more high-priority tags
        medium_priority = medium_priority[:10]
        low_priority = low_priority[:12]
    else:
        high_priority = []
        medium_priority = []
        low_priority = []
    
    # Incorporate manual keywords if provided
    if manual_keywords:
        if "high_priority" in manual_keywords:
            for kw in manual_keywords["high_priority"]:
                if kw not in high_priority and kw not in medium_priority and kw not in low_priority:
                    high_priority.append(kw)
        
        if "medium_priority" in manual_keywords:
            for kw in manual_keywords["medium_priority"]:
                if kw not in high_priority and kw not in medium_priority and kw not in low_priority:
                    medium_priority.append(kw)
        
        if "low_priority" in manual_keywords:
            for kw in manual_keywords["low_priority"]:
                if kw not in high_priority and kw not in medium_priority and kw not in low_priority:
                    low_priority.append(kw)
    
    return {
        "high_priority": high_priority,
        "medium_priority": medium_priority,
        "low_priority": low_priority
    }

def get_related_tags(tag: str, all_tags: List[str]) -> List[str]:
    """
    Find tags that are semantically related to the input tag.
    
    Args:
        tag: Tag to find relations for
        all_tags: List of all available tags
        
    Returns:
        List of related tags
    """
    if not nlp:
        return []
        
    tag_doc = nlp(normalize_text(tag))
    
    # Calculate similarity between tag and all other tags
    similarities = []
    for other_tag in all_tags:
        if other_tag == tag:
            continue
            
        other_doc = nlp(normalize_text(other_tag))
        similarity = safe_similarity(tag_doc, other_doc, default_score=0.0)
        similarities.append((other_tag, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top related tags (similarity > 0.5)
    related = [t for t, s in similarities if s > 0.5]
    
    return related[:5]  # Limit to top 5 related tags

def parse_content_into_blocks(text: str) -> List[Dict]:
    """
    Parse content into blocks (sentences and paragraphs) for rating.
    
    Args:
        text: Text to parse
        
    Returns:
        List of content blocks with metadata
    """
    if not nlp:
        return []
        
    doc = nlp(text)
    
    # Extract sentences
    sentences = []
    for sent in doc.sents:
        if len(sent.text.split()) >= 3:  # Skip very short sentences
            sentences.append({"text": sent.text.strip(), "is_content_group": False, "component_content": []})
    
    # Create paragraph groups (consecutive 2-3 sentences)
    paragraph_groups = []
    for i in range(0, len(sentences) - 1):
        # Create a group of 2 sentences
        group_text = sentences[i]["text"] + " " + sentences[i+1]["text"]
        component_sentences = [sentences[i]["text"], sentences[i+1]["text"]]
        
        paragraph_groups.append({
            "text": group_text,
            "is_content_group": True,
            "component_content": component_sentences
        })
        
        # Create a group of 3 sentences if possible
        if i < len(sentences) - 2:
            group_text = sentences[i]["text"] + " " + sentences[i+1]["text"] + " " + sentences[i+2]["text"]
            component_sentences = [sentences[i]["text"], sentences[i+1]["text"], sentences[i+2]["text"]]
            
            paragraph_groups.append({
                "text": group_text,
                "is_content_group": True,
                "component_content": component_sentences
            })
    
    # Combine all content blocks
    all_blocks = sentences + paragraph_groups
    
    # Extract tags for each block
    for block in all_blocks:
        block["tags"] = extract_tags_from_text(block["text"])
    
    return all_blocks

def analyze_content_block_similarity(blocks: List[Dict]) -> Dict[str, List[str]]:
    """
    Analyze similarity between content blocks to find duplicates or near-duplicates.
    
    Args:
        blocks: List of content blocks
        
    Returns:
        Dict mapping content block text to similar blocks
    """
    if not nlp:
        return {}
        
    similarity_map = {}
    processed_docs = {}
    
    # Process each block with spaCy
    for i, block in enumerate(blocks):
        text = block.get("text", "")
        if not text:
            continue
            
        # Process with spaCy
        doc = nlp(text)
        processed_docs[i] = doc
        
    # Compare each block with every other block
    for i, doc_i in processed_docs.items():
        similar_blocks = []
        
        for j, doc_j in processed_docs.items():
            if i == j:
                continue
                
            # Calculate similarity
            similarity = safe_similarity(doc_i, doc_j, default_score=0.0)
            
            # If similarity is above threshold, add to similar blocks
            if similarity > 0.8:  # Adjust threshold as needed
                similar_blocks.append({
                    "text": blocks[j].get("text", ""),
                    "similarity": similarity
                })
        
        # Sort by similarity (highest first)
        similar_blocks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Add to similarity map
        similarity_map[blocks[i].get("text", "")] = similar_blocks
        
    return similarity_map

def identify_sentence_groups(paragraph: str, nlp=None) -> List[Dict]:
    """
    Identify sentence groups within a paragraph.
    
    This function uses spaCy to split a paragraph into sentences and then
    identifies which sentences should be grouped together based on semantic
    and syntactic relationships.
    
    Args:
        paragraph: Paragraph text to process
        nlp: Optional spaCy model (uses global nlp if not provided)
        
    Returns:
        List of sentence groups with metadata
    """
    # Use provided nlp model or global one
    spacy_nlp = nlp if nlp is not None else globals().get('nlp')
    
    if not spacy_nlp:
        # Fallback to simple sentence splitting if no spaCy model
        sentences = [s.strip() for s in re.split(r'[.!?]+', paragraph) if s.strip()]
        return [{"text": s, "is_sentence_group": False, "component_sentences": [s]} for s in sentences]
    
    # Process the paragraph with spaCy
    doc = spacy_nlp(paragraph)
    
    # Extract sentences
    sentences = list(doc.sents)
    
    # If only one sentence, return it directly
    if len(sentences) <= 1:
        return [{
            "text": paragraph,
            "is_sentence_group": False,
            "component_sentences": [paragraph]
        }]
    
    # Initialize groups
    groups = []
    current_group = []
    
    # Group sentences based on semantic and syntactic relationships
    for i, sentence in enumerate(sentences):
        sentence_text = sentence.text.strip()
        
        # Skip empty sentences
        if not sentence_text:
            continue
        
        # Start a new group if needed
        if not current_group:
            current_group.append(sentence_text)
        else:
            # Check if this sentence should be grouped with the previous ones
            # Criteria for grouping:
            # 1. Short sentences (less than 5 words)
            # 2. Sentences starting with conjunctions or relative pronouns
            # 3. Sentences without a main verb
            
            words = [token.text for token in sentence if not token.is_punct]
            
            # Check for short sentences
            is_short = len(words) < 5
            
            # Check for sentences starting with conjunctions or relative pronouns
            starts_with_conjunction = False
            if len(sentence) > 0:
                first_token = sentence[0]
                starts_with_conjunction = first_token.pos_ in ["CCONJ", "SCONJ"] or first_token.dep_ == "mark"
            
            # Check for sentences without a main verb
            has_main_verb = any(token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"] for token in sentence)
            
            # Decide whether to group with previous sentence
            if is_short or starts_with_conjunction or not has_main_verb:
                current_group.append(sentence_text)
            else:
                # Finish current group and start a new one
                if current_group:
                    groups.append({
                        "text": " ".join(current_group),
                        "is_sentence_group": len(current_group) > 1,
                        "component_sentences": current_group.copy()
                    })
                current_group = [sentence_text]
    
    # Add the last group if not empty
    if current_group:
        groups.append({
            "text": " ".join(current_group),
            "is_sentence_group": len(current_group) > 1,
            "component_sentences": current_group.copy()
        })
    
    return groups

def assign_tags_with_spacy(text: str, categories: Dict, max_tags: int = 5, nlp=None, context: str = "") -> List[Dict]:
    """
    Assign tags to text using spaCy-based semantic similarity and keyword matching.
    
    This function uses spaCy to:
    1. Extract potential tags from the text
    2. Match extracted tags to categories from the YAML file
    3. Calculate confidence scores for each tag
    
    Args:
        text: Text to assign tags to
        categories: Categories from YAML file
        max_tags: Maximum number of tags to return
        nlp: Optional spaCy model (uses global nlp if not provided)
        context: Optional context for better tag assignment
        
    Returns:
        List of tags with confidence scores
    """
    # Use provided nlp model or global one
    spacy_nlp = nlp if nlp is not None else globals().get('nlp')
    
    if not spacy_nlp:
        return []
    
    # Process the text with spaCy
    doc = spacy_nlp(text)
    
    # Extract potential tags
    extracted_tags = extract_tags_from_text(text)
    
    # Add context if provided
    if context:
        context_tags = extract_tags_from_text(context)
        # Add context tags with lower weight
        extracted_tags.extend(context_tags)
    
    # Get all category keywords
    all_keywords = []
    category_weights = {}
    
    for category in categories.get("categories", []):
        category_name = category.get("name", "")
        keywords = category.get("keywords", [])
        weight = category.get("weight", 1.0)
        
        all_keywords.extend(keywords)
        
        # Store category weight for each keyword
        for keyword in keywords:
            category_weights[keyword] = weight
    
    # Calculate tag scores
    tag_scores = []
    
    # Check for direct matches first
    direct_matches = set(extracted_tags) & set(all_keywords)
    
    for tag in direct_matches:
        # Direct matches get high confidence
        tag_scores.append({
            "name": tag,
            "confidence": 0.9 * category_weights.get(tag, 1.0),
            "match_type": "direct"
        })
    
    # Check for semantic similarity with spaCy
    for keyword in all_keywords:
        if keyword in direct_matches:
            continue  # Skip already matched keywords
            
        # Get spaCy docs for comparison
        keyword_doc = spacy_nlp(normalize_text(keyword))
        
        # Calculate similarity with the full text
        similarity = safe_similarity(keyword_doc, doc, default_score=0.0)
        
        # Only include if similarity is above threshold
        if similarity > 0.6:  # Adjust threshold as needed
            tag_scores.append({
                "name": keyword,
                "confidence": similarity * category_weights.get(keyword, 1.0),
                "match_type": "semantic"
            })
    
    # Sort by confidence score (highest first)
    tag_scores.sort(key=lambda x: x["confidence"], reverse=True)
    
    # Return top N tags
    return tag_scores[:max_tags]

def find_similar_tags(tag: str, all_tags: List[str], max_results: int = 5) -> List[Tuple[str, float]]:
    """
    Find similar tags to a given tag.
    
    Args:
        tag: Tag to find similar tags for
        all_tags: List of all tags to search
        max_results: Maximum number of results to return
        
    Returns:
        List of tuples (tag, similarity) of similar tags
    """
    if not nlp:
        return []
    
    # Process tag with spaCy
    tag_text = normalize_text(tag)
    tag_doc = nlp(tag_text)
    
    # Find similar tags
    similarities = []
    for other_tag in all_tags:
        if other_tag == tag:
            continue
            
        other_text = normalize_text(other_tag)
        other_doc = nlp(other_text)
        similarity = safe_similarity(tag_doc, other_doc, default_score=0.0)
        similarities.append((other_tag, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top similar tags
    return similarities[:max_results]
