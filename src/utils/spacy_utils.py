#!/usr/bin/env python3
"""
spaCy Utilities - Helper functions for NLP processing using spaCy.

This module provides utilities for processing text using spaCy,
including tag extraction, content parsing, and job tag prioritization.
"""

import spacy
import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter

# Load spaCy model 
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    # Fallback to small model if medium is not available
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Warning: Using small spaCy model instead of medium. Some features may be limited.")
    except OSError:
        print("Error: No spaCy model found. Please install one with: python -m spacy download en_core_web_md")
        nlp = None

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

def prioritize_tags_for_job(job_text: str, categories: Dict) -> Dict[str, List[str]]:
    """
    Analyze job text and prioritize tags based on importance.
    
    This function uses spaCy to:
    1. Extract all potential tags from the job description
    2. Categorize tags based on frequency and importance
    3. Sort tags into high, medium, and low priority buckets
    
    Args:
        job_text: Job posting text
        categories: Categories from YAML file
        
    Returns:
        Dict with high, medium, and low priority tags
    """
    if not nlp:
        return {"high_priority": [], "medium_priority": [], "low_priority": []}
    
    # Extract all potential tags from job text
    all_tags = extract_tags_from_text(job_text)
    
    # Get frequency of each tag in the text
    tag_counter = Counter()
    for tag in all_tags:
        tag_counter[tag] += 1
    
    # Get the most frequent tags
    most_frequent = [tag for tag, count in tag_counter.most_common(20)]
    
    # Map tags to categories
    categorized_tags = defaultdict(list)
    for tag in all_tags:
        # Find which category this tag belongs to
        for category, tags in categories.items():
            if any(cat_tag in tag for cat_tag in tags) or any(tag in cat_tag for cat_tag in tags):
                categorized_tags[category].append(tag)
                break
    
    # Create priority buckets
    high_priority = []
    medium_priority = []
    low_priority = []
    
    # First, add the most frequent tags to high priority
    for tag in most_frequent[:5]:
        if tag not in high_priority:
            high_priority.append(tag)
    
    # Next, add categorized tags based on their category importance
    important_categories = ["skills_competencies", "product_outcomes"]
    medium_categories = ["team_people", "industry_domain"]
    
    for category, tags in categorized_tags.items():
        if category in important_categories:
            # Add to high priority if not already there
            for tag in tags[:3]:  # Top 3 tags from important categories
                if tag not in high_priority and len(high_priority) < 10:
                    high_priority.append(tag)
        elif category in medium_categories:
            # Add to medium priority
            for tag in tags[:3]:  # Top 3 tags from medium categories
                if tag not in high_priority and tag not in medium_priority and len(medium_priority) < 15:
                    medium_priority.append(tag)
        else:
            # Add to low priority
            for tag in tags[:3]:  # Top 3 tags from other categories
                if tag not in high_priority and tag not in medium_priority and tag not in low_priority and len(low_priority) < 20:
                    low_priority.append(tag)
    
    # Add remaining frequent tags to appropriate buckets
    for tag in most_frequent:
        if tag not in high_priority and tag not in medium_priority and tag not in low_priority:
            if len(medium_priority) < 15:
                medium_priority.append(tag)
            elif len(low_priority) < 20:
                low_priority.append(tag)
    
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
        
    tag_doc = nlp(tag)
    
    # Calculate similarity between tag and all other tags
    similarities = []
    for other_tag in all_tags:
        if other_tag == tag:
            continue
            
        other_doc = nlp(other_tag)
        similarity = tag_doc.similarity(other_doc)
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
            similarity = doc_i.similarity(doc_j)
            
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
        keyword_doc = spacy_nlp(keyword)
        
        # Calculate similarity with the full text
        similarity = max(keyword_doc.similarity(doc), 0.0)  # Ensure non-negative
        
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
