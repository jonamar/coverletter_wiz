#!/usr/bin/env python3
"""
Generate Report CLI - Generate job match reports with content matching and cover letters.

This module provides a command-line interface for generating comprehensive
job match reports that analyze job descriptions, identify matching content
from the user's content database, and optionally generate cover letter drafts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Union

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR, REPORTS_DIR
from src.utils.spacy_utils import prioritize_tags_for_job, nlp, safe_similarity, has_vector, normalize_text, assign_tags_with_spacy, preprocess_job_text, analyze_content_block_similarity
from src.utils.diff_utils import display_text_differences
from src.utils.ollama_utils import generate_cover_letter

# Constants
DEFAULT_JOBS_FILE = os.path.join(DATA_DIR, "json/analyzed_jobs.json")
DEFAULT_CONTENT_FILE = os.path.join(DATA_DIR, "json/cover_letter_content.json")
CATEGORIES_FILE = os.path.join(DATA_DIR, "config/categories.yaml")
MIN_RATING_THRESHOLD = 6.0  # Minimum rating for content blocks to be included
DEFAULT_CONTENT_WEIGHT = 0.7  # Weight to give content rating vs tag matching
DEFAULT_LLM_MODEL = "llama3"  # Default LLM model for cover letter generation
DEFAULT_SIMILARITY_THRESHOLD = 0.8  # Default similarity threshold for semantic deduplication

# Import locally to avoid circular imports
from src.utils.spacy_utils import prioritize_tags_for_job, nlp, safe_similarity, has_vector, normalize_text, assign_tags_with_spacy, preprocess_job_text

def add_keywords_to_categories(keywords: List[str], categories: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], List[Tuple[str, str, float]]]:
    """Adds new keywords to appropriate categories based on semantic similarity.
    
    This function takes a list of keywords and adds them to the most semantically similar
    category in the provided categories dictionary. It uses spaCy for semantic similarity
    matching between keywords and category names/existing keywords.
    
    Args:
        keywords: List of keywords to add to categories.
        categories: Dictionary mapping category names to lists of keywords.
            
    Returns:
        Tuple containing:
            - Updated categories dictionary with new keywords added.
            - List of tuples with (keyword, category, score) for each added keyword.
    """
    added_keywords = []
    
    for keyword in keywords:
        # Skip if keyword already exists in any category
        if any(keyword in cat_keywords for cat_keywords in categories.values()):
            print(f"Keyword '{keyword}' already exists in categories")
            continue
        
        # Find best category
        best_category = None
        best_score = 0
        
        # Normalize keyword for better spaCy processing
        keyword_normalized = normalize_text(keyword)
        keyword_doc = nlp(keyword_normalized)
        
        # Check similarity with each category
        for category, cat_keywords in categories.items():
            # Check category name
            category_text = normalize_text(category)
            category_doc = nlp(category_text)
            category_score = safe_similarity(keyword_doc, category_doc) * 2.0
            
            # Check existing keywords
            keyword_scores = []
            for existing_kw in cat_keywords:
                existing_normalized = normalize_text(existing_kw)
                existing_doc = nlp(existing_normalized)
                keyword_scores.append(safe_similarity(keyword_doc, existing_doc))
            
            # Get average of top 3 keyword scores
            top_scores = sorted(keyword_scores, reverse=True)[:3]
            kw_score = sum(top_scores) / len(top_scores) if top_scores else 0
            
            # Combined score
            score = (category_score * 0.7) + (kw_score * 0.3)
            
            if score > best_score:
                best_score = score
                best_category = category
        
        # Add to best category
        if best_category and best_score > 0.5:
            categories[best_category].append(keyword)
            added_keywords.append((keyword, best_category, best_score))
            print(f"Added '{keyword}' to '{best_category}' (score: {best_score:.2f})")
        else:
            # If no good match, add to skills_competencies as a default
            if "skills_competencies" in categories:
                categories["skills_competencies"].append(keyword)
                added_keywords.append((keyword, "skills_competencies", 0.0))
                print(f"Added '{keyword}' to 'skills_competencies' (default category)")
            else:
                print(f"No suitable category found for '{keyword}' and no default category available")
    
    return categories, added_keywords

def generate_report(job_id: Optional[str] = None, 
                   job_url: Optional[str] = None, 
                   include_cover_letter: bool = True, 
                   llm_model: str = DEFAULT_LLM_MODEL,
                   keywords: Optional[List[str]] = None,
                   save_keywords: bool = False,
                   min_rating: float = MIN_RATING_THRESHOLD,
                   content_weight: float = DEFAULT_CONTENT_WEIGHT,
                   show_preprocessed_text: bool = False,
                   use_semantic_dedup: bool = False,
                   similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> Optional[str]:
    """Generates a comprehensive job match report for a specified job.
    
    This function creates a detailed report that includes job requirements (tags),
    matching content blocks from the user's content database, content gaps analysis,
    and optionally a cover letter draft generated by an LLM.
    
    Args:
        job_id: ID of the job to analyze. Either job_id or job_url must be provided.
        job_url: URL of the job to analyze. Used as an alternative to job_id.
        include_cover_letter: Whether to include a cover letter draft in the report.
        llm_model: LLM model to use for cover letter generation.
        keywords: Optional list of keywords to prioritize in tag matching.
        save_keywords: Whether to save provided keywords to categories.yaml based on
            semantic similarity.
        min_rating: Minimum rating threshold for content blocks to be included.
        content_weight: Weight to give to content rating when scoring matches
            (higher value prioritizes rating more heavily).
        show_preprocessed_text: Whether to include preprocessed job text in the report.
        use_semantic_dedup: Whether to use semantic similarity for deduplication.
        similarity_threshold: Similarity threshold for semantic deduplication.
        
    Returns:
        Path to the generated report file if successful, None if an error occurred.
        
    Raises:
        FileNotFoundError: If the jobs or content file does not exist.
        json.JSONDecodeError: If any of the data files contain invalid JSON.
    """
    try:
        # Load job data
        with open(DEFAULT_JOBS_FILE, 'r') as f:
            jobs_data = json.load(f)
        
        # Find the job by ID or URL
        job = None
        
        if job_id:
            for j in jobs_data.get("jobs", []):
                if j.get("id") == job_id:
                    job = j
                    break
        elif job_url:
            for j in jobs_data.get("jobs", []):
                if j.get("url") == job_url:
                    job = j
                    break
        
        if not job:
            print("Job not found. Please provide a valid job ID or URL.")
            return None
        
        # Extract job information
        job_title = job.get("job_title", "Unknown Job")
        org_name = job.get("org_name", "Unknown Organization")
        job_url = job.get("url", "")
        job_summary = job.get("summary", "")
        job_text = job.get("raw_text", "")
        
        print(f"Found job: {job_title} at {org_name}")
        
        # Load categories for tag prioritization
        with open(CATEGORIES_FILE, 'r') as f:
            categories = yaml.safe_load(f)
        
        # Save keywords to categories if requested
        added_keywords: List[Tuple[str, str, float]] = []
        if keywords and save_keywords:
            print(f"Adding keywords to categories: {', '.join(keywords)}")
            categories, added_keywords = add_keywords_to_categories(keywords, categories)
            
            # Save updated categories back to file
            if added_keywords:
                print(f"Saving updated categories to {CATEGORIES_FILE}")
                with open(CATEGORIES_FILE, 'w') as f:
                    yaml.dump(categories, f, default_flow_style=False, sort_keys=False)
                print(f"Successfully saved {len(added_keywords)} new keywords to categories")
                
                # Re-tag content blocks with the updated keywords
                print("Re-tagging content blocks with updated keywords...")
                try:
                    # Load content blocks
                    with open(DEFAULT_CONTENT_FILE, 'r') as f:
                        content_data = json.load(f)
                    
                    # Re-tag each content block based on the actual file structure
                    updated_count = 0
                    
                    # Create a flat list of all tags from all categories
                    all_tags = []
                    for category_name, tags in categories.items():
                        all_tags.extend(tags)
                    
                    # Iterate through each document in the content data
                    for doc_key, doc_data in content_data.items():
                        # Skip if doc_data is not a dictionary
                        if not isinstance(doc_data, dict):
                            continue
                        
                        # Skip if content is not present or not a dictionary
                        if "content" not in doc_data or not isinstance(doc_data["content"], dict):
                            continue
                        
                        # Process paragraphs
                        paragraphs = doc_data.get("content", {}).get("paragraphs", [])
                        
                        for paragraph in paragraphs:
                            if not isinstance(paragraph, dict):
                                continue
                                
                            text = paragraph.get("text", "")
                            if text:
                                # Get the top matching tags for this text
                                tag_results = assign_tags_with_spacy(text, categories, max_tags=10)
                                # Extract just the tag names
                                paragraph["tags"] = [item["name"] for item in tag_results if item.get("name")]
                                updated_count += 1
                            
                            # Process sentences within paragraphs
                            if "sentences" in paragraph and isinstance(paragraph["sentences"], list):
                                for sentence in paragraph["sentences"]:
                                    if not isinstance(sentence, dict):
                                        continue
                                        
                                    text = sentence.get("text", "")
                                    if text:
                                        # Get the top matching tags for this text
                                        tag_results = assign_tags_with_spacy(text, categories, max_tags=10)
                                        # Extract just the tag names
                                        sentence["tags"] = [item["name"] for item in tag_results if item.get("name")]
                                        updated_count += 1
                        
                        # Update document tags if present
                        if "document_tags" in doc_data["content"]:
                            # Use the first paragraph text as representative of the document
                            if doc_data["content"]["paragraphs"]:
                                doc_text = " ".join([p.get("text", "") for p in doc_data["content"]["paragraphs"][:3] if isinstance(p, dict)])
                                # Get the top matching tags for this text
                                tag_results = assign_tags_with_spacy(doc_text, categories, max_tags=15)
                                # Extract just the tag names
                                doc_data["content"]["document_tags"] = [item["name"] for item in tag_results if item.get("name")]
                                updated_count += 1
                    
                    # Save updated content blocks
                    with open(DEFAULT_CONTENT_FILE, 'w') as f:
                        json.dump(content_data, f, indent=2)
                    
                    print(f"Successfully re-tagged {updated_count} content blocks with updated keywords")
                except Exception as e:
                    import traceback
                    print(f"Error re-tagging content blocks: {e}")
                    print(traceback.format_exc())
            else:
                print("No new keywords were added (all already exist)")
        
        # Create manual keywords for this run (even if not saving)
        manual_keywords = None
        if keywords:
            manual_keywords = {"high_priority": keywords}
        
        # Generate tags for the job using prioritize_tags_for_job
        job_tags = prioritize_tags_for_job(job_text, categories, manual_keywords)
        
        # Find content blocks from cover letter content data
        with open(DEFAULT_CONTENT_FILE, 'r') as f:
            content_data = json.load(f)
        
        # Extract content blocks with ratings
        cover_letter_data: List[Dict[str, Any]] = []
        
        # Process the content data
        for file_key, file_data in content_data.items():
            if not isinstance(file_data, dict) or "content" not in file_data:
                continue
            
            # Process paragraphs and sentences with ratings
            paragraphs = file_data.get("content", {}).get("paragraphs", [])
            
            for paragraph in paragraphs:
                if "sentences" in paragraph:
                    # Process individual sentences as content blocks
                    for sentence in paragraph.get("sentences", []):
                        if "rating" in sentence and float(sentence.get("rating", 0)) >= min_rating:
                            cover_letter_data.append({
                                "id": f"{file_data.get('id', 'unknown')}-{len(cover_letter_data)}",
                                "content": sentence.get("text", ""),
                                "rating": float(sentence.get("rating", 0)),
                                "tags": sentence.get("tags", [])
                            })
                else:
                    # Use the whole paragraph as a content block
                    cover_letter_data.append({
                        "id": f"{file_data.get('id', 'unknown')}-{len(cover_letter_data)}",
                        "content": paragraph.get("text", ""),
                        "rating": float(paragraph.get("rating", 0)) if "rating" in paragraph else min_rating,
                        "tags": paragraph.get("tags", [])
                    })
        
        # Check if we have missing high-quality content
        missing_content_tags: Dict[str, Set[str]] = {
            "high_priority": set(job_tags.get("high_priority", [])),
            "medium_priority": set(job_tags.get("medium_priority", [])),
            "low_priority": set(job_tags.get("low_priority", []))
        }
        
        # Track which tags are covered by matched blocks
        matched_block_ids: Set[str] = set()
        
        # Calculate the maximum possible score for normalization (10 rating × content_weight)
        max_possible_score = 10.0 * content_weight
        
        # Initialize match lists
        high_priority_matches: List[Dict[str, Any]] = []
        medium_priority_matches: List[Dict[str, Any]] = []
        low_priority_matches: List[Dict[str, Any]] = []
        
        # Process high priority tags
        for tag in job_tags.get("high_priority", []):
            for block in cover_letter_data:
                # Skip already matched blocks
                if block.get("id") in matched_block_ids:
                    continue
                
                # Skip content with low ratings
                rating = float(block.get("rating", 0))
                if rating < min_rating:
                    continue
                
                # Check for match
                content = block.get("content", "")
                tag_match = False
                similarity_score = 0
                
                # Direct text match check (with normalized tag for underscore handling)
                tag_normalized = normalize_text(tag)
                if tag_normalized.lower() in normalize_text(content).lower():
                    tag_match = True
                    similarity_score = 1.0
                else:
                    # Semantic similarity check using normalized text
                    content_normalized = normalize_text(content)
                    content_doc = nlp(content_normalized)
                    tag_doc = nlp(tag_normalized)
                    similarity = safe_similarity(content_doc, tag_doc)
                    if similarity > 0.6:  # Threshold for semantic matching
                        tag_match = True
                        similarity_score = similarity
                
                if tag_match:
                    # Calculate score with higher weight for high priority tags
                    raw_score = similarity_score * (rating / 10) * content_weight
                    score_percentage = min(100, int((raw_score / max_possible_score) * 100))
                    
                    # Multiply percentage by a factor to make scores more meaningful
                    score_percentage = min(100, score_percentage * 3)
                    
                    high_priority_matches.append({
                        "block": block,
                        "raw_score": raw_score,
                        "score_percentage": score_percentage,
                        "tag": tag,
                        "matched_tags": {"high": [tag], "medium": [], "low": []}
                    })
        
        # Sort high priority matches by score
        high_priority_matches.sort(key=lambda x: x["raw_score"], reverse=True)
        
        # Add top high priority matches to final list, avoiding duplicates
        matching_blocks_list: List[Dict[str, Any]] = []
        for match in high_priority_matches:
            if match["block"].get("id") not in matched_block_ids:
                matching_blocks_list.append(match)
                matched_block_ids.add(match["block"].get("id"))
        
        # Process medium priority tags (if we need more content)
        if len(matching_blocks_list) < 5:
            for tag in job_tags.get("medium_priority", []):
                for block in cover_letter_data:
                    # Skip already matched blocks
                    if block.get("id") in matched_block_ids:
                        continue
                    
                    # Skip content with low ratings
                    rating = float(block.get("rating", 0))
                    if rating < min_rating:
                        continue
                    
                    # Check for match
                    content = block.get("content", "")
                    tag_match = False
                    similarity_score = 0
                    
                    # Direct text match check (with normalized tag for underscore handling)
                    tag_normalized = normalize_text(tag)
                    if tag_normalized.lower() in normalize_text(content).lower():
                        tag_match = True
                        similarity_score = 1.0
                    else:
                        # Semantic similarity check using normalized text
                        content_normalized = normalize_text(content)
                        content_doc = nlp(content_normalized)
                        tag_doc = nlp(tag_normalized)
                        similarity = safe_similarity(content_doc, tag_doc)
                        if similarity > 0.6:
                            tag_match = True
                            similarity_score = similarity
                    
                    if tag_match:
                        # Calculate score with medium weight for medium priority tags
                        raw_score = similarity_score * (rating / 10) * content_weight * 0.8
                        score_percentage = min(100, int((raw_score / max_possible_score) * 100))
                        
                        # Multiply percentage by a factor to make scores more meaningful
                        score_percentage = min(100, score_percentage * 3)
                        
                        medium_priority_matches.append({
                            "block": block,
                            "raw_score": raw_score,
                            "score_percentage": score_percentage,
                            "tag": tag,
                            "matched_tags": {"high": [], "medium": [tag], "low": []}
                        })
            
            # Sort medium priority matches by score
            medium_priority_matches.sort(key=lambda x: x["raw_score"], reverse=True)
            
            # Add top medium priority matches to final list
            for match in medium_priority_matches:
                if match["block"].get("id") not in matched_block_ids and len(matching_blocks_list) < 7:
                    matching_blocks_list.append(match)
                    matched_block_ids.add(match["block"].get("id"))
        
        # Process low priority tags (if we still need more content)
        if len(matching_blocks_list) < 5:
            for tag in job_tags.get("low_priority", []):
                for block in cover_letter_data:
                    # Skip already matched blocks
                    if block.get("id") in matched_block_ids:
                        continue
                    
                    # Skip content with low ratings
                    rating = float(block.get("rating", 0))
                    if rating < min_rating:
                        continue
                    
                    # Check for match
                    content = block.get("content", "")
                    tag_match = False
                    similarity_score = 0
                    
                    # Direct text match check (with normalized tag for underscore handling)
                    tag_normalized = normalize_text(tag)
                    if tag_normalized.lower() in normalize_text(content).lower():
                        tag_match = True
                        similarity_score = 1.0
                    else:
                        # Semantic similarity check using normalized text
                        content_normalized = normalize_text(content)
                        content_doc = nlp(content_normalized)
                        tag_doc = nlp(tag_normalized)
                        similarity = safe_similarity(content_doc, tag_doc)
                        if similarity > 0.6:
                            tag_match = True
                            similarity_score = similarity
                    
                    if tag_match:
                        # Calculate score with lower weight for low priority tags
                        raw_score = similarity_score * (rating / 10) * content_weight * 0.6
                        score_percentage = min(100, int((raw_score / max_possible_score) * 100))
                        
                        # Multiply percentage by a factor to make scores more meaningful
                        score_percentage = min(100, score_percentage * 3)
                        
                        low_priority_matches.append({
                            "block": block,
                            "raw_score": raw_score,
                            "score_percentage": score_percentage,
                            "tag": tag,
                            "matched_tags": {"high": [], "medium": [], "low": [tag]}
                        })
            
            # Sort low priority matches by score
            low_priority_matches.sort(key=lambda x: x["raw_score"], reverse=True)
            
            # Add top low priority matches to final list
            for match in low_priority_matches:
                if match["block"].get("id") not in matched_block_ids and len(matching_blocks_list) < 7:
                    matching_blocks_list.append(match)
                    matched_block_ids.add(match["block"].get("id"))
        
        # Combine all matches with others that match the same block
        for i, match in enumerate(matching_blocks_list):
            block_id = match["block"].get("id")
            
            # Add any other matching tags from high priority
            for hp_match in high_priority_matches:
                if hp_match["block"].get("id") == block_id and hp_match["tag"] not in match["matched_tags"]["high"]:
                    match["matched_tags"]["high"].append(hp_match["tag"])
            
            # Add any other matching tags from medium priority
            for mp_match in medium_priority_matches:
                if mp_match["block"].get("id") == block_id and mp_match["tag"] not in match["matched_tags"]["medium"]:
                    match["matched_tags"]["medium"].append(mp_match["tag"])
            
            # Add any other matching tags from low priority
            for lp_match in low_priority_matches:
                if lp_match["block"].get("id") == block_id and lp_match["tag"] not in match["matched_tags"]["low"]:
                    match["matched_tags"]["low"].append(lp_match["tag"])
        
        # Sort final list by score
        matching_blocks_list.sort(key=lambda x: x["raw_score"], reverse=True)
        
        # Deduplicate blocks by content while preserving tag matches
        deduplicated_blocks: Dict[str, Dict[str, Any]] = {}
        
        if use_semantic_dedup and len(matching_blocks_list) > 1:
            # Extract blocks for similarity analysis
            blocks_for_analysis = []
            for match in matching_blocks_list:
                block_content = match["block"].get("content", "")
                blocks_for_analysis.append({
                    "text": block_content,
                    "id": match["block"].get("id", ""),
                    "original_match": match
                })
            
            # Get similarity map
            similarity_map = analyze_content_block_similarity(blocks_for_analysis)
            
            # Process blocks in order of score (highest first)
            processed_ids = set()
            
            for match in matching_blocks_list:
                block_id = match["block"].get("id", "")
                content = match["block"].get("content", "")
                
                # Skip if this block has already been processed as a duplicate
                if block_id and block_id in processed_ids:
                    continue
                
                # Add this block to deduplicated blocks
                if content not in deduplicated_blocks:
                    deduplicated_blocks[content] = match.copy()
                    deduplicated_blocks[content]["similar_blocks"] = []
                    
                    # Find similar blocks
                    if content in similarity_map:
                        similar_blocks = similarity_map[content]
                        for similar in similar_blocks:
                            similar_text = similar["text"]
                            similarity = similar["similarity"]
                            
                            if similarity >= similarity_threshold:
                                # Find the original match for this text
                                similar_match = None
                                for m in matching_blocks_list:
                                    if m["block"].get("content", "") == similar_text:
                                        similar_match = m
                                        break
                                
                                if similar_match:
                                    similar_id = similar_match["block"].get("id", "")
                                    if similar_id:
                                        processed_ids.add(similar_id)
                                    
                                    # Add to similar blocks
                                    deduplicated_blocks[content]["similar_blocks"].append({
                                        "content": similar_text,
                                        "similarity": similarity,
                                        "id": similar_id,
                                        "match": similar_match
                                    })
                                    
                                    # Merge tag matches
                                    for priority in ["high", "medium", "low"]:
                                        for tag in similar_match["matched_tags"][priority]:
                                            if tag not in deduplicated_blocks[content]["matched_tags"][priority]:
                                                deduplicated_blocks[content]["matched_tags"][priority].append(tag)
                                    
                                    # Keep the highest score
                                    if similar_match["raw_score"] > deduplicated_blocks[content]["raw_score"]:
                                        deduplicated_blocks[content]["raw_score"] = similar_match["raw_score"]
                                        deduplicated_blocks[content]["score_percentage"] = similar_match["score_percentage"]
        else:
            # Use exact text matching for deduplication (original method)
            for match in matching_blocks_list:
                content = match["block"].get("content", "")
                if content in deduplicated_blocks:
                    # Merge tag matches from this duplicate into the existing entry
                    for priority in ["high", "medium", "low"]:
                        for tag in match["matched_tags"][priority]:
                            if tag not in deduplicated_blocks[content]["matched_tags"][priority]:
                                deduplicated_blocks[content]["matched_tags"][priority].append(tag)
                    
                    # Keep the highest score
                    if match["raw_score"] > deduplicated_blocks[content]["raw_score"]:
                        deduplicated_blocks[content]["raw_score"] = match["raw_score"]
                        deduplicated_blocks[content]["score_percentage"] = match["score_percentage"]
                else:
                    # Add new unique content block
                    deduplicated_blocks[content] = match
        
        # Convert back to list
        matching_blocks_list = list(deduplicated_blocks.values())
        
        # Re-sort after deduplication
        matching_blocks_list.sort(key=lambda x: x["raw_score"], reverse=True)
        
        # Prepare report
        report: List[str] = []
        
        # Add report header
        report.append(f"# Job Match Report: {job_title}")
        report.append("")
        report.append(f"**Company:** {org_name}")
        report.append(f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        if job_url:
            report.append(f"**Job URL:** [{job_url}]({job_url})")
        report.append("")
        
        # Add job summary
        report.append("## Job Summary")
        report.append("")
        report.append(job_summary)
        report.append("")
        
        # Add preprocessed job text section
        if show_preprocessed_text:
            processed_text, filtered_text = preprocess_job_text(job_text)
            
            report.append("## Preprocessed Job Text")
            report.append("")
            report.append("The following text was used for tag analysis after preprocessing:")
            report.append("")
            report.append("```")
            # Show full preprocessed text without truncation
            report.append(processed_text)
            report.append("```")
            report.append("")
        
        # Add job requirements (tags)
        report.append("## Job Requirements (Tags)")
        report.append("")
        
        # High priority tags
        report.append("### High Priority")
        report.append("")
        if job_tags.get("high_priority"):
            for tag in job_tags["high_priority"]:
                report.append(f"- {tag}")
        else:
            report.append("- None")
        report.append("")
        
        # Medium priority tags
        report.append("### Medium Priority")
        report.append("")
        if job_tags.get("medium_priority"):
            for tag in job_tags["medium_priority"]:
                report.append(f"- {tag}")
        else:
            report.append("- None")
        report.append("")
        
        # Low priority tags
        report.append("### Low Priority")
        report.append("")
        if job_tags.get("low_priority"):
            for tag in job_tags["low_priority"]:
                report.append(f"- {tag}")
        else:
            report.append("- None")
        report.append("")
        
        # Add manual keywords section if keywords were provided
        if keywords:
            report.append("## Manual Keywords")
            report.append("")
            report.append(", ".join(keywords))
            
            # Show which categories keywords were added to
            if added_keywords:
                report.append("")
                report.append("**Keywords added to categories:**")
                for kw, cat, score in added_keywords:
                    report.append(f"- '{kw}' added to '{cat}' (score: {score:.2f})")
            
            report.append("")
        
        # Add content gaps section if needed
        for block in matching_blocks_list:
            for tag in block["matched_tags"]["high"]:
                missing_content_tags["high_priority"].discard(tag)
            for tag in block["matched_tags"]["medium"]:
                missing_content_tags["medium_priority"].discard(tag)
            for tag in block["matched_tags"]["low"]:
                missing_content_tags["low_priority"].discard(tag)
        
        if any(missing_content_tags.values()):
            report.append("## Content Gaps")
            report.append("")
            report.append("The following tags don't have matching high-quality content (rated 8+). Consider improving or adding content for these topics:")
            report.append("")
            
            if missing_content_tags["high_priority"]:
                report.append("### High Priority Gaps")
                for tag in sorted(missing_content_tags["high_priority"]):
                    report.append(f"- {tag}")
                report.append("")
            
            if missing_content_tags["medium_priority"]:
                report.append("### Medium Priority Gaps")
                for tag in sorted(missing_content_tags["medium_priority"]):
                    report.append(f"- {tag}")
                report.append("")
            
            if missing_content_tags["low_priority"]:
                report.append("### Low Priority Gaps")
                for tag in sorted(missing_content_tags["low_priority"]):
                    report.append(f"- {tag}")
                report.append("")
        
        # Add matching content blocks section
        if matching_blocks_list:
            report.append("## Matching Content Blocks")
            report.append("")
            report.append("These content blocks from your database match the job requirements:")
            report.append("")
            
            for i, match in enumerate(matching_blocks_list[:10], 1):  # Limit to top 10
                # Calculate tag counts
                high_count = len(match["matched_tags"]["high"])
                medium_count = len(match["matched_tags"]["medium"])
                low_count = len(match["matched_tags"]["low"])
                total_tags = high_count + medium_count + low_count
                
                # Format matched tags by priority
                tag_sections = []
                if high_count > 0:
                    tag_sections.append(f"**High Priority ({high_count}):** {', '.join(match['matched_tags']['high'])}")
                if medium_count > 0:
                    tag_sections.append(f"**Medium Priority ({medium_count}):** {', '.join(match['matched_tags']['medium'])}")
                if low_count > 0:
                    tag_sections.append(f"**Low Priority ({low_count}):** {', '.join(match['matched_tags']['low'])}")
                
                # Add block to report
                report.append(f"### {i}. Match Score: {match['score_percentage']:.0f}% ({total_tags} tags)")
                report.append("")
                report.append(f"> {match['block'].get('content', '')}")
                report.append("")
                report.append(f"**Rating:** {match['block'].get('rating', 0):.1f}/10")
                report.append("")
                
                # Add matched tags
                report.append("**Matched Tags:**")
                for section in tag_sections:
                    report.append(section)
                    report.append("")
                
                # Add similar blocks if available (when using semantic deduplication)
                if use_semantic_dedup and "similar_blocks" in match and match["similar_blocks"]:
                    report.append("**Similar Blocks:**")
                    report.append("")
                    
                    for j, similar in enumerate(match["similar_blocks"], 1):
                        similar_content = similar.get("content", "")
                        similarity = similar.get("similarity", 0)
                        similar_id = similar.get("id", "N/A")
                        similar_match = similar.get("match", {})
                        similar_rating = similar_match.get("block", {}).get("rating", 0)
                        
                        report.append(f"#### Similar Block {j} (Similarity: {similarity:.1%} | Rating: {similar_rating:.1f} | ID: {similar_id})")
                        report.append("")
                        report.append(f"> {similar_content}")
                        report.append("")
                        
                        # Show differences
                        diff1, diff2 = display_text_differences(match["block"].get("content", ""), similar_content)
                        report.append("**Differences:**")
                        report.append("```")
                        report.append(f"Original: {diff1}")
                        report.append(f"Similar:  {diff2}")
                        report.append("```")
                        report.append("")
                
                report.append("---")
                report.append("")
        
        # Generate cover letter
        report.append("## Cover Letter Draft")
        report.append("")
        
        if include_cover_letter:
            print("Generating cover letter using LLM...")
            
            # Prepare prompt for LLM
            prompt = f"""
You are a professional cover letter writer. Create a cover letter for a job application based on the following information:

Job Title: {job_title}
Company: {org_name}
Job Summary: {job_summary}

Use the following content blocks as the basis for the cover letter. These are high-quality, pre-written paragraphs that should be adapted and integrated into a cohesive letter:

"""
            
            # Add top matches to the prompt
            for i, block in enumerate(matching_blocks_list[:8], 1):
                prompt += f"{i}. \"{block['block']['content']}\"\n\n"
                
            prompt += """
Guidelines:
1. Write in a professional, confident tone
2. Maintain a conversational style
3. Adapt and integrate the provided content blocks naturally
4. Keep the letter concise (300-400 words)
5. Include a standard cover letter format with placeholders for contact information
6. Focus on how the candidate's experience aligns with the job requirements
7. End with a call to action

Create a complete cover letter that feels cohesive and tailored to this specific job.
"""
            
            # Generate cover letter using Ollama
            try:
                response = ollama.generate(
                    model=llm_model,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40
                    }
                )
                
                cover_letter = response.get("response", "")
                
                # Add cover letter to report
                report.append(cover_letter)
                
            except Exception as e:
                report.append("Error generating cover letter:")
                report.append(str(e))
                report.append("")
                report.append("Please try again with a different LLM model.")
        else:
            report.append("Cover letter generation was skipped.")
        
        # Create reports directory if it doesn't exist
        os.makedirs(REPORTS_DIR, exist_ok=True)
        
        # Generate filename based on company name
        company_slug = org_name.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(REPORTS_DIR, f"{company_slug}_report_{timestamp}.md")
        
        # Write report to file
        report_content = "\n".join(report)
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Print preview
        print(f"Report generated successfully and saved to {report_file}")
        print("\n" + "=" * 80)
        print("REPORT PREVIEW")
        print("=" * 80)
        
        # Show first 20 lines of the report
        preview_lines = report[:20]
        print("\n".join(preview_lines))
        print("...")
        print(f"\nFull report saved to {report_file}")
        
        return report_file
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None

def setup_argparse(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Sets up argument parser for the report generator CLI.
    
    Args:
        parser: Optional pre-existing ArgumentParser instance. If None, a new 
            parser will be created.
            
    Returns:
        argparse.ArgumentParser: Configured argument parser ready for parsing arguments.
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Generate job match report")
    
    # Job identification (mutually exclusive)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id", type=str, help="ID of the job to analyze")
    group.add_argument("--job-url", type=str, help="URL of the job to analyze")
    
    # Cover letter options
    parser.add_argument("--no-cover-letter", action="store_true", 
                       help="Skip cover letter generation")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL,
                       help=f"LLM model to use for cover letter generation (default: {DEFAULT_LLM_MODEL})")
    
    # Content matching options
    parser.add_argument("--min-rating", type=float, default=MIN_RATING_THRESHOLD,
                       help=f"Minimum rating threshold for content blocks (default: {MIN_RATING_THRESHOLD})")
    parser.add_argument("--content-weight", type=float, default=DEFAULT_CONTENT_WEIGHT,
                       help=f"Weight to give content rating vs tag matching (default: {DEFAULT_CONTENT_WEIGHT})")
    
    # Keyword options
    parser.add_argument("--keywords", type=str, nargs="+",
                       help="List of keywords to prioritize in tag matching")
    parser.add_argument("--save-keywords", action="store_true",
                       help="Save provided keywords to categories.yaml based on semantic similarity")
    
    # Display options
    parser.add_argument("--show-preprocessed", action="store_true",
                       help="Include preprocessed job text in the report")
    
    # Deduplication options
    parser.add_argument("--semantic-dedup", action="store_true",
                       help="Use semantic similarity for content deduplication")
    parser.add_argument("--similarity-threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD,
                       help=f"Similarity threshold for semantic deduplication (default: {DEFAULT_SIMILARITY_THRESHOLD})")
    
    return parser

def main(args: Optional[argparse.Namespace] = None) -> None:
    """Runs the report generator CLI with the provided arguments.
    
    This function handles the main execution flow of the report generator CLI,
    processing command-line arguments and generating a job match report based
    on the specified criteria.
    
    Args:
        args: Optional pre-parsed command line arguments. If None, arguments 
            will be parsed from sys.argv.
            
    Returns:
        None
    """
    if args is None:
        parser = setup_argparse()
        args = parser.parse_args()
    
    # Generate the report
    report_file = generate_report(
        job_id=args.job_id,
        job_url=args.job_url,
        include_cover_letter=not args.no_cover_letter,
        llm_model=args.llm_model,
        keywords=args.keywords,
        save_keywords=args.save_keywords,
        min_rating=args.min_rating,
        content_weight=args.content_weight,
        show_preprocessed_text=args.show_preprocessed,
        use_semantic_dedup=args.semantic_dedup,
        similarity_threshold=args.similarity_threshold
    )
    
    if report_file:
        print(f"Report generated successfully: {report_file}")
    else:
        print("Failed to generate report.")
        sys.exit(1)

if __name__ == "__main__":
    main()
