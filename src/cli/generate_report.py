#!/usr/bin/env python3
"""
Generate Report CLI - Generate job match reports with content matching and cover letters.

This module provides a command-line interface for generating comprehensive
job match reports with tag prioritization, content matching, and cover letter generation.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import time
import yaml

# Add parent directory to path to allow imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR, DEFAULT_LLM_MODEL
import ollama

# Constants
DEFAULT_JOBS_FILE = os.path.join(DATA_DIR, "json/analyzed_jobs.json")
DEFAULT_CONTENT_FILE = os.path.join(DATA_DIR, "json/cover_letter_content.json")
CATEGORIES_FILE = os.path.join(DATA_DIR, "config/categories.yaml")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
MIN_RATING_THRESHOLD = 8.0  # Minimum rating to consider a content block
DEFAULT_CONTENT_WEIGHT = 2.0  # Default weight to give to content rating

# Import locally to avoid circular imports
from src.utils.spacy_utils import prioritize_tags_for_job, nlp, safe_similarity, has_vector, normalize_text, assign_tags_with_spacy, preprocess_job_text

def add_keywords_to_categories(keywords, categories):
    """
    Add new keywords to appropriate categories based on semantic similarity.
    
    Args:
        keywords: List of keywords to add
        categories: Dictionary of category -> keywords
        
    Returns:
        Updated categories dict, list of added keywords
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

def generate_report(job_id: str = None, job_url: str = None, 
                   include_cover_letter: bool = True, 
                   llm_model: str = DEFAULT_LLM_MODEL,
                   keywords: list = None,
                   save_keywords: bool = False,
                   min_rating: float = MIN_RATING_THRESHOLD,
                   content_weight: float = DEFAULT_CONTENT_WEIGHT,
                   show_preprocessed_text: bool = False) -> str:
    """
    Generate a job match report.
    
    Args:
        job_id: ID of the job to analyze
        job_url: URL of the job to analyze (alternative to job_id)
        include_cover_letter: Whether to include a cover letter draft
        llm_model: LLM model to use for generation
        keywords: Optional list of keywords to prioritize
        save_keywords: Whether to save keywords to categories.yaml
        min_rating: Minimum rating threshold for content blocks
        content_weight: Weight to give to content rating (higher = prioritize rating more)
        show_preprocessed_text: Whether to include preprocessed job text in the report
        
    Returns:
        Path to the generated report
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
        added_keywords = []
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
                        
                        # Skip if paragraphs is not present or not a list
                        if "paragraphs" not in doc_data["content"] or not isinstance(doc_data["content"]["paragraphs"], list):
                            continue
                        
                        # Process paragraphs
                        for paragraph in doc_data["content"]["paragraphs"]:
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
        cover_letter_data = []
        
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
        missing_content_tags = {
            "high_priority": set(job_tags.get("high_priority", [])),
            "medium_priority": set(job_tags.get("medium_priority", [])),
            "low_priority": set(job_tags.get("low_priority", []))
        }
        
        # Track which tags are covered by matched blocks
        matched_block_ids = set()
        
        # Calculate the maximum possible score for normalization (10 rating × content_weight)
        max_possible_score = 10.0 * content_weight
        
        # Initialize match lists
        high_priority_matches = []
        medium_priority_matches = []
        low_priority_matches = []
        
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
        matching_blocks_list = []
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
        deduplicated_blocks = {}
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
        report = []
        
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
        
        # Add content blocks to report
        for i, block in enumerate(matching_blocks_list, 1):
            content = block["block"].get("content", "")
            rating = float(block["block"].get("rating", 0))
            
            report.append(f"### {i}. Match Score: {block['score_percentage']}%, Rating: {rating}")
            report.append("")
            report.append(f"> {content}")
            report.append("")
            report.append("**Matched Tags:**")
            
            if block['matched_tags']['high']:
                report.append("- *High Priority:* " + ", ".join(block['matched_tags']['high']))
            if block['matched_tags']['medium']:
                report.append("- *Medium Priority:* " + ", ".join(block['matched_tags']['medium']))
            if block['matched_tags']['low']:
                report.append("- *Low Priority:* " + ", ".join(block['matched_tags']['low']))
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

def setup_argparse(parser=None):
    """Set up argument parser for the CLI."""
    if parser is None:
        parser = argparse.ArgumentParser(description="Generate a job match report")
    
    # Job selection options
    parser.add_argument("--job-id", help="ID of the job to analyze")
    parser.add_argument("--job-url", help="URL of the job to analyze")
    
    # Report options
    parser.add_argument("--no-cover-letter", action="store_true", help="Don't include a cover letter draft")
    parser.add_argument("--model", help=f"LLM model to use (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--show-preprocessed-text", action="store_true", help="Include preprocessed job text in the report")
    
    # Keyword options
    parser.add_argument("--keywords", help="Comma-separated list of keywords to prioritize")
    parser.add_argument("--save-keywords", action="store_true", 
                      help="Save provided keywords to categories.yaml based on semantic similarity")
    
    # Matching options
    parser.add_argument("--min-rating", type=float, default=MIN_RATING_THRESHOLD, 
                        help=f"Minimum rating threshold for content blocks (default: {MIN_RATING_THRESHOLD})")
    parser.add_argument("--content-weight", type=float, default=DEFAULT_CONTENT_WEIGHT,
                        help=f"Weight to give to content rating (default: {DEFAULT_CONTENT_WEIGHT})")
    
    return parser

def main(args=None):
    """Main function to run the report generator."""
    if args is None:
        parser = setup_argparse()
        args = parser.parse_args()
    
    if not args.job_id and not args.job_url:
        # Default to the first job if no ID or URL is provided
        args.job_id = "1"
    
    # Parse keywords
    keywords = None
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",")]
    
    generate_report(
        job_id=args.job_id,
        job_url=args.job_url,
        include_cover_letter=not args.no_cover_letter,
        llm_model=args.model or DEFAULT_LLM_MODEL,
        keywords=keywords,
        save_keywords=args.save_keywords,
        min_rating=args.min_rating,
        content_weight=args.content_weight,
        show_preprocessed_text=args.show_preprocessed_text
    )

if __name__ == "__main__":
    main()
