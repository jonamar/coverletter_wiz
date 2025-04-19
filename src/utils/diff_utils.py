"""Utilities for visualizing differences between content blocks.

This module provides functions to highlight and display differences between
similar content blocks to help users make better rating decisions.
"""

import difflib
from typing import List, Tuple


def get_text_differences(text1: str, text2: str) -> Tuple[List[str], List[str]]:
    """Get highlighted differences between two text blocks.
    
    Args:
        text1: First text block
        text2: Second text block
        
    Returns:
        Tuple containing:
            - List of strings with differences highlighted for text1
            - List of strings with differences highlighted for text2
    """
    # Split texts into words
    words1 = text1.split()
    words2 = text2.split()
    
    # Get the differences using difflib
    diff = difflib.ndiff(words1, words2)
    
    # Process the differences
    highlighted1 = []
    highlighted2 = []
    
    for line in diff:
        if line.startswith('- '):
            # Word only in text1
            highlighted1.append(f"\033[91m{line[2:]}\033[0m")  # Red
        elif line.startswith('+ '):
            # Word only in text2
            highlighted2.append(f"\033[92m{line[2:]}\033[0m")  # Green
        elif line.startswith('  '):
            # Word in both texts
            highlighted1.append(line[2:])
            highlighted2.append(line[2:])
        elif line.startswith('? '):
            # Indicator line, skip
            continue
    
    return highlighted1, highlighted2


def display_text_differences(text1: str, text2: str) -> Tuple[str, str]:
    """Display differences between two text blocks with highlighting.
    
    Args:
        text1: First text block
        text2: Second text block
        
    Returns:
        Tuple containing:
            - String with differences highlighted for text1
            - String with differences highlighted for text2
    """
    highlighted1, highlighted2 = get_text_differences(text1, text2)
    
    return ' '.join(highlighted1), ' '.join(highlighted2)


def display_markdown_differences(text1: str, text2: str) -> Tuple[str, str]:
    """Display differences between two text blocks with Markdown-compatible highlighting.
    
    Args:
        text1: First text block
        text2: Second text block
        
    Returns:
        Tuple containing:
            - String with differences highlighted for text1 using Markdown
            - String with differences highlighted for text2 using Markdown
    """
    # Split texts into words
    words1 = text1.split()
    words2 = text2.split()
    
    # Get the differences using difflib
    diff = difflib.ndiff(words1, words2)
    
    # Process the differences
    highlighted1 = []
    highlighted2 = []
    
    for line in diff:
        if line.startswith('- '):
            # Word only in text1
            highlighted1.append(f"**{line[2:]}**")  # Bold for text1 differences
        elif line.startswith('+ '):
            # Word only in text2
            highlighted2.append(f"**{line[2:]}**")  # Bold for text2 differences
        elif line.startswith('  '):
            # Word in both texts
            highlighted1.append(line[2:])
            highlighted2.append(line[2:])
        elif line.startswith('? '):
            # Indicator line, skip
            continue
    
    return ' '.join(highlighted1), ' '.join(highlighted2)


def print_text_differences(text1: str, text2: str) -> None:
    """Print differences between two text blocks with highlighting.
    
    Args:
        text1: First text block
        text2: Second text block
    """
    highlighted1, highlighted2 = display_text_differences(text1, text2)
    
    print("Text 1 (with differences):")
    print(highlighted1)
    print("\nText 2 (with differences):")
    print(highlighted2)
