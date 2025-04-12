#!/usr/bin/env python3
"""
Version bumping utility for coverletter_wiz.

Usage:
    python scripts/bump_version.py (major|minor|patch) ["Changelog message"]
"""

import os
import re
import sys
import datetime
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
VERSION_FILE = PROJECT_ROOT / "src" / "__init__.py"
CHANGELOG_FILE = PROJECT_ROOT / "CHANGELOG.md"


def get_current_version():
    """Extract the current version from __init__.py."""
    with open(VERSION_FILE, "r") as f:
        content = f.read()
    
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        print("Error: Could not find version string in __init__.py")
        sys.exit(1)
    
    return match.group(1)


def update_version_file(new_version):
    """Update the version in __init__.py."""
    with open(VERSION_FILE, "r") as f:
        content = f.read()
    
    updated_content = re.sub(
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        f'__version__ = "{new_version}"',
        content
    )
    
    with open(VERSION_FILE, "w") as f:
        f.write(updated_content)


def bump_version(current_version, bump_type):
    """Bump the version according to semver rules."""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        print(f"Error: Unknown bump type '{bump_type}'. Use 'major', 'minor', or 'patch'.")
        sys.exit(1)


def update_changelog(new_version, message=None):
    """Update the CHANGELOG.md file with the new version."""
    with open(CHANGELOG_FILE, "r") as f:
        content = f.readlines()
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    unreleased_index = None
    
    # Find the [Unreleased] section
    for i, line in enumerate(content):
        if line.startswith("## [Unreleased]"):
            unreleased_index = i
            break
    
    if unreleased_index is None:
        print("Error: Could not find [Unreleased] section in CHANGELOG.md")
        sys.exit(1)
    
    # Check if there are any changes in the Unreleased section
    has_changes = False
    for i in range(unreleased_index + 1, len(content)):
        if content[i].startswith("## ["):
            break
        if content[i].startswith("### "):
            has_changes = True
    
    # Add new message to [Unreleased] if provided
    if message:
        # Find the appropriate section (default to Added)
        section_found = False
        for i in range(unreleased_index + 1, len(content)):
            if content[i].startswith("## ["):
                break
            if content[i].startswith("### Added"):
                content.insert(i + 1, f"- {message}\n")
                section_found = True
                has_changes = True
                break
        
        # If no Added section was found, create one
        if not section_found:
            content.insert(unreleased_index + 1, "\n### Added\n")
            content.insert(unreleased_index + 2, f"- {message}\n\n")
            has_changes = True
    
    # If there are no changes and no message, add a placeholder
    if not has_changes and not message:
        print("Warning: No changes found in [Unreleased] section and no message provided.")
        print("Adding a placeholder message.")
        content.insert(unreleased_index + 1, "\n### Changed\n")
        content.insert(unreleased_index + 2, f"- Version bump to {new_version}\n\n")
    
    # Create new version section from [Unreleased]
    new_section_lines = []
    in_unreleased = False
    unreleased_content = []
    
    for line in content:
        if line.startswith("## [Unreleased]"):
            in_unreleased = True
        elif in_unreleased and line.startswith("## ["):
            in_unreleased = False
        
        if in_unreleased and not line.startswith("## [Unreleased]"):
            unreleased_content.append(line)
    
    # Create the new version section
    new_section = [
        f"## [{new_version}] - {today}\n",
    ]
    new_section.extend(unreleased_content)
    
    # Reset the [Unreleased] section
    unreleased_reset = [
        "## [Unreleased]\n",
        "\n",
        "### Added\n",
        "- Future changes will be listed here before release\n",
        "\n"
    ]
    
    # Insert the new version section after [Unreleased]
    new_content = []
    for i, line in enumerate(content):
        new_content.append(line)
        if line.startswith("## [Unreleased]"):
            # Skip until next version section
            for j in range(i + 1, len(content)):
                if content[j].startswith("## ["):
                    break
            else:
                j = len(content)
            
            # Replace with reset content + new section
            new_content = new_content[:i+1]  # Include [Unreleased] line
            new_content.extend(unreleased_reset)
            new_content.extend(new_section)
            new_content.extend(content[j:])
            break
    
    # Update the compare links at the bottom
    link_pattern = re.compile(r'\[Unreleased\]: .*compare/v([\d.]+)\.\.\.HEAD')
    for i, line in enumerate(new_content):
        if link_pattern.match(line):
            # Update Unreleased link
            new_content[i] = f"[Unreleased]: https://github.com/yourusername/coverletter_wiz/compare/v{new_version}...HEAD\n"
            # Add new version link if not exists
            next_line = new_content[i+1] if i+1 < len(new_content) else ""
            if not next_line.startswith(f"[{new_version}]"):
                prev_version = link_pattern.match(line).group(1)
                version_link = f"[{new_version}]: https://github.com/yourusername/coverletter_wiz/compare/v{prev_version}...v{new_version}\n"
                new_content.insert(i+1, version_link)
            break
    
    # Write the updated content back to the file
    with open(CHANGELOG_FILE, "w") as f:
        f.writelines(new_content)


def main():
    if len(sys.argv) < 2:
        print("Usage: python bump_version.py (major|minor|patch) [\"Changelog message\"]")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    message = sys.argv[2] if len(sys.argv) > 2 else None
    
    if bump_type not in ["major", "minor", "patch"]:
        print(f"Error: Unknown bump type '{bump_type}'. Use 'major', 'minor', or 'patch'.")
        sys.exit(1)
    
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"Bumping version: {current_version} -> {new_version}")
    
    update_version_file(new_version)
    update_changelog(new_version, message)
    
    print(f"Updated version to {new_version} in __init__.py")
    print(f"Updated CHANGELOG.md with new version {new_version}")
    print("\nNext steps:")
    print(f"1. Run tests: python -m unittest discover tests")
    print(f"2. Review changes to CHANGELOG.md")
    print(f"3. Commit changes: git commit -am \"Bump version to {new_version}\"")
    print(f"4. Tag the release: git tag -a v{new_version} -m \"Release v{new_version}\"")
    print(f"5. Push changes: git push && git push --tags")


if __name__ == "__main__":
    main()
