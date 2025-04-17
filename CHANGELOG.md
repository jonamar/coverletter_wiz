# Changelog

All notable changes to the coverletter_wiz project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Future changes will be listed here before release

## [0.1.3] - 2025-04-17

### Changed
- Enhanced documentation across all core and utility modules
- Added comprehensive type annotations to improve code quality
- Updated config.py with improved type safety and documentation

### Improved
- Implemented Google-style docstrings in core and utility files
- Maintained existing functionality while improving code readability

## [1.0.0] - 2025-04-17

### Changed
- Removed legacy CLI script (`coverletter_wiz.py`) as all functionality has been migrated to the unified CLI
- Updated README to clarify that the unified CLI is now the only supported interface
- Completed the migration of all functionality to the new unified CLI structure

### Breaking Changes
- The legacy CLI script (`coverletter_wiz.py`) is no longer available
- All commands must now be run through the unified CLI (`./coverletter`)

## [0.3.0] - 2025-04-16

### Added
- Added `rate` command to unified CLI for managing content ratings
- Integrated comprehensive rating functionality including batch rating, tournament mode, and legends tournament
- Added ability to view content statistics through the unified CLI

### Changed
- Updated README with examples for the rate command and its options
- Maintained text editing capabilities during rating process

## [0.2.7] - 2025-04-16

### Added
- Added `process` command to unified CLI for processing cover letters from text-archive
- Integrated text processing functionality into the unified CLI structure

### Changed
- Updated text processing to use en_core_web_lg spaCy model by default for better NLP performance
- Improved CLI help documentation with examples for the process command
- Removed deprecated process command from legacy CLI while preserving other functionality

## [0.2.6] - 2025-04-13

### Added
- Made preprocessed job text in reports optional with new --show-preprocessed-text flag
- Added special handling for high-value product management tags

### Changed
- Improved tag prioritization with special rules for domain-specific tags
- Ensured "prioritization" tag is always at least medium priority
- Adjusted tag scoring weights for better relevance

## [0.2.5] - 2025-04-13

### Added
- Added preprocessed job text section to reports for transparency
- Enhanced job text preprocessing with better filtering of company descriptions and legal text

### Changed
- Further improved boilerplate detection for organization descriptions
- Made text preprocessing more selective to focus on core job requirements

## [0.2.4] - 2025-04-13

### Added
- Added job text preprocessing to filter out boilerplate and legal text
- Improved tag prioritization with frequency-based scoring

### Changed
- Cleaned up categories.yaml to remove redundant tags
- Made tag prioritization more selective with higher thresholds
- Reduced the number of tags in each priority level for more focused results

## [0.2.3] - 2025-04-13

### Fixed
- Fixed issue with duplicate content blocks appearing in generated reports
- Improved content block deduplication while preserving all tag matches

## [0.2.2] - 2025-04-13

### Added
- Improved cover letter generation prompt to preserve original phrasing
- Enhanced keyword management with automatic content re-tagging

### Changed
- Updated cover letter generation to avoid content duplication
- Improved error handling in content re-tagging functionality
- Enhanced type checking for more robust content processing

### Fixed
- Fixed issue with content re-tagging when new keywords are added
- Improved handling of content blocks structure in re-tagging process

## [0.2.1] - 2025-04-13

### Added
- Improved keyword management system with better semantic matching
- Added "strategic_execution" tag to skills_competencies category
- Automatic re-tagging of content blocks when new keywords are added

### Changed
- Enhanced keyword saving functionality with better feedback and fallback options
- Removed unused category_expansions.yaml file and references
- Updated documentation to reflect keyword management improvements

### Fixed
- Fixed issue with keywords not being properly saved to categories.yaml
- Improved semantic matching for compound keywords (e.g., strategic_execution)
- Enhanced error handling in keyword addition process

## [0.2.0] - 2025-04-13

### Added
- Unified CLI structure with single entry point (`./coverletter`)
- Subcommands for all major functionality (analyze, match, export, report)
- Improved semantic matching for keywords and content blocks
- Content gaps section in job reports to identify missing high-quality content
- Flexible keyword injection with semantic category matching
- CLI support for adding and saving keywords

### Changed
- Moved standalone scripts to organized CLI module structure
- Refactored report generation with improved content prioritization
- Enhanced semantic matching with normalized text handling
- Updated documentation to reflect new CLI structure
- Improved handling of underscore-separated keywords

### Fixed
- Resolved issues with empty vector warnings in spaCy similarity calculations
- Fixed data privacy concerns by ensuring all personal data is stored in external data repository

## [0.1.2] - 2025-04-12

### Changed
- Moved .windsurfrules file to root directory for improved visibility
- Consolidated multiple requirements.txt files into a single file
- Added environment_info.txt for better environment documentation
- Created activation script (activate_env.sh) for easier environment setup

### Fixed
- Resolved environment visibility issues for development tools
- Updated dependencies to latest compatible versions

## [0.1.1] - 2025-04-12

### Fixed
- Fixed content block rating preservation when editing blocks in tournament mode
- Ensured consistent edit behavior across all rating modes (tournament, legends, batch)
- Updated tests to verify rating preservation during content editing

## [0.1.0] - 2025-04-12

### Added
- Initial release of coverletter_wiz
- NLP capabilities using spaCy with en_core_web_lg model
- External data storage structure in separate repository
- Cover letter content matching and analysis functionality
- CLI interface for job matching and content generation
- Configuration system with separate data directory

### Changed
- Moved data storage to external directory for privacy
- Updated data paths to reference external data location

[Unreleased]: https://github.com/yourusername/coverletter_wiz/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/coverletter_wiz/compare/v0.3.0...v1.0.0
[0.3.0]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.7...v0.3.0
[0.2.7]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/yourusername/coverletter_wiz/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/yourusername/coverletter_wiz/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/yourusername/coverletter_wiz/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yourusername/coverletter_wiz/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yourusername/coverletter_wiz/releases/tag/v0.1.0
