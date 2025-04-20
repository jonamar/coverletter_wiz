# Changelog

All notable changes to the coverletter_wiz project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Future changes will be listed here before release

## [0.6.0] - 2025-04-19

### Removed
- Completely removed the `analyze` command from the CLI interface and codebase
- Deleted unused `analyze_job.py` module

### Changed
- Finished migration of all job analysis functionality to the `report` command
- Updated README.md to reflect command consolidation
- Enhanced CLI interface with additional parameters for `report`:
  - Added `--list` parameter to list all available jobs
  - Added `--weights` parameter for priority weight customization
  - Added `--min-rating` parameter for content rating thresholds

### Fixed
- Fixed lint errors in test_job_analyzer.py
- Updated system architecture diagrams to reflect the current design

## [0.5.2] - 2025-04-19

### Added
- Added smart HTML content extraction to the `report` command for better job description analysis
- The `report` command can now directly fetch, analyze, and store jobs from URLs

### Changed
- Removed the redundant `analyze` command, integrating its most valuable functionality into the `report` command
- Improved CLI interface consistency through command consolidation

## [0.5.1] - 2025-04-19

### Fixed
- Fixed CLI import error by removing references to the deprecated `match_content` module

## [0.5.0] - 2025-04-19

### Added
- Added customizable scoring weights via the `--weights` parameter
- Implemented multi-tag bonus for content blocks that match multiple tags
- Added `--list` argument to report command for listing available jobs

### Changed
- Improved scoring algorithm to use configurable weights for tag priorities
- Enhanced content block matching with more sophisticated scoring

### Removed
- Completely removed redundant `match_content.py` module after consolidating all functionality into `generate_report.py`

## [0.4.2] - 2025-04-19

### Changed
- Removed redundant `match` command in favor of the more comprehensive `report` command
- Simplified CLI interface by consolidating duplicate functionality
- Updated main entry point to use consistent command naming

## [0.4.1] - 2025-04-19

### Added
- Added formal project configuration with pyproject.toml
- Created minimal setup.py for compatibility with older tools
- Configured package metadata for potential PyPI publishing
- Updated GitHub username in repository links

## [0.4.0] - 2025-04-19

### Changed
- Completely redesigned report format to focus on content blocks rather than metadata
- Removed differences sections with ANSI color codes from reports
- Eliminated duplicate similar blocks from reports
- Improved error handling for Ollama model availability checks
- Fixed job ID lookup to properly handle both string and integer IDs
- Enhanced cover letter generation with gemma3:12b model

## [0.3.2] - 2025-04-18

### Added
- Enhanced cover letter report with top 15 content blocks ordered by rating
- Added block IDs to report output for easier reference
- Created new ollama_utils module for cover letter generation
- Added clickable link to report file in terminal output

### Changed
- Updated cover letter generation to use gemma3:12b as the default model
- Improved cover letter prompt to better preserve original content
- Enhanced report filename format for better identification
- Made semantic deduplication enabled by default

## [0.3.1] - 2023-07-25

### Added
- Enhanced semantic deduplication in export_content.py using existing analyze_content_block_similarity function
- Added visual highlighting of differences between similar content blocks in exports
- Added semantic deduplication option to generate_report.py for more intelligent content comparison
- Improved content block identification with IDs in reports and exports

### Changed
- Refactored similarity detection to use the existing analyze_content_block_similarity function
- Removed redundant calculate_content_similarity function to maintain DRY principles
- Enhanced tournament system to use the more robust similarity detection

## [0.3.0] - 2023-07-25

### Added
- Unique content block identifiers (format: "B123") for reliable tracking
- Similarity-based tournament pairing to prioritize comparing similar content blocks
- Visual highlighting of differences between similar content blocks
- Enhanced tournament interface showing block IDs and similarity percentages
- Support for tournaments with just 2 blocks (previously required more)

### Changed
- Tournament system now prioritizes comparing semantically similar content
- Improved content block identification using IDs instead of just text content
- Enhanced rating persistence to work with the new ID system
- Refactored tournament code for better maintainability

## [0.2.2] - 2025-04-18

### Improved
- Eliminated intermediate file completely with direct-to-canonical approach
- TextProcessor now writes directly to canonical file via DataManager
- Enhanced documentation with updated architecture diagram
- Added test for direct-to-canonical file approach

## [0.2.1] - 2025-04-18

### Added
- Comprehensive unit tests for `DataManager` with `test_data_manager.py`
- Integration tests for DataManager and ContentProcessor interactions
- Robust error handling test coverage for data management

### Improved
- Enhanced test coverage for singleton pattern handling
- Improved testing approach with minimal mocking of core functionality
- Added real file handling tests with temporary test directories

## [0.2.0] - 2025-04-18

### Added
- Centralized data management with new `DataManager` class in `src/core/data_manager.py`
- Comprehensive test coverage for data integration with `test_data_integration.py`
- Singleton pattern implementation for consistent data access across components

### Fixed
- Resolved synchronization issues between text processing and rating systems
- Fixed content rating persistence across different application components
- Eliminated data duplication between multiple JSON files

### Changed
- Consolidated to a single canonical data file (`cover_letter_content.json`)
- Standardized file naming convention with `processed_text_files.json` for intermediate data
- Removed legacy migration code after successful data consolidation
- Updated all components to use the centralized DataManager

### Improved
- Enhanced error handling and logging throughout data management
- Added data validation for file access and processing
- Implemented robust handling of file permissions and content types

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

## [0.1.3] - 2025-04-17

### Fixed
- Updated test suite to use new module-based CLI entry point
- Resolved CLI help command test failure by migrating to `python -m src`

### Changed
- Enhanced documentation across all core and utility modules
- Added comprehensive type annotations to improve code quality
- Updated config.py with improved type safety and documentation

### Improved
- Implemented Google-style docstrings in core and utility files
- Maintained existing functionality while improving code readability

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

[Unreleased]: https://github.com/jonamar/coverletter_wiz/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/jonamar/coverletter_wiz/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/jonamar/coverletter_wiz/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/jonamar/coverletter_wiz/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/jonamar/coverletter_wiz/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/jonamar/coverletter_wiz/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/jonamar/coverletter_wiz/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/jonamar/coverletter_wiz/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/jonamar/coverletter_wiz/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/jonamar/coverletter_wiz/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.7...v0.3.0
[0.2.7]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.6...v0.2.7
[0.2.6]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.5...v0.2.6
[0.2.5]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/jonamar/coverletter_wiz/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/jonamar/coverletter_wiz/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/jonamar/coverletter_wiz/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/jonamar/coverletter_wiz/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/jonamar/coverletter_wiz/releases/tag/v0.1.0
