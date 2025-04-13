# Environment Setup for coverletter_wiz

## Project Structure
- Main app repo: `/Users/jonamar/Documents/coverletter_container/coverletter_wiz`
- Data repo: `/Users/jonamar/Documents/coverletter_container/coverletter_data`
- Virtual environment: `.venv` (located in the root of coverletter_wiz)

## Dependencies
All dependencies are managed in a single `requirements.txt` file at the root of the project.
Key dependencies include:
- spacy>=3.7.2 (with en_core_web_lg-3.8.0 model)
- beautifulsoup4>=4.13.3
- PyYAML>=6.0.2
- ollama>=0.4.7

## Data Structure
The data is stored in a separate directory for privacy:
- json/ (analyzed_jobs.json, cover_letter_content.json)
- config/ (categories.yaml)
- text-archive/ (text files)
- exports/ (exported content)
- reports/ (generated reports)

## Environment Activation
To activate the virtual environment:
```bash
cd /Users/jonamar/Documents/coverletter_container/coverletter_wiz
source .venv/bin/activate
```

## Note for AI Assistants
When working with this project, always reference the virtual environment in `.venv` and the consolidated requirements in `requirements.txt`. All data files are stored in the external data directory.
