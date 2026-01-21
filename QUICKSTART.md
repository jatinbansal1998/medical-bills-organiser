# Quick Start

```bash
# 1. Set your API key
echo "OPENROUTER_API_KEY=your-key-here" > .env

# 2. Run
poetry run medical-sorter /path/to/documents

# With specific model
poetry run medical-sorter /path/to/docs --model "google/gemini-2.0-flash-exp:free"

# Skip confirmation
poetry run medical-sorter /path/to/documents --yes

poetry run medical-sorter /Users/jatinbansal/Documents/test-medical-docs/ --model "google/gemma-3-27b-it:free" --debug
```

## Vision-Capable Models (Free)
- `google/gemini-2.0-flash-exp:free` ✅
- `google/gemini-2.0-flash-001` ✅

## Models That DON'T Work
- `mistralai/*` - No vision support
- `allenai/molmo-*` - Returns empty responses
