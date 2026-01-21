# Quick Start

Requires Python 3.12.

```bash
# 1. Set your API key
echo "OPENROUTER_API_KEY=your-key-here" > .env

# 2. Install dependencies (includes PaddleOCR)
poetry install

# 3. Run with default settings (PaddleOCR)
poetry run medical-sorter /path/to/documents
```

## Pipeline Modes

### OCR + Text (Default, PaddleOCR)
```bash
poetry run medical-sorter "/Users/jatinbansal/Documents/Medical Bills/test docs/" \
  --pipeline ocr-text \
  --debug
```

### LLM Image + Text (Vision OCR)
```bash
poetry run medical-sorter "/Users/jatinbansal/Documents/Medical Bills/test docs/" \
  --pipeline llm-image-text \
  --vision-model "google/gemma-3-27b-it:free" \
  --sort-model "openai/gpt-oss-120b:free" \
  --debug
```

## OCR Backend Modes (Advanced)

### PaddleOCR (Fast & Free)
```bash
poetry run medical-sorter "/Users/jatinbansal/Documents/Medical Bills/test docs/" \
  --ocr-backend paddleocr \
  --debug
```

### LLM Mode - Best for Handwriting
```bash
poetry run medical-sorter "/Users/jatinbansal/Documents/Medical Bills/test docs/" \
  --ocr-backend llm \
  --vision-model "google/gemma-3-27b-it:free" \
  --sort-model "openai/gpt-oss-120b:free" \
  --debug
```

### Hybrid Mode - PaddleOCR + LLM Fallback
```bash
poetry run medical-sorter "/Users/jatinbansal/Documents/Medical Bills/test docs/" \
  --ocr-backend hybrid \
  --ocr-confidence-threshold 0.7 \
  --vision-model "google/gemma-3-27b-it:free" \
  --sort-model "openai/gpt-oss-120b:free" \
  --debug
```

## Other Examples

```bash
# Skip confirmation prompt
poetry run medical-sorter /path/to/documents --yes

# Custom output filename
poetry run medical-sorter /path/to/documents --output "MyRecords.pdf"
```

## Checkpoint Workflow (Save & Resume)

OCR extraction results are automatically saved as checkpoints. Use this for large document sets or to resume interrupted processing.

```bash
# Step 1: Run OCR extraction only (saves checkpoint and exits)
poetry run medical-sorter "/Users/jatinbansal/Documents/Medical Bills/test docs/" \
  --extract-only

# Step 2: Resume from checkpoint to sort and merge
poetry run medical-sorter "/Users/jatinbansal/Documents/Medical Bills/test docs/" \
  --resume-from "/Users/jatinbansal/Documents/Medical Bills/test docs/.medical_sorter_checkpoints/extracted_content_20260121_140000.json" \
  --sort-model "openai/gpt-oss-120b:free"

# You can also resume directly from a PaddleOCR output folder:
# --resume-from "/path/to/.medical_sorter_checkpoints/paddleocr_output_20260121_140000"

# Custom checkpoint directory
poetry run medical-sorter /path/to/docs --extract-only --checkpoint-dir /path/to/checkpoints/
```

## Vision-Capable Models (Free)
- `google/gemma-3-27b-it:free` ✅
- `google/gemini-2.0-flash-exp:free` ✅
- `google/gemini-2.0-flash-001` ✅

## Models That DON'T Work
- `mistralai/*` - No vision support
- `allenai/molmo-*` - Returns empty responses
