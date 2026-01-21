# Intelligent Medical File Sorter CLI

A Python-based CLI tool for macOS that ingests a folder of disorganized medical documents (images and PDFs), identifies relationships between prescriptions and bills using a Multimodal LLM (GPT-4o), and merges them into a single, structured PDF.

## Features

- 📄 **Multi-format support**: Handles PDF, JPG, JPEG, PNG, and HEIC files
- 🤖 **AI-powered organization**: Uses GPT-4o Vision to classify and group documents
- 🔗 **Smart matching**: Links prescriptions with related bills based on content (medicines, doctor names)
- 📑 **PDF merging**: Combines all documents into a single organized PDF
- ✅ **User confirmation**: Review the AI's grouping before merging
- ⚡ **Multiple OCR backends**: Choose between PaddleOCR (fast/free), Vision LLM (best for handwriting), or Hybrid mode

## Quick Setup

The easiest way to get started is to run the setup script:

Requires Python 3.12.

```bash
# Clone the repository
git clone https://github.com/your-username/medical-bills-organiser.git
cd medical-bills-organiser

# Run the setup script (installs all dependencies automatically)
chmod +x setup.sh
./setup.sh
```

The setup script will automatically:
- Install pipx (via Homebrew on macOS)
- Install Poetry (via pipx)
- Install poppler (via Homebrew on macOS)
- Install Python dependencies
- Create `.env` from template

## Manual Installation

If you prefer to install dependencies manually:

### 1. System Dependencies

```bash
# macOS
brew install pipx poppler
pipx install poetry
```

### 2. Project Setup

```bash
# Install Python dependencies (includes PaddleOCR)
poetry install

# Copy environment template
cp .env.example .env

# Activate virtual environment
poetry shell
```

## Usage

```bash
# Basic usage (OCR + text pipeline with PaddleOCR)
python -m medical_file_sorter.main /path/to/medical/documents

# Use LLM image + text pipeline (vision OCR)
python -m medical_file_sorter.main ~/Documents/medical_files --pipeline llm-image-text

# Use Hybrid mode (PaddleOCR with LLM fallback for handwriting)
python -m medical_file_sorter.main ~/Documents/medical_files --ocr-backend hybrid

# With options
python -m medical_file_sorter.main ~/Documents/medical_files \
    --output "My_Medical_Records.pdf" \
    --max-dimension 1200

# Skip confirmation prompt
python -m medical_file_sorter.main /path/to/docs --yes
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `folder_path` | - | Required | Path to folder containing medical documents |
| `--output` | `-o` | `Merged_Medical_Records.pdf` | Output filename |
| `--max-dimension` | `-m` | `1000` | Maximum image dimension for processing |
| `--yes` | `-y` | `False` | Skip confirmation prompt |
| `--pipeline` | - | - | Pipeline: `ocr-text` (PaddleOCR) or `llm-image-text` (vision OCR). Overrides `--ocr-backend`. |
| `--ocr-backend` | - | `paddleocr` | OCR backend: `paddleocr`, `llm`, `hybrid` |
| `--ocr-confidence-threshold` | - | `0.7` | Confidence threshold for hybrid mode (0.0-1.0) |
| `--ocr-lang` | - | `en` | Language for PaddleOCR |

## OCR Backends

Choose the best OCR backend for your documents:

| Backend | Speed | Cost | Handwriting | Best For |
|---------|-------|------|-------------|----------|
| `llm` | Slower | API costs | ✅ Excellent | Handwritten prescriptions |
| `paddleocr` | Fast | Free | ⚠️ Limited | Typed/printed documents |
| `hybrid` | Medium | Minimal | ✅ Good | Mixed documents |

Use `--pipeline ocr-text` (default) for PaddleOCR + text sorting, or `--pipeline llm-image-text` for vision OCR + text sorting. **Hybrid mode** uses PaddleOCR first, then automatically falls back to Vision LLM for pages with low OCR confidence (e.g., handwritten text).

Reference: [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)

## How It Works

1. **Scanning**: The tool scans the specified folder for valid files (PDF, images)
2. **Processing**: Each file is converted to a standardized image format for AI analysis
3. **AI Analysis**: GPT-4o Vision analyzes all documents to:
   - Classify each as "Prescription" or "Bill"
   - Group related documents (e.g., a prescription with its associated bills)
4. **Review**: The groupings are displayed for user confirmation
5. **Merging**: Documents are merged into a single PDF in the organized order

## Project Structure

```
medical-bills-organiser/
├── src/
│   └── medical_file_sorter/
│       ├── __init__.py
│       ├── main.py                # CLI entry point
│       ├── image_processor.py     # File → Base64 conversion
│       ├── content_extractor.py   # LLM-based OCR
│       ├── paddle_ocr_extractor.py # PaddleOCR-based extraction
│       ├── ocr_factory.py         # OCR backend factory
│       ├── llm_sorter.py          # Document classification logic
│       └── pdf_merger.py          # PDF merging logic
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## Development

```bash
# Install with dev dependencies
poetry install

# Run tests (when available)
poetry run pytest

# Format code
poetry run black src/
poetry run isort src/
```

## Error Handling

- **API Failures**: Gracefully handles OpenAI API connection errors
- **Missing Files**: Skips files that the AI mentions but don't exist (logs a warning)
- **Empty Folder**: Exits clearly if no valid files are found
- **Large Batches**: Warns user when processing >20 files (may affect accuracy)

## License

MIT
