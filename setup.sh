#!/bin/bash

# Setup script for Intelligent Medical File Sorter
# This script sets up the development environment automatically

set -e

echo "🔧 Setting up Medical File Sorter..."
echo ""

# Detect OS
OS="$(uname -s)"

# Check for Homebrew (macOS only)
if [[ "$OS" == "Darwin" ]]; then
    if ! command -v brew &> /dev/null; then
        echo ""
        echo "⚠️  Homebrew is not installed."
        echo "   Install it from: https://brew.sh"
        echo ""
        read -p "Continue without Homebrew? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✓ Found Homebrew"
    fi
fi

# Check for uv, install if missing
if ! command -v uv &> /dev/null; then
    echo ""
    echo "📦 uv is not installed."
    if [[ "$OS" == "Darwin" ]] && command -v brew &> /dev/null; then
        echo "   Installing uv via Homebrew..."
        brew install uv
        echo "✓ Installed uv"
    else
        echo "   Installing uv via curl..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo "✓ Installed uv"
    fi
else
    echo "✓ Found uv"
fi

# Create virtual environment with Python 3.12 using uv
echo ""
echo "🐍 Setting up Python 3.12 environment..."
uv venv --python 3.12 --clear
source .venv/bin/activate
echo "✓ Virtual environment created with $(python3 --version)"

# Check for pipx, install if missing (macOS)
if ! command -v pipx &> /dev/null; then
    echo ""
    echo "📦 pipx is not installed."
    if [[ "$OS" == "Darwin" ]] && command -v brew &> /dev/null; then
        echo "   Installing pipx via Homebrew..."
        brew install pipx
        pipx ensurepath
        echo "✓ Installed pipx"
    else
        echo "❌ Please install pipx manually:"
        echo "   macOS: brew install pipx"
        echo "   Linux: python3 -m pip install --user pipx"
        exit 1
    fi
else
    echo "✓ Found pipx"
fi

# Check for Poetry, install if missing
if ! command -v poetry &> /dev/null; then
    echo ""
    echo "📦 Poetry is not installed."
    echo "   Installing Poetry via pipx..."
    pipx install poetry
    echo "✓ Installed Poetry"
else
    POETRY_VERSION=$(poetry --version)
    echo "✓ Found $POETRY_VERSION"
fi

# Check for poppler (required for pdf2image)
if ! command -v pdftoppm &> /dev/null; then
    echo ""
    echo "📦 poppler is not installed (required for PDF processing)."
    if [[ "$OS" == "Darwin" ]] && command -v brew &> /dev/null; then
        echo "   Installing poppler via Homebrew..."
        brew install poppler
        echo "✓ Installed poppler"
    else
        echo "⚠️  Please install poppler manually:"
        echo "   macOS: brew install poppler"
        echo "   Linux: sudo apt-get install poppler-utils"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "✓ Found poppler"
fi

# Configure Poetry to use in-project virtualenv
echo ""
echo "📦 Configuring Poetry..."
poetry config virtualenvs.in-project true --local

# Install dependencies (includes PaddleOCR)
echo ""
echo "📥 Installing dependencies..."
poetry install

# Check for .env file
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and configure your LLM backend"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and configure your LLM backend (OpenRouter, OpenAI, or LM Studio)"
echo "  2. Activate the virtual environment: poetry shell"
echo "  3. Run the tool: medical-sorter /path/to/documents"
echo ""
echo "OCR Backends available:"
echo "  • paddleocr (default) - Fast, free, local OCR"
echo "  • llm                 - Vision LLM (best for handwriting)"
echo "  • hybrid              - PaddleOCR + LLM fallback"
echo ""
echo "Run: poetry run medical-sorter /path/to/documents"
