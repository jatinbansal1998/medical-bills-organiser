"""
Main entry point for the Intelligent Medical File Sorter CLI.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .content_extractor import ContentExtractor
from .ocr_factory import OCRBackend, create_extractor, get_ocr_backend_from_string
from .paddle_ocr_extractor import load_paddleocr_output_dir
from .image_processor import ImageProcessor
from .llm_sorter import (
    BACKEND_CONFIGS,
    LLMBackend,
    LLMSorter,
    get_backend_from_string,
    list_backends,
    parse_bill_amount,
)
from .pdf_merger import PDFMerger


def display_groups(sort_result: dict) -> None:
    """
    Display the sorted groups to the user.

    Args:
        sort_result: Dictionary with 'groups' and 'uncategorized' keys.
    """
    print("\n" + "=" * 60)
    print("📋 DOCUMENT ORGANIZATION RESULTS")
    print("=" * 60)

    groups = sort_result.get("groups", [])
    total_bill_amount = 0.0
    
    if groups:
        for idx, group in enumerate(groups, 1):
            # Handle both old format (list) and new format (dict with files/summary)
            if isinstance(group, dict):
                files = group.get("files", [])
                summary = group.get("summary", "")
                patient_name = group.get("patient_name", "Unknown")
                bill_amount = parse_bill_amount(group.get("bill_amount", 0))
            else:
                files = group
                summary = ""
                patient_name = "Unknown"
                bill_amount = 0.0

            total_bill_amount += bill_amount

            print(f"\n🗂️  Group {idx} (Transaction):")
            print(f"    👤 Patient: {patient_name}")
            if summary:
                print(f"    📝 {summary}")
            if bill_amount > 0:
                print(f"    💰 Bill Amount: ₹{bill_amount:,.2f}")
            for filename in files:
                print(f"    • {filename}")
    else:
        print("\n⚠️  No document groups were identified.")

    uncategorized = sort_result.get("uncategorized", [])
    if uncategorized:
        print(f"\n📁 Uncategorized ({len(uncategorized)} files):")
        for filename in uncategorized:
            print(f"    • {filename}")

    print("\n" + "=" * 60)
    
    # Show grand total
    if total_bill_amount > 0:
        print(f"\n💵 TOTAL BILLS PAID: ₹{total_bill_amount:,.2f}")
        print("=" * 60)


def get_user_confirmation() -> bool:
    """
    Ask the user for confirmation to proceed with merging.

    Returns:
        True if user confirms, False otherwise.
    """
    while True:
        response = input("\n❓ Proceed with merging these documents? (Y/N): ").strip().upper()
        if response in ("Y", "YES"):
            return True
        elif response in ("N", "NO"):
            return False
        else:
            print("Please enter Y or N.")


def get_api_key_for_backend(backend: LLMBackend) -> str | None:
    """
    Get the appropriate API key for the specified backend.
    
    Args:
        backend: The LLM backend being used.
        
    Returns:
        API key string or None if not required/found.
    """
    config = BACKEND_CONFIGS[backend]
    
    if not config.requires_api_key:
        return None
    
    # Check for backend-specific API key first
    if backend == LLMBackend.OPENROUTER:
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            return key
    
    # Fall back to generic OPENAI_API_KEY
    return os.getenv("OPENAI_API_KEY")


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    # Load environment variables
    load_dotenv()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Intelligent Medical File Sorter - Organize medical documents using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Using OpenRouter (default)
  %(prog)s /path/to/medical/documents

  # LLM image + text pipeline (vision OCR)
  %(prog)s /path/to/docs --pipeline llm-image-text --vision-model "google/gemma-3-27b-it:free"

  # Using LM Studio (local)
  %(prog)s /path/to/docs --backend lmstudio --model "your-local-model"

  # Using OpenRouter
  %(prog)s /path/to/docs --backend openrouter --model "anthropic/claude-3.5-sonnet"

{list_backends()}

Requirements:
  - API key set via environment variable (see .env.example)
  - poppler must be installed (brew install poppler on macOS)
        """,
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing medical documents",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="Merged_Medical_Records.pdf",
        help="Output filename (default: Merged_Medical_Records.pdf)",
    )
    parser.add_argument(
        "--max-dimension",
        "-m",
        type=int,
        default=1000,
        help="Maximum image dimension for processing (default: 1000)",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt and proceed with merging",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="openrouter",
        help="LLM backend: openai, openrouter, lmstudio (default: openrouter)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (uses backend default if not specified)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom API base URL (overrides backend default)",
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default=None,
        help="Model for Stage 1 content extraction (vision). Uses --model if not specified.",
    )
    parser.add_argument(
        "--sort-model",
        type=str,
        default=None,
        help="Model for Stage 2 document sorting (text). Uses --model if not specified.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save extracted content and results to debug folder",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process per vision LLM call in Stage 1 (default: 10)",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["ocr-text", "llm-image-text"],
        default=None,
        help=(
            "Processing pipeline: ocr-text (PaddleOCR + text sorting) or "
            "llm-image-text (vision OCR + text sorting). Overrides --ocr-backend."
        ),
    )
    parser.add_argument(
        "--ocr-backend",
        type=str,
        default=os.getenv("OCR_BACKEND", "paddleocr"),
        choices=["paddleocr", "llm", "hybrid"],
        help=(
            "OCR backend: paddleocr (fast/free, default), llm (best for handwriting), "
            "hybrid (auto-fallback). Use --pipeline for a simpler choice."
        ),
    )
    parser.add_argument(
        "--ocr-confidence-threshold",
        type=float,
        default=float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.7")),
        help="Confidence threshold for hybrid mode fallback (0.0-1.0, default: 0.7)",
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default=os.getenv("OCR_LANG", "en"),
        help="Language for PaddleOCR (default: en). See PaddleOCR docs for codes.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Run only OCR extraction and save results to a checkpoint file. Skip sorting and merging.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a previously saved extraction checkpoint JSON file. Skips OCR and uses saved content.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save extraction checkpoint (default: .medical_sorter_checkpoints in input folder)",
    )

    args = parser.parse_args()

    if args.pipeline:
        pipeline_map = {
            "ocr-text": "paddleocr",
            "llm-image-text": "llm",
        }
        args.ocr_backend = pipeline_map[args.pipeline]

    # Validate folder path
    folder_path = Path(args.folder_path).expanduser().resolve()
    if not folder_path.exists():
        print(f"❌ Error: Folder does not exist: {folder_path}")
        return 1
    if not folder_path.is_dir():
        print(f"❌ Error: Path is not a directory: {folder_path}")
        return 1

    # Parse backend
    try:
        backend = get_backend_from_string(args.backend)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    
    backend_config = BACKEND_CONFIGS[backend]
    
    # Get API key
    api_key = get_api_key_for_backend(backend)
    if backend_config.requires_api_key and not api_key:
        if backend == LLMBackend.OPENROUTER:
            print("❌ Error: OPENROUTER_API_KEY or OPENAI_API_KEY environment variable is not set.")
        else:
            print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        print("   Set it in your environment or create a .env file.")
        return 1

    # Determine models for each stage
    base_model = args.model or backend_config.default_model
    vision_model = args.vision_model or base_model
    sort_model = args.sort_model or base_model

    # Parse OCR backend
    try:
        ocr_backend = get_ocr_backend_from_string(args.ocr_backend)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    if ocr_backend == OCRBackend.PADDLEOCR:
        pipeline_label = "OCR + text (PaddleOCR)"
    elif ocr_backend == OCRBackend.LLM:
        pipeline_label = "LLM image + text (vision OCR)"
    else:
        pipeline_label = "Hybrid OCR + text (PaddleOCR + LLM fallback)"

    print(f"\n🔍 Scanning folder: {folder_path}")
    print(f"🧭 Pipeline: {pipeline_label}")
    print(f"🤖 Using LLM backend: {backend_config.name}")
    print(f"   📷 OCR backend (Stage 1): {ocr_backend.value}")
    if ocr_backend == OCRBackend.LLM:
        print(f"   👁️  Vision model: {vision_model}")
    elif ocr_backend == OCRBackend.HYBRID:
        print(f"   👁️  Vision model (fallback): {vision_model}")
        print(f"   🎚️  Confidence threshold: {args.ocr_confidence_threshold:.0%}")
    print(f"   📋 Sort model (Stage 2): {sort_model}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup debug directory if enabled
    debug_dir = None
    if args.debug:
        debug_dir = folder_path / ".medical_sorter_debug" / run_timestamp
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"   🐛 Debug output: {debug_dir}")

    # Step 1: Process files (convert to images)
    print("\n📸 Step 1: Processing files...")
    processor = ImageProcessor(max_dimension=args.max_dimension)

    try:
        valid_files = processor.get_valid_files(folder_path)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    if not valid_files:
        print("❌ Error: No valid files found in the folder.")
        print("   Supported formats: PDF, JPG, JPEG, PNG, HEIC")
        return 1

    print(f"   Found {len(valid_files)} valid files")

    # Warn about large batches
    if len(valid_files) > 20:
        print(f"\n⚠️  Warning: Processing {len(valid_files)} files may affect accuracy.")
        print("   Consider processing in smaller batches for best results.")

    # Convert files to Base64
    print("\n🔄 Converting files to processable format...")
    file_data = processor.process_folder(folder_path)

    if not file_data:
        print("❌ Error: Failed to process any files.")
        return 1

    # Count total pages
    total_pages = sum(
        len(v) if isinstance(v, list) else 1 
        for v in file_data.values()
    )
    print(f"   Successfully processed {len(file_data)} files ({total_pages} total pages)")

    # Step 2: Extract content (Stage 1) - or resume from checkpoint
    if args.resume_from:
        # Resume from saved checkpoint
        resume_path = Path(args.resume_from).expanduser().resolve()
        if not resume_path.exists():
            print(f"❌ Error: Checkpoint file does not exist: {resume_path}")
            return 1
        
        print(f"\n📂 Step 2: Loading extracted content from checkpoint...")
        print(f"   📄 {resume_path}")
        
        try:
            if resume_path.is_dir():
                extracted_content = load_paddleocr_output_dir(resume_path)
            else:
                with open(resume_path, "r", encoding="utf-8") as f:
                    extracted_content = json.load(f)
            if not extracted_content:
                print("❌ Error: No extracted content found in checkpoint.")
                return 1
            print(f"   ✅ Loaded content for {len(extracted_content)} files")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return 1
    else:
        # Normal OCR extraction
        checkpoint_dir = (
            Path(args.checkpoint_dir)
            if args.checkpoint_dir
            else folder_path / ".medical_sorter_checkpoints"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ocr_output_dir = None
        if ocr_backend in {OCRBackend.PADDLEOCR, OCRBackend.HYBRID}:
            ocr_output_dir = checkpoint_dir / f"paddleocr_output_{run_timestamp}"
            ocr_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   🧾 OCR outputs: {ocr_output_dir}")

        ocr_label = {
            OCRBackend.PADDLEOCR: "PaddleOCR",
            OCRBackend.LLM: "vision model",
            OCRBackend.HYBRID: "hybrid (PaddleOCR + LLM fallback)",
        }[ocr_backend]
        print(f"\n👁️  Step 2: Extracting content with {ocr_label}...")
        
        try:
            # Determine extra headers for OpenRouter (used for LLM and hybrid modes)
            extra_headers = {}
            if backend == LLMBackend.OPENROUTER:
                extra_headers = {
                    "HTTP-Referer": "https://github.com/medical-file-sorter",
                    "X-Title": "Medical File Sorter"
                }
            
            # Create the appropriate extractor based on OCR backend
            extractor = create_extractor(
                backend=ocr_backend,
                # PaddleOCR config
                paddle_lang=args.ocr_lang,
                paddle_use_gpu=False,
                # LLM config (for llm and hybrid modes)
                llm_api_key=api_key,
                llm_model=vision_model,
                llm_base_url=args.base_url or backend_config.base_url,
                llm_extra_headers=extra_headers if extra_headers else None,
                # Hybrid config
                confidence_threshold=args.ocr_confidence_threshold,
            )
            
            extract_kwargs = {"batch_size": args.batch_size}
            if ocr_output_dir:
                extract_kwargs["output_dir"] = ocr_output_dir
            extracted_content = extractor.extract_batch(file_data, **extract_kwargs)
            
        except Exception as e:
            print(f"❌ Error during content extraction: {e}")
            return 1

        if not extracted_content:
            print("❌ Error: Failed to extract content from any files.")
            return 1

        print(f"   Extracted content from {len(extracted_content)} files")

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f"extracted_content_{run_timestamp}.json"
        
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(extracted_content, f, indent=2, ensure_ascii=False)
        print(f"   💾 Saved checkpoint: {checkpoint_path}")

        # If extract-only mode, exit here
        if args.extract_only:
            print(f"\n✅ Extraction complete! Checkpoint saved to:")
            print(f"   {checkpoint_path}")
            print(f"\n   To continue processing, run:")
            print(f"   poetry run medical-sorter \"{folder_path}\" --resume-from \"{checkpoint_path}\"")
            return 0

    # Save extracted content to debug folder (if not resuming)
    if debug_dir and not args.resume_from:
        with open(debug_dir / "extracted_content.json", "w") as f:
            json.dump(extracted_content, f, indent=2, ensure_ascii=False)
        print(f"   💾 Saved extracted content to debug folder")

    # Step 3: Sort with LLM using extracted text (Stage 2)
    print(f"\n📋 Step 3: Sorting documents with text model...")
    
    try:
        sorter = LLMSorter(
            api_key=api_key,
            model=sort_model,
            backend=backend,
            base_url=args.base_url,
        )
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    try:
        raw_result = sorter.sort_documents_by_text(extracted_content)
        sort_result = sorter.validate_result(raw_result, set(file_data.keys()))
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    # Save sort result to debug folder
    if debug_dir:
        with open(debug_dir / "sort_result.json", "w") as f:
            json.dump(sort_result, f, indent=2, ensure_ascii=False)
        print(f"   💾 Saved sort result to debug folder")

    # Display results
    display_groups(sort_result)

    # Get user confirmation
    if not args.yes:
        if not get_user_confirmation():
            print("\n🚫 Operation cancelled. You can manually organize files and try again.")
            return 0

    # Step 4: Merge PDFs
    print("\n📄 Step 4: Merging documents...")
    merger = PDFMerger(output_filename=args.output)
    output_path = merger.merge_documents(folder_path, sort_result)

    if output_path:
        print(f"\n✅ Success! Merged PDF saved to: {output_path}")
        return 0
    else:
        print("\n❌ Failed to create merged PDF.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
