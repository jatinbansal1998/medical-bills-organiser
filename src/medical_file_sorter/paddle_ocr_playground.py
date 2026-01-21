import json
import sys
import tempfile
from pathlib import Path
from typing import TextIO

from PIL import Image
from paddleocr import PaddleOCR
from pillow_heif import register_heif_opener

# Register HEIF opener so PIL can read HEIC files
register_heif_opener()


def convert_heic_to_png(
    input_path: Path, temp_dir: Path, log: TextIO, max_size: int = 2000
) -> Path:
    """Convert HEIC file to PNG and return the path to the converted file."""
    img = Image.open(input_path)
    # Resize if too large to speed up OCR
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size))
        log.write(f"Resized image from {Image.open(input_path).size} to {img.size}\n")
    png_path = temp_dir / f"{input_path.stem}.png"
    img.save(png_path, "PNG")
    log.write(f"Converted HEIC to PNG: {png_path}\n")
    return png_path


def extract_text_from_json(json_path: Path) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    texts = data.get("rec_texts", [])
    return "\n".join(text for text in texts if text)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m medical_file_sorter.paddle_ocr_playground "
            "/path/to/image [output_dir]"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store original stem for output file naming
    original_stem = input_path.stem
    temp_dir = None
    converted_path = None

    # Create log file for output
    log_path = output_dir / f"{original_stem}_log.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        # Convert HEIC to PNG if needed
        if input_path.suffix.lower() in (".heic", ".heif"):
            temp_dir = Path(tempfile.mkdtemp())
            converted_path = convert_heic_to_png(input_path, temp_dir, log)
            input_path = converted_path

        ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        result = ocr.predict(input=str(input_path))

        # Log summary info
        for res in result:
            num_regions = len(res.get("rec_texts", []))
            log.write(f"Input: {res.get('input_path', input_path)}\n")
            log.write(f"Text regions detected: {num_regions}\n")
            res.save_to_img(str(output_dir))
            res.save_to_json(str(output_dir))

        json_files = sorted(output_dir.glob(f"{input_path.stem}*_res.json"))
        if not json_files:
            log.write("No JSON output files found to extract text.\n")
            # Cleanup temp file if created
            if temp_dir and temp_dir.exists():
                import shutil

                shutil.rmtree(temp_dir)
            return

        combined_text = "\n\n".join(
            extract_text_from_json(json_file) for json_file in json_files
        ).strip()

        text_path = output_dir / f"{original_stem}_text.txt"
        text_path.write_text(combined_text + "\n", encoding="utf-8")

        log.write(f"\nOutput files:\n")
        log.write(f"  Text: {text_path}\n")
        for json_file in json_files:
            log.write(f"  JSON: {json_file}\n")

        # Cleanup temp file if created
        if temp_dir and temp_dir.exists():
            import shutil

            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
