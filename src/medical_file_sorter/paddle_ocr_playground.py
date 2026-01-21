import json
import sys
from pathlib import Path

from paddleocr import PaddleOCR


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

    output_dir = (
        Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path("output")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    result = ocr.predict(input=str(input_path))
    print(result)

    for res in result:
        res.print()
        res.save_to_img(str(output_dir))
        res.save_to_json(str(output_dir))

    json_files = sorted(output_dir.glob(f"{input_path.stem}*_res.json"))
    if not json_files:
        print("No JSON output files found to extract text.")
        return

    combined_text = "\n\n".join(
        extract_text_from_json(json_file) for json_file in json_files
    ).strip()

    text_path = output_dir / f"{input_path.stem}_text.txt"
    text_path.write_text(combined_text + "\n", encoding="utf-8")

    print("\n=== OCR TEXT OUTPUT ===")
    print(combined_text)
    print(f"\nSaved text to: {text_path}")


if __name__ == "__main__":
    main()