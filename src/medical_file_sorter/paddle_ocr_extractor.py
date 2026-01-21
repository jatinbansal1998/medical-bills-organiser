"""
PaddleOCR-based content extraction for medical documents.

Alternative to LLM-based extraction in content_extractor.py.
Uses PaddleOCR for fast, local, free text extraction with confidence scores.

References:
- PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
- PaddleOCR Documentation: https://paddlepaddle.github.io/PaddleOCR/
"""

import base64
import io
import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from collections import defaultdict
import re

from PIL import Image


@dataclass
class ExtractionResult:
    """Result of text extraction from a single page/image."""

    text: str
    confidence: float  # Average confidence score (0.0 - 1.0)
    region_count: int  # Number of text regions detected


class PaddleOCRExtractor:
    """
    Extracts text content from images using PaddleOCR.

    Provides confidence scores for each extraction, enabling hybrid mode
    where low-confidence results can fall back to LLM extraction.
    """

    def __init__(
        self,
        use_gpu: bool = False,
        lang: str = "en",
        use_angle_cls: bool = True,
        show_log: bool = False,
    ):
        """
        Initialize the PaddleOCR extractor.

        Args:
            use_gpu: Whether to use GPU for inference. Requires CUDA.
            lang: Language code for OCR. Common codes:
                  - 'en': English
                  - 'ch': Chinese
                  - 'hi': Hindi
                  - 'japan': Japanese
                  - 'korean': Korean
                  See https://github.com/PaddlePaddle/PaddleOCR for full list.
            use_angle_cls: Whether to use text angle classification.
            show_log: Whether to show PaddleOCR logs (deprecated in PaddleOCR 3.x,
                      now uses Python's standard logging module).
        """
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError(
                "PaddleOCR is not installed. Install it with:\n"
                "  pip install paddlepaddle paddleocr\n"
                "Or with poetry:\n"
                "  poetry install -E paddleocr"
            )

        # Configure PaddleOCR logging (PaddleOCR 3.x uses Python's logging module)
        # Suppress logs unless show_log is True
        if not show_log:
            try:
                # PaddleOCR 3.x: use the paddleocr logger
                import paddleocr
                if hasattr(paddleocr, 'logger'):
                    paddleocr.logger.setLevel(logging.ERROR)
            except Exception:
                pass
            # Also suppress ppocr logs
            logging.getLogger("ppocr").setLevel(logging.ERROR)

        # Initialize PaddleOCR without show_log (removed in 3.x)
        ocr_kwargs = {
            "use_angle_cls": use_angle_cls,
            "lang": lang,
        }
        if use_gpu:
            ocr_kwargs["use_gpu"] = True

        try:
            self.ocr = PaddleOCR(**ocr_kwargs)
        except Exception as e:
            # Some PaddleOCR versions reject use_gpu; retry without it.
            if "use_gpu" in str(e):
                ocr_kwargs.pop("use_gpu", None)
                self.ocr = PaddleOCR(**ocr_kwargs)
            else:
                raise
        self.lang = lang

    def _base64_to_image(self, base64_data: str) -> Image.Image:
        """Convert Base64 encoded string to PIL Image."""
        image_bytes = base64.b64decode(base64_data)
        return Image.open(io.BytesIO(image_bytes))

    def _extract_from_lines(self, lines: list) -> tuple[list[str], list[float]]:
        texts: list[str] = []
        scores: list[float] = []
        for line in lines:
            if line and isinstance(line, (list, tuple)) and len(line) >= 2:
                text_conf = line[1]
                if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                    text = text_conf[0]
                    conf = text_conf[1]
                    if text:
                        texts.append(str(text))
                    if isinstance(conf, (int, float)):
                        scores.append(float(conf))
        return texts, scores

    def _extract_from_saved_json(self, result_obj) -> tuple[list[str], list[float]]:
        if not hasattr(result_obj, "save_to_json"):
            return ([], [])

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            existing = set(tmp_path.glob("*.json"))
            result_obj.save_to_json(str(tmp_path))
            new_files = [p for p in tmp_path.glob("*.json") if p not in existing]
            if not new_files:
                return ([], [])
            target = max(new_files, key=lambda p: p.stat().st_mtime)
            data = json.loads(target.read_text(encoding="utf-8"))
            texts = data.get("rec_texts", []) or []
            scores = data.get("rec_scores", []) or []
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(scores, (int, float)):
                scores = [scores]
            return (list(texts), [float(s) for s in scores if isinstance(s, (int, float))])

    def _extract_texts_and_scores(self, result) -> tuple[list[str], list[float]]:
        if not result:
            return ([], [])

        if isinstance(result, list):
            # Legacy format for single image: [ [bbox, (text, conf)], ... ]
            if (
                result
                and isinstance(result[0], (list, tuple))
                and len(result[0]) >= 2
                and isinstance(result[0][1], (list, tuple))
                and len(result[0][1]) >= 2
                and isinstance(result[0][1][0], str)
            ):
                texts, scores = self._extract_from_lines(result)
                if texts:
                    return (texts, scores)

            texts: list[str] = []
            scores: list[float] = []
            for item in result:
                item_texts, item_scores = self._extract_texts_and_scores(item)
                texts.extend(item_texts)
                scores.extend(item_scores)
            return (texts, scores)

        if isinstance(result, dict):
            texts = result.get("rec_texts") or result.get("texts") or result.get("text") or []
            scores = result.get("rec_scores") or result.get("scores") or []
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(scores, (int, float)):
                scores = [scores]
            return (list(texts), [float(s) for s in scores if isinstance(s, (int, float))])

        if hasattr(result, "rec_texts"):
            texts = getattr(result, "rec_texts", []) or []
            scores = getattr(result, "rec_scores", []) or []
            if isinstance(texts, str):
                texts = [texts]
            if isinstance(scores, (int, float)):
                scores = [scores]
            return (list(texts), [float(s) for s in scores if isinstance(s, (int, float))])

        if isinstance(result, (list, tuple)):
            return self._extract_from_lines(list(result))

        return self._extract_from_saved_json(result)

    def _save_predict_outputs(self, result, output_dir: Path) -> None:
        if not result:
            return
        if isinstance(result, list):
            for item in result:
                if hasattr(item, "save_to_img"):
                    item.save_to_img(str(output_dir))
                if hasattr(item, "save_to_json"):
                    item.save_to_json(str(output_dir))
            return
        if hasattr(result, "save_to_img"):
            result.save_to_img(str(output_dir))
        if hasattr(result, "save_to_json"):
            result.save_to_json(str(output_dir))

    def _extract_from_pil_image(
        self,
        image: Image.Image,
        output_path: Optional[Path] = None,
    ) -> ExtractionResult:
        """
        Extract text from a PIL Image.

        Args:
            image: PIL Image to extract text from.

        Returns:
            ExtractionResult with text, confidence, and region count.
        """
        output_dir = output_path.parent if output_path else None
        cleanup_tmp = False
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            image.save(output_path, format="JPEG")
            tmp_path = str(output_path)
        else:
            # Save to temporary file (PaddleOCR works best with file paths)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                image.save(tmp.name, format="JPEG")
                tmp_path = tmp.name
            cleanup_tmp = True

        try:
            result = None
            if output_dir and hasattr(self.ocr, "predict"):
                try:
                    result = self.ocr.predict(input=tmp_path)
                except TypeError:
                    result = self.ocr.predict(tmp_path)

            if result is None:
                try:
                    result = self.ocr.ocr(tmp_path, cls=True)
                except TypeError as e:
                    if "cls" in str(e):
                        result = self.ocr.ocr(tmp_path)
                    else:
                        raise
                except AttributeError:
                    try:
                        result = self.ocr.predict(input=tmp_path)
                    except TypeError:
                        result = self.ocr.predict(tmp_path)
                except Exception:
                    if hasattr(self.ocr, "predict"):
                        try:
                            result = self.ocr.predict(input=tmp_path)
                        except TypeError:
                            result = self.ocr.predict(tmp_path)
                    else:
                        raise

            if output_dir and result is not None:
                self._save_predict_outputs(result, output_dir)

            if not result:
                return ExtractionResult(text="", confidence=0.0, region_count=0)

            texts, confidences = self._extract_texts_and_scores(result)

            if not texts:
                return ExtractionResult(text="", confidence=0.0, region_count=0)

            combined_text = "\n".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return ExtractionResult(
                text=combined_text,
                confidence=avg_confidence,
                region_count=len(texts),
            )

        finally:
            if cleanup_tmp:
                Path(tmp_path).unlink(missing_ok=True)

    def extract_from_base64(
        self, base64_data: str, output_path: Optional[Path] = None
    ) -> ExtractionResult:
        """
        Extract text from a Base64 encoded image.

        Args:
            base64_data: Base64 encoded image data.

        Returns:
            ExtractionResult with text, confidence, and region count.
        """
        image = self._base64_to_image(base64_data)
        return self._extract_from_pil_image(image, output_path=output_path)

    def extract_from_file(
        self, file_path: str | Path, output_path: Optional[Path] = None
    ) -> ExtractionResult:
        """
        Extract text from an image file.

        Args:
            file_path: Path to the image file.

        Returns:
            ExtractionResult with text, confidence, and region count.
        """
        image = Image.open(file_path)
        return self._extract_from_pil_image(image, output_path=output_path)

    def extract_from_images(
        self,
        filename: str,
        base64_images: list[str],
        output_dir: Optional[Path] = None,
    ) -> tuple[str, float]:
        """
        Extract text from multiple Base64 images (multi-page document).

        Args:
            filename: Name of the file (for logging).
            base64_images: List of Base64 encoded images.

        Returns:
            Tuple of (combined_text, average_confidence).
        """
        if not base64_images:
            return ("", 0.0)

        all_texts = []
        all_confidences = []

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        for page_idx, base64_data in enumerate(base64_images, 1):
            output_path = None
            if output_dir:
                output_path = output_dir / f"{filename}__page_{page_idx}.jpg"
            result = self.extract_from_base64(base64_data, output_path=output_path)
            if result.text:
                all_texts.append(f"--- Page {page_idx} ---\n{result.text}")
                all_confidences.append(result.confidence)

        if not all_texts:
            return ("", 0.0)

        combined_text = "\n\n".join(all_texts)
        avg_confidence = sum(all_confidences) / len(all_confidences)

        return (combined_text, avg_confidence)

    def extract_batch(
        self,
        file_data: dict[str, list[str] | str],
        verbose: bool = True,
        output_dir: Optional[Path] = None,
    ) -> dict[str, tuple[str, float]]:
        """
        Extract text from multiple files with confidence scores.

        Args:
            file_data: Dictionary mapping filenames to base64 images.
                       Value is list[str] for multi-page PDFs, str for single images.
            verbose: Whether to print progress.

        Returns:
            Dictionary mapping filenames to (text, confidence) tuples.
        """
        results = {}

        for idx, (filename, images) in enumerate(file_data.items(), 1):
            if verbose:
                print(f"   📄 [{idx}/{len(file_data)}] Processing {filename}...")

            # Normalize to list
            if isinstance(images, str):
                images = [images]

            text, confidence = self.extract_from_images(
                filename, images, output_dir=output_dir
            )
            results[filename] = (text, confidence)

            if verbose:
                conf_emoji = "✓" if confidence >= 0.7 else "⚠️"
                print(f"      {conf_emoji} Confidence: {confidence:.2%}")

        return results

    def extract_batch_text_only(
        self,
        file_data: dict[str, list[str] | str],
        verbose: bool = True,
        output_dir: Optional[Path] = None,
    ) -> dict[str, str]:
        """
        Extract text from multiple files (text only, no confidence).

        Compatible interface with ContentExtractor.extract_batch().

        Args:
            file_data: Dictionary mapping filenames to base64 images.
            verbose: Whether to print progress.

        Returns:
            Dictionary mapping filenames to extracted text.
        """
        results_with_conf = self.extract_batch(
            file_data, verbose=verbose, output_dir=output_dir
        )
        return {filename: text for filename, (text, _) in results_with_conf.items()}


def load_paddleocr_output_dir(output_dir: Path) -> dict[str, str]:
    """
    Load extracted text from a PaddleOCR output directory.

    Expects JSON files named like: <original_filename>__page_<n>_res.json
    """
    pattern = re.compile(r"^(?P<original>.+)__page_(?P<page>\d+)_res$")
    grouped: dict[str, list[tuple[int, str]]] = defaultdict(list)

    for json_path in sorted(output_dir.glob("*_res.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        stem = json_path.stem
        match = pattern.match(stem)
        if match:
            original = match.group("original")
            page = int(match.group("page"))
        else:
            input_path = data.get("input_path", "")
            original = Path(input_path).name if input_path else stem.replace("_res", "")
            page = 1

        texts = data.get("rec_texts") or []
        if isinstance(texts, str):
            texts = [texts]
        page_text = "\n".join(text for text in texts if text).strip()
        grouped[original].append((page, page_text))

    extracted: dict[str, str] = {}
    for original, pages in grouped.items():
        pages_sorted = sorted(pages, key=lambda item: item[0])
        combined_parts: list[str] = []
        for page, text in pages_sorted:
            if len(pages_sorted) > 1:
                combined_parts.append(f"--- Page {page} ---\n{text}")
            else:
                combined_parts.append(text)
        combined_text = "\n\n".join(part for part in combined_parts if part).strip()
        extracted[original] = combined_text

    return extracted
