"""
OCR Factory for managing different OCR backends.

Provides three modes:
- paddleocr: Fast, free, local OCR using PaddleOCR
- llm: Vision LLM-based OCR (best for handwriting)
- hybrid: PaddleOCR with automatic LLM fallback for low-confidence results

References:
- PaddleOCR GitHub: https://github.com/PaddlePaddle/PaddleOCR
"""

from enum import Enum
from pathlib import Path
from typing import Any, Optional, Protocol


class OCRBackend(Enum):
    """Available OCR backends for text extraction."""

    PADDLEOCR = "paddleocr"
    LLM = "llm"
    HYBRID = "hybrid"


class TextExtractor(Protocol):
    """Protocol for text extractors (duck typing interface)."""

    def extract_batch(
        self, file_data: dict[str, list[str] | str], **kwargs: Any
    ) -> dict[str, str]:
        """Extract text from multiple files."""
        ...


class HybridExtractor:
    """
    Hybrid OCR extractor that uses PaddleOCR with LLM fallback.

    For each file:
    1. First attempts extraction with PaddleOCR
    2. If confidence is below threshold, falls back to LLM extraction
    3. Returns the best result for each file
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        paddle_lang: str = "en",
        paddle_use_gpu: bool = False,
        llm_api_key: Optional[str] = None,
        llm_model: str = "openai/gpt-4o",
        llm_base_url: Optional[str] = None,
        llm_extra_headers: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the hybrid extractor.

        Args:
            confidence_threshold: Minimum confidence to accept PaddleOCR result.
                                  Below this, falls back to LLM.
            paddle_lang: Language for PaddleOCR.
            paddle_use_gpu: Whether to use GPU for PaddleOCR.
            llm_api_key: API key for LLM backend.
            llm_model: Model name for LLM extraction.
            llm_base_url: Base URL for LLM API.
            llm_extra_headers: Extra headers for LLM API requests.
        """
        self.confidence_threshold = confidence_threshold

        # Lazy initialization - only import when needed
        self._paddle_extractor = None
        self._llm_extractor = None

        # Store config for lazy init
        self._paddle_config = {
            "lang": paddle_lang,
            "use_gpu": paddle_use_gpu,
        }
        self._llm_config = {
            "api_key": llm_api_key,
            "model": llm_model,
            "base_url": llm_base_url,
            "extra_headers": llm_extra_headers,
        }

    @property
    def paddle_extractor(self):
        """Lazy-load PaddleOCR extractor."""
        if self._paddle_extractor is None:
            from .paddle_ocr_extractor import PaddleOCRExtractor

            self._paddle_extractor = PaddleOCRExtractor(**self._paddle_config)
        return self._paddle_extractor

    @property
    def llm_extractor(self):
        """Lazy-load LLM extractor."""
        if self._llm_extractor is None:
            from .content_extractor import ContentExtractor

            self._llm_extractor = ContentExtractor(**self._llm_config)
        return self._llm_extractor

    def extract_batch(
        self,
        file_data: dict[str, list[str] | str],
        batch_size: int = 10,
        verbose: bool = True,
        output_dir: Optional[Path] = None,
    ) -> dict[str, str]:
        """
        Extract text from multiple files using hybrid approach.

        Args:
            file_data: Dictionary mapping filenames to base64 images.
            batch_size: Batch size for LLM fallback processing.
            verbose: Whether to print progress.

        Returns:
            Dictionary mapping filenames to extracted text.
        """
        results: dict[str, str] = {}
        low_confidence_files: dict[str, list[str] | str] = {}

        # Step 1: Try PaddleOCR on all files
        if verbose:
            print("   🔍 Stage 1: PaddleOCR extraction...")

        paddle_results = self.paddle_extractor.extract_batch(
            file_data, verbose=verbose, output_dir=output_dir
        )

        # Step 2: Identify low-confidence files
        for filename, (text, confidence) in paddle_results.items():
            if confidence >= self.confidence_threshold:
                results[filename] = text
            else:
                low_confidence_files[filename] = file_data[filename]
                if verbose:
                    print(
                        f"      ⚠️  {filename}: confidence {confidence:.2%} < threshold, "
                        f"will use LLM fallback"
                    )

        # Step 3: LLM fallback for low-confidence files
        if low_confidence_files:
            if verbose:
                print(
                    f"\n   🤖 Stage 2: LLM fallback for {len(low_confidence_files)} files..."
                )

            llm_results = self.llm_extractor.extract_batch(
                low_confidence_files, batch_size=batch_size
            )
            results.update(llm_results)
        elif verbose:
            print("\n   ✅ All files extracted with high confidence, no LLM fallback needed")

        return results


def get_ocr_backend_from_string(backend_str: str) -> OCRBackend:
    """
    Convert a string to OCRBackend enum.

    Args:
        backend_str: Backend name (paddleocr, llm, hybrid).

    Returns:
        OCRBackend enum value.

    Raises:
        ValueError: If invalid backend name.
    """
    backend_map = {
        "paddleocr": OCRBackend.PADDLEOCR,
        "paddle": OCRBackend.PADDLEOCR,
        "llm": OCRBackend.LLM,
        "vision": OCRBackend.LLM,
        "hybrid": OCRBackend.HYBRID,
        "auto": OCRBackend.HYBRID,
    }

    normalized = backend_str.lower().strip()
    if normalized not in backend_map:
        valid = ", ".join(sorted(set(backend_map.keys())))
        raise ValueError(f"Invalid OCR backend '{backend_str}'. Valid options: {valid}")

    return backend_map[normalized]


def create_extractor(
    backend: OCRBackend,
    # PaddleOCR config
    paddle_lang: str = "en",
    paddle_use_gpu: bool = False,
    # LLM config
    llm_api_key: Optional[str] = None,
    llm_model: str = "openai/gpt-4o",
    llm_base_url: Optional[str] = None,
    llm_extra_headers: Optional[dict[str, str]] = None,
    # Hybrid config
    confidence_threshold: float = 0.7,
) -> TextExtractor:
    """
    Factory function to create the appropriate text extractor.

    Args:
        backend: Which OCR backend to use.
        paddle_lang: Language for PaddleOCR.
        paddle_use_gpu: Whether to use GPU for PaddleOCR.
        llm_api_key: API key for LLM backend.
        llm_model: Model for LLM extraction.
        llm_base_url: Base URL for LLM API.
        llm_extra_headers: Extra headers for LLM API.
        confidence_threshold: Confidence threshold for hybrid mode.

    Returns:
        Text extractor instance.
    """
    if backend == OCRBackend.PADDLEOCR:
        from .paddle_ocr_extractor import PaddleOCRExtractor

        extractor = PaddleOCRExtractor(lang=paddle_lang, use_gpu=paddle_use_gpu)
        # Wrap to provide compatible interface
        return _PaddleOCRWrapper(extractor)

    elif backend == OCRBackend.LLM:
        from .content_extractor import ContentExtractor

        return ContentExtractor(
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
            extra_headers=llm_extra_headers,
        )

    elif backend == OCRBackend.HYBRID:
        return HybridExtractor(
            confidence_threshold=confidence_threshold,
            paddle_lang=paddle_lang,
            paddle_use_gpu=paddle_use_gpu,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_extra_headers=llm_extra_headers,
        )

    else:
        raise ValueError(f"Unknown OCR backend: {backend}")


class _PaddleOCRWrapper:
    """Wrapper to make PaddleOCRExtractor compatible with TextExtractor protocol."""

    def __init__(self, extractor):
        self._extractor = extractor

    def extract_batch(
        self, file_data: dict[str, list[str] | str], **kwargs: Any
    ) -> dict[str, str]:
        """Extract text using PaddleOCR (text only, no confidence in output)."""
        verbose = kwargs.get("verbose", True)
        output_dir = kwargs.get("output_dir")
        return self._extractor.extract_batch_text_only(
            file_data, verbose=verbose, output_dir=output_dir
        )
