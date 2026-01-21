"""
ContentExtractor class for extracting text content from documents using vision LLM.
Stage 1 of the two-stage pipeline: Vision model OCR.
"""

import re
from typing import Any, Optional

from openai import OpenAI


# System prompt for batch content extraction
BATCH_EXTRACTION_SYSTEM_PROMPT = """You are a document OCR specialist. Your task is to extract ALL text content from the provided document images.

You will receive multiple files, each possibly with multiple pages. Each image is labeled with its filename and page number.

For EACH file, extract:
1. **All visible text** - headers, body text, footnotes, watermarks
2. **Structured data** - tables, lists, forms with their formatting preserved
3. **Key identifiers** - dates, amounts, names, reference numbers, addresses

Rules:
- Preserve the document structure (headings, paragraphs, bullet points)
- For tables, use markdown table format
- For handwritten text, do your best to transcribe
- If text is unclear, mark as [UNCLEAR: best guess]

OUTPUT FORMAT - CRITICAL:
You MUST structure your response exactly like this for EACH file:

=== FILE START: {exact_filename} ===
{extracted text content for this file, all pages combined}
=== FILE END: {exact_filename} ===

Use the EXACT filename provided. Separate each file's content clearly."""


class ContentExtractor:
    """
    Extracts text content from images/PDFs using a vision LLM.
    Stage 1 of the two-stage document processing pipeline.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o",
        base_url: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the ContentExtractor.

        Args:
            api_key: API key for the LLM service.
            model: Model to use for vision extraction.
            base_url: Custom API base URL.
            extra_headers: Additional headers for API requests.
        """
        self.model = model
        self._extra_headers = extra_headers or {}

        # Initialize the OpenAI client
        client_kwargs = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        else:
            client_kwargs["api_key"] = "lm-studio"

        self.client = OpenAI(**client_kwargs)

    def _create_image_content(self, base64_data: str) -> dict[str, Any]:
        """
        Create the image content block for the API request.

        Args:
            base64_data: Base64 encoded image data.

        Returns:
            Image content dictionary.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_data}",
                "detail": "high",  # Use high detail for OCR accuracy
            },
        }

    def _parse_batch_response(self, response_text: str, filenames: list[str]) -> dict[str, str]:
        """
        Parse the batch response to extract content for each file.

        Args:
            response_text: The raw LLM response.
            filenames: List of expected filenames.

        Returns:
            Dictionary mapping filenames to extracted content.
        """
        results = {}
        
        # Try to parse using delimiters
        pattern = r"=== FILE START: (.+?) ===\s*(.*?)\s*=== FILE END: \1 ==="
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for filename, content in matches:
            filename = filename.strip()
            results[filename] = content.strip()
        
        # Fallback: if some files are missing, try a simpler split
        if len(results) < len(filenames):
            # Try alternate pattern without FILE END
            alt_pattern = r"=== FILE START: (.+?) ===\s*(.*?)(?==== FILE START:|$)"
            alt_matches = re.findall(alt_pattern, response_text, re.DOTALL)
            
            for filename, content in alt_matches:
                filename = filename.strip()
                if filename not in results:
                    # Clean up content (remove any trailing FILE END markers)
                    content = re.sub(r"=== FILE END:.*?===\s*$", "", content, flags=re.DOTALL)
                    results[filename] = content.strip()
        
        # For any files still missing, assign empty string
        for filename in filenames:
            if filename not in results:
                print(f"   ⚠️  Warning: Could not parse content for {filename}")
                results[filename] = ""
        
        return results

    def extract_files_in_batch(
        self, file_batch: list[tuple[str, list[str]]]
    ) -> dict[str, str]:
        """
        Extract text content from multiple files in a single LLM call.

        Args:
            file_batch: List of (filename, list_of_base64_images) tuples.

        Returns:
            Dictionary mapping filenames to extracted text content.
        """
        if not file_batch:
            return {}

        filenames = [f[0] for f in file_batch]
        
        # Build the user message with all files and pages
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": f"Extract text from the following {len(file_batch)} medical document(s). "
                        f"Files: {', '.join(filenames)}",
            }
        ]

        # Add each file's images with clear labels
        for filename, images in file_batch:
            total_pages = len(images)
            for page_idx, base64_data in enumerate(images, 1):
                # Add text label before each image
                user_content.append({
                    "type": "text",
                    "text": f"=== FILE: {filename} (Page {page_idx}/{total_pages}) ===",
                })
                user_content.append(self._create_image_content(base64_data))

        try:
            # Build request
            request_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": BATCH_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": 16384,  # Larger for batch processing
                "temperature": 0.1,
            }

            if self._extra_headers:
                request_kwargs["extra_headers"] = self._extra_headers

            response = self.client.chat.completions.create(**request_kwargs)

            if not response.choices:
                print(f"   ⚠️  Warning: No response for batch")
                return {f: "" for f in filenames}

            content = response.choices[0].message.content
            if not content:
                return {f: "" for f in filenames}

            # Parse the response to extract content for each file
            return self._parse_batch_response(content.strip(), filenames)

        except Exception as e:
            print(f"   ❌ Error in batch extraction: {e}")
            # Return empty results for all files in this batch
            return {f: "" for f in filenames}

    def extract_from_images(
        self, filename: str, base64_images: list[str]
    ) -> str:
        """
        Extract text content from one file (single file mode, used as fallback).

        Args:
            filename: Name of the file being processed.
            base64_images: List of Base64 encoded images (one per page).

        Returns:
            Extracted text content from all pages.
        """
        # Use batch method with single file
        result = self.extract_files_in_batch([(filename, base64_images)])
        return result.get(filename, "")

    def extract_batch(
        self,
        file_data: dict[str, list[str] | str],
        batch_size: int = 10,
        **_: Any,
    ) -> dict[str, str]:
        """
        Extract text content from multiple files using batch processing.

        Args:
            file_data: Dictionary mapping filenames to base64 images.
                       Value is list[str] for multi-page PDFs, str for single images.
            batch_size: Number of files to process in a single LLM call.

        Returns:
            Dictionary mapping filenames to extracted text content.
        """
        results = {}
        
        # Normalize all entries to (filename, list[str]) format
        file_list: list[tuple[str, list[str]]] = []
        for filename, images in file_data.items():
            if isinstance(images, str):
                file_list.append((filename, [images]))
            else:
                file_list.append((filename, images))
        
        # Process in batches
        total_files = len(file_list)
        for i in range(0, total_files, batch_size):
            batch = file_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_files + batch_size - 1) // batch_size
            
            # Count pages in this batch
            total_pages = sum(len(images) for _, images in batch)
            batch_filenames = [f for f, _ in batch]
            
            print(f"   📦 Batch {batch_num}/{total_batches}: {len(batch)} files, {total_pages} pages")
            print(f"      Files: {', '.join(batch_filenames)}")
            
            batch_results = self.extract_files_in_batch(batch)
            results.update(batch_results)
        
        return results
