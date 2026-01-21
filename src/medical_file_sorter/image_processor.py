"""
ImageProcessor class for converting files to Base64 encoded strings.
Handles PDF pages and images for LLM processing.
"""

import base64
import io
from pathlib import Path
from typing import Optional

from pdf2image import convert_from_path
from PIL import Image

# Register HEIC support with PIL
import pillow_heif
pillow_heif.register_heif_opener()


class ImageProcessor:
    """
    Handles conversion of medical documents (PDFs and images) to Base64 encoded strings
    for processing by multimodal LLMs.
    """

    VALID_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".heic"}
    MAX_DIMENSION = 1000  # Maximum dimension for resizing

    def __init__(self, max_dimension: int = 1000):
        """
        Initialize the ImageProcessor.

        Args:
            max_dimension: Maximum dimension (width or height) for resized images.
        """
        self.max_dimension = max_dimension

    def is_valid_file(self, file_path: Path) -> bool:
        """
        Check if a file is a valid medical document type.

        Args:
            file_path: Path to the file.

        Returns:
            True if the file is a valid type, False otherwise.
        """
        if file_path.name.startswith("."):
            return False
        return file_path.suffix.lower() in self.VALID_EXTENSIONS

    def get_valid_files(self, folder_path: Path) -> list[Path]:
        """
        Get all valid files from a folder.

        Args:
            folder_path: Path to the folder.

        Returns:
            List of valid file paths.
        """
        if not folder_path.is_dir():
            raise ValueError(f"Path is not a directory: {folder_path}")

        valid_files = []
        for file_path in folder_path.iterdir():
            if file_path.is_file() and self.is_valid_file(file_path):
                valid_files.append(file_path)

        return sorted(valid_files)

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """
        Resize an image to fit within max_dimension while maintaining aspect ratio.

        Args:
            image: PIL Image object.

        Returns:
            Resized PIL Image object.
        """
        width, height = image.size

        if width <= self.max_dimension and height <= self.max_dimension:
            return image

        if width > height:
            new_width = self.max_dimension
            new_height = int(height * (self.max_dimension / width))
        else:
            new_height = self.max_dimension
            new_width = int(width * (self.max_dimension / height))

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert a PIL Image to a Base64 encoded string.

        Args:
            image: PIL Image object.

        Returns:
            Base64 encoded string.
        """
        # Convert to RGB if necessary (for RGBA or P mode images)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    def process_pdf(self, file_path: Path) -> Optional[list[str]]:
        """
        Convert all pages of a PDF to Base64 encoded strings.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of Base64 encoded strings (one per page), or None if conversion fails.
        """
        try:
            # Convert all pages
            images = convert_from_path(file_path)
            if not images:
                return None

            result = []
            for page_image in images:
                resized = self._resize_image(page_image)
                result.append(self._image_to_base64(resized))
            
            return result
        except Exception as e:
            print(f"Warning: Failed to process PDF {file_path.name}: {e}")
            return None

    def process_pdf_first_page(self, file_path: Path) -> Optional[str]:
        """
        Convert only the first page of a PDF to a Base64 encoded string.
        (Legacy method for backwards compatibility)

        Args:
            file_path: Path to the PDF file.

        Returns:
            Base64 encoded string of the first page, or None if conversion fails.
        """
        try:
            images = convert_from_path(file_path, first_page=1, last_page=1)
            if not images:
                return None

            first_page = images[0]
            resized = self._resize_image(first_page)
            return self._image_to_base64(resized)
        except Exception as e:
            print(f"Warning: Failed to process PDF {file_path.name}: {e}")
            return None

    def process_image(self, file_path: Path) -> Optional[str]:
        """
        Convert an image file to a Base64 encoded string.

        Args:
            file_path: Path to the image file.

        Returns:
            Base64 encoded string, or None if conversion fails.
        """
        try:
            with Image.open(file_path) as image:
                resized = self._resize_image(image)
                return self._image_to_base64(resized)
        except Exception as e:
            print(f"Warning: Failed to process image {file_path.name}: {e}")
            return None

    def process_file(self, file_path: Path) -> Optional[list[str] | str]:
        """
        Process a file and return its Base64 encoded representation.

        Args:
            file_path: Path to the file.

        Returns:
            For PDFs: List of Base64 strings (one per page).
            For images: Single Base64 string.
            None if processing fails.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self.process_pdf(file_path)
        elif suffix in {".jpg", ".jpeg", ".png", ".heic"}:
            return self.process_image(file_path)
        else:
            print(f"Warning: Unsupported file type: {file_path.name}")
            return None

    def process_folder(self, folder_path: Path) -> dict[str, list[str] | str]:
        """
        Process all valid files in a folder and return their Base64 representations.

        Args:
            folder_path: Path to the folder.

        Returns:
            Dictionary mapping filenames to:
            - list[str] for PDFs (one base64 per page)
            - str for images (single base64)
        """
        valid_files = self.get_valid_files(folder_path)

        if not valid_files:
            return {}

        results: dict[str, list[str] | str] = {}
        for file_path in valid_files:
            base64_data = self.process_file(file_path)
            if base64_data:
                results[file_path.name] = base64_data

        return results

