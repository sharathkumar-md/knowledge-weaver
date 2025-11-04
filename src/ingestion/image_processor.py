"""Image processor with OCR support"""

from pathlib import Path
from typing import Dict, Any
import pytesseract
from PIL import Image
from .base_processor import BaseProcessor
from loguru import logger


class ImageProcessor(BaseProcessor):
    """Processor for images with OCR (screenshots, scanned docs)"""

    SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ocr_enabled = config.get('ingestion', {}).get('ocr_enabled', True)

    def can_process(self, file_path: Path) -> bool:
        """Check if file is an image"""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract_text(self, file_path: Path) -> str:
        """Extract text from image using OCR"""
        if not self.ocr_enabled:
            logger.warning(f"OCR disabled, skipping image: {file_path.name}")
            return ""

        try:
            # Open image
            image = Image.open(file_path)

            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Perform OCR
            text = pytesseract.image_to_string(image)

            if text.strip():
                logger.info(f"Extracted text from image with OCR: {file_path.name}")
            else:
                logger.warning(f"No text found in image: {file_path.name}")

            return text

        except pytesseract.TesseractNotFoundError:
            logger.error("Tesseract OCR not installed. Please install Tesseract.")
            logger.error("Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            logger.error("Linux: sudo apt-get install tesseract-ocr")
            logger.error("Mac: brew install tesseract")
            return ""

        except Exception as e:
            logger.error(f"Failed to process image {file_path}: {e}")
            return ""

    def _create_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create metadata with image-specific info"""
        metadata = super()._create_metadata(file_path)

        try:
            image = Image.open(file_path)

            metadata['image_format'] = image.format
            metadata['image_mode'] = image.mode
            metadata['image_size'] = image.size  # (width, height)
            metadata['image_width'] = image.size[0]
            metadata['image_height'] = image.size[1]

            # Extract EXIF data if available
            if hasattr(image, '_getexif') and image._getexif():
                exif = image._getexif()
                if exif:
                    metadata['has_exif'] = True

        except Exception as e:
            logger.warning(f"Failed to extract image metadata from {file_path.name}: {e}")

        return metadata
