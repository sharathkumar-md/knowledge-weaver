"""PDF document processor"""

from pathlib import Path
from typing import Dict, Any
import PyPDF2
import pdfplumber
from .base_processor import BaseProcessor
from loguru import logger


class PDFProcessor(BaseProcessor):
    """Processor for PDF files"""

    SUPPORTED_EXTENSIONS = {'.pdf'}

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a PDF"""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF using multiple methods"""
        text = None

        # Try pdfplumber first (better for complex layouts)
        try:
            text = self._extract_with_pdfplumber(file_path)
            if text and len(text.strip()) > 100:
                logger.info(f"Extracted PDF with pdfplumber: {file_path.name}")
                return text
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path.name}: {e}")

        # Fallback to PyPDF2
        try:
            text = self._extract_with_pypdf2(file_path)
            if text and len(text.strip()) > 100:
                logger.info(f"Extracted PDF with PyPDF2: {file_path.name}")
                return text
        except Exception as e:
            logger.warning(f"PyPDF2 failed for {file_path.name}: {e}")

        # If both fail, return empty or raise
        if not text or len(text.strip()) < 50:
            logger.error(f"Failed to extract meaningful text from {file_path.name}")
            return ""

        return text

    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber"""
        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num}]\n{page_text}")

        return '\n\n'.join(text_parts)

    def _extract_with_pypdf2(self, file_path: Path) -> str:
        """Extract text using PyPDF2"""
        text_parts = []

        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)

            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {page_num}]\n{page_text}")

        return '\n\n'.join(text_parts)

    def _create_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create metadata with PDF-specific info"""
        metadata = super()._create_metadata(file_path)

        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)

                metadata['num_pages'] = len(pdf_reader.pages)

                # Extract PDF metadata
                pdf_info = pdf_reader.metadata
                if pdf_info:
                    if pdf_info.title:
                        metadata['title'] = pdf_info.title
                    if pdf_info.author:
                        metadata['author'] = pdf_info.author
                    if pdf_info.subject:
                        metadata['subject'] = pdf_info.subject
                    if pdf_info.creator:
                        metadata['creator'] = pdf_info.creator

        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata from {file_path.name}: {e}")

        return metadata
