"""Base processor class for document ingestion"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Document:
    """Represents a processed document"""
    content: str
    metadata: Dict[str, Any]
    chunks: List[str] = None

    def __post_init__(self):
        if self.chunks is None:
            self.chunks = []


class BaseProcessor(ABC):
    """Abstract base class for document processors"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chunk_size = config.get('ingestion', {}).get('chunk_size', 512)
        self.chunk_overlap = config.get('ingestion', {}).get('chunk_overlap', 50)

    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the given file"""
        pass

    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Extract raw text from the file"""
        pass

    def process(self, file_path: Path) -> Document:
        """Process a file and return a Document object"""
        # Extract text
        text = self.extract_text(file_path)

        # Clean text
        text = self._clean_text(text)

        # Create metadata
        metadata = self._create_metadata(file_path)

        # Create document
        doc = Document(content=text, metadata=metadata)

        # Chunk the text
        doc.chunks = self._chunk_text(text)

        return doc

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove null bytes
        text = text.replace('\x00', '')

        # Normalize whitespace
        text = ' '.join(text.split())

        # Remove excessive newlines
        while '\n\n\n' in text:
            text = text.replace('\n\n\n', '\n\n')

        return text.strip()

    def _create_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create metadata for the document"""
        stat = file_path.stat()

        return {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix,
            'size_bytes': stat.st_size,
            'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'processed_at': datetime.now().isoformat(),
            'processor': self.__class__.__name__
        }

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n']:
                    last_punct = text[start:end].rfind(punct)
                    if last_punct != -1:
                        end = start + last_punct + len(punct)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.chunk_overlap

        return chunks
