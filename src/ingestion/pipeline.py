"""Main ingestion pipeline that orchestrates all processors"""

from pathlib import Path
from typing import Dict, Any, List
import json
from tqdm import tqdm
from loguru import logger

from .base_processor import BaseProcessor, Document
from .markdown_processor import MarkdownProcessor
from .pdf_processor import PDFProcessor
from .docx_processor import DOCXProcessor
from .image_processor import ImageProcessor
from .video_processor import VideoProcessor


class IngestionPipeline:
    """Orchestrates document ingestion using multiple processors"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize all processors
        self.processors: List[BaseProcessor] = [
            MarkdownProcessor(config),
            PDFProcessor(config),
            DOCXProcessor(config),
            ImageProcessor(config),
            VideoProcessor(config),
        ]

        # Get supported file extensions from config
        self.supported_formats = set(
            config.get('ingestion', {}).get('supported_formats', [
                '.md', '.txt', '.pdf', '.docx', '.png', '.jpg', '.jpeg'
            ])
        )

        logger.info(f"Initialized ingestion pipeline with {len(self.processors)} processors")

    def run(self, input_path: str, output_path: str = None) -> List[Document]:
        """Run the ingestion pipeline on input directory"""
        input_dir = Path(input_path)

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_path}")

        # Collect all files to process
        files_to_process = self._collect_files(input_dir)

        if not files_to_process:
            logger.warning(f"No supported files found in {input_path}")
            return []

        logger.info(f"Found {len(files_to_process)} files to process")

        # Process each file
        documents = []
        failed_files = []

        for file_path in tqdm(files_to_process, desc="Processing documents"):
            try:
                doc = self._process_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                failed_files.append(str(file_path))

        # Save results if output path specified
        if output_path:
            self._save_documents(documents, output_path)

        # Report statistics
        logger.info(f"Successfully processed {len(documents)} documents")
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")

        return documents

    def _collect_files(self, input_dir: Path) -> List[Path]:
        """Collect all supported files from input directory"""
        files = []

        # Walk through directory
        for file_path in input_dir.rglob('*'):
            if file_path.is_file():
                # Check if supported
                if self._is_supported_file(file_path):
                    files.append(file_path)

        return sorted(files)

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file is supported by any processor"""
        # Check extension
        if file_path.suffix.lower() in self.supported_formats:
            return True

        # Check with processors
        for processor in self.processors:
            if processor.can_process(file_path):
                return True

        return False

    def _process_file(self, file_path: Path) -> Document:
        """Process a single file using appropriate processor"""
        # Find matching processor
        for processor in self.processors:
            if processor.can_process(file_path):
                logger.debug(f"Processing {file_path.name} with {processor.__class__.__name__}")
                return processor.process(file_path)

        logger.warning(f"No processor found for {file_path.name}")
        return None

    def _save_documents(self, documents: List[Document], output_path: str) -> None:
        """Save processed documents to output directory"""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save each document
        for idx, doc in enumerate(documents):
            # Create filename based on source
            source_name = Path(doc.metadata['source']).stem
            output_file = output_dir / f"{idx:04d}_{source_name}.json"

            # Prepare data
            doc_data = {
                'content': doc.content,
                'chunks': doc.chunks,
                'metadata': doc.metadata
            }

            # Save as JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

        # Save summary
        summary = {
            'total_documents': len(documents),
            'total_chunks': sum(len(doc.chunks) for doc in documents),
            'source_files': [doc.metadata['source'] for doc in documents]
        }

        summary_file = output_dir / '_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(documents)} documents to {output_path}")

    def process_single_file(self, file_path: str) -> Document:
        """Process a single file and return Document"""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        return self._process_file(path)
