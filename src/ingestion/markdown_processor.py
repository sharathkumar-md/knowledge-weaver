"""Markdown document processor"""

from pathlib import Path
from typing import Dict, Any
import markdown
from bs4 import BeautifulSoup
from .base_processor import BaseProcessor


class MarkdownProcessor(BaseProcessor):
    """Processor for Markdown files (.md, .markdown)"""

    SUPPORTED_EXTENSIONS = {'.md', '.markdown', '.txt'}

    def can_process(self, file_path: Path) -> bool:
        """Check if file is a Markdown document"""
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def extract_text(self, file_path: Path) -> str:
        """Extract text from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # Convert Markdown to HTML
            html = markdown.markdown(md_content, extensions=['extra', 'codehilite'])

            # Extract text from HTML
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator='\n')

            return text

        except Exception as e:
            raise RuntimeError(f"Failed to process Markdown file {file_path}: {e}")

    def _create_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Create metadata with Markdown-specific info"""
        metadata = super()._create_metadata(file_path)

        # Try to extract frontmatter (YAML metadata)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if content.startswith('---'):
                # Simple frontmatter extraction
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = parts[1].strip()
                    # Parse as YAML (simplified)
                    for line in frontmatter.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            metadata[f'fm_{key.strip()}'] = value.strip()

        except Exception:
            pass  # Frontmatter extraction is optional

        return metadata
