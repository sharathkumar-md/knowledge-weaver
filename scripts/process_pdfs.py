"""
PDF Processing Script
Downloads and extracts text from saved PDF URLs
"""

import json
from pathlib import Path
import requests
from typing import List, Dict
import PyPDF2
import io
from loguru import logger

# Configure paths
DATA_DIR = Path(__file__).parent.parent / 'data'
WEBPAGES_DIR = DATA_DIR / 'raw' / 'webpages'
PAGES_LOG = DATA_DIR / 'raw' / 'pages_log.json'
PDF_DIR = DATA_DIR / 'raw' / 'pdfs'
PDF_DIR.mkdir(parents=True, exist_ok=True)


def get_saved_pdfs() -> List[Dict]:
    """Get all saved PDF URLs from pages log"""
    if not PAGES_LOG.exists():
        return []

    with open(PAGES_LOG, 'r', encoding='utf-8') as f:
        pages = json.load(f)

    # Filter only PDFs
    pdfs = [p for p in pages if p.get('type') == 'pdf']
    logger.info(f"Found {len(pdfs)} saved PDFs")
    return pdfs


def download_pdf(url: str, filename: str) -> bool:
    """Download PDF from URL"""
    try:
        logger.info(f"Downloading: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        filepath = PDF_DIR / filename
        with open(filepath, 'wb') as f:
            f.write(response.content)

        logger.success(f"Downloaded to: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_text_from_pdf(filepath: Path) -> str:
    """Extract text from PDF file"""
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = []
            for page in reader.pages:
                text.append(page.extract_text())

            full_text = '\n\n'.join(text)
            logger.info(f"Extracted {len(full_text)} characters from {filepath.name}")
            return full_text
    except Exception as e:
        logger.error(f"Failed to extract text from {filepath}: {e}")
        return ""


def save_extracted_text(pdf_info: Dict, text: str):
    """Save extracted text as markdown"""
    if not text:
        return

    # Create markdown filename
    title = pdf_info.get('title', 'PDF').replace('.pdf', '')
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    filename = f"pdf_extracted_{safe_title}.md"
    filepath = WEBPAGES_DIR / filename

    # Create markdown content
    markdown = f"""# {pdf_info.get('title', 'PDF Document')}

**URL:** {pdf_info['url']}
**Type:** pdf
**Domain:** {pdf_info.get('domain', 'unknown')}
**Saved At:** {pdf_info.get('savedAt', '')}

---

## Extracted Content

{text}
"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(markdown)

    logger.success(f"Saved extracted text to: {filepath}")


def process_all_pdfs():
    """Download and process all saved PDFs"""
    pdfs = get_saved_pdfs()

    if not pdfs:
        logger.info("No PDFs to process")
        return

    for pdf_info in pdfs:
        url = pdf_info['url']
        title = pdf_info.get('title', 'document')

        # Create safe filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
        pdf_filename = f"{safe_title}.pdf"
        pdf_filepath = PDF_DIR / pdf_filename

        # Skip if already downloaded
        if pdf_filepath.exists():
            logger.info(f"Already downloaded: {pdf_filename}")
        else:
            # Download PDF
            if not download_pdf(url, pdf_filename):
                continue

        # Extract text
        text = extract_text_from_pdf(pdf_filepath)

        # Save as markdown
        if text:
            save_extracted_text(pdf_info, text)


if __name__ == '__main__':
    logger.info("Starting PDF processing...")
    logger.info(f"PDF directory: {PDF_DIR}")
    logger.info(f"Output directory: {WEBPAGES_DIR}")

    try:
        process_all_pdfs()
        logger.success("PDF processing complete!")
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
