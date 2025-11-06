"""Utility functions for Knowledge Weaver"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """Configure logging using loguru"""
    if config is None:
        config = load_config()

    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
    log_file = log_config.get('log_file', './logs/knowledge_weaver.log')

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        lambda msg: print(msg, end=''),
        format=log_format,
        level=log_level,
        colorize=True
    )

    # Add file handler
    logger.add(
        log_file,
        format=log_format,
        level=log_level,
        rotation="10 MB",
        retention="7 days"
    )


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_json(path: str) -> Any:
    """Load JSON file"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Save data to JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def get_data_path(subdir: str = "") -> Path:
    """Get path to data directory"""
    root = get_project_root()
    data_path = root / "data" / subdir
    ensure_dir(data_path)
    return data_path


def get_model_path(subdir: str = "") -> Path:
    """Get path to models directory"""
    root = get_project_root()
    model_path = root / "models" / subdir
    ensure_dir(model_path)
    return model_path


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - chunk_overlap

    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Remove common artifacts
    text = text.replace('\x00', '')

    return text.strip()


class TripleFormat:
    """Utility class for formatting knowledge graph triples"""

    @staticmethod
    def to_string(subject: str, relation: str, obj: str) -> str:
        """Convert triple to string format: S:subject|R:relation|O:object"""
        return f"S:{subject}|R:{relation}|O:{obj}"

    @staticmethod
    def from_string(triple_str: str) -> tuple:
        """Parse triple string back to tuple"""
        parts = triple_str.split('|')
        subject = parts[0].replace('S:', '').strip()
        relation = parts[1].replace('R:', '').strip()
        obj = parts[2].replace('O:', '').strip()
        return (subject, relation, obj)

    @staticmethod
    def to_dict(subject: str, relation: str, obj: str, **metadata) -> Dict[str, Any]:
        """Convert triple to dictionary with metadata"""
        return {
            'subject': subject,
            'relation': relation,
            'object': obj,
            **metadata
        }

    @staticmethod
    def from_dict(triple_dict: Dict[str, Any]) -> tuple:
        """Extract triple from dictionary"""
        return (
            triple_dict['subject'],
            triple_dict['relation'],
            triple_dict['object']
        )
