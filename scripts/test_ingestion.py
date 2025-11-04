"""Test script for the ingestion pipeline"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.pipeline import IngestionPipeline
from src.utils import load_config, setup_logging


def main():
    # Load config
    config = load_config('configs/config.yaml')
    setup_logging(config)

    # Initialize pipeline
    pipeline = IngestionPipeline(config)

    # Test with sample data
    input_dir = './data/raw'
    output_dir = './data/processed'

    print(f"\n{'='*60}")
    print("Testing Ingestion Pipeline")
    print(f"{'='*60}\n")

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Run pipeline
    documents = pipeline.run(input_dir, output_dir)

    # Display results
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}\n")

    print(f"Total documents processed: {len(documents)}")

    for idx, doc in enumerate(documents, 1):
        print(f"\nDocument {idx}:")
        print(f"  Source: {doc.metadata['filename']}")
        print(f"  Type: {doc.metadata['file_type']}")
        print(f"  Size: {doc.metadata['size_bytes']} bytes")
        print(f"  Chunks: {len(doc.chunks)}")
        print(f"  Content length: {len(doc.content)} characters")
        print(f"  Preview: {doc.content[:200]}...")

    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
