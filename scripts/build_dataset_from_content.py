"""Build training dataset from collected YouTube transcripts and articles"""

import json
import sys
from pathlib import Path
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config
from src.models.dataset_builder import DatasetBuilder, TrainingExample
from src.agents.extractor import BaselineExtractor
from loguru import logger


def extract_triples_from_text(text: str, extractor: BaselineExtractor, source: str):
    """Extract triples using baseline extractor"""
    # Split into chunks (max 300 words per chunk)
    words = text.split()
    chunks = []

    for i in range(0, len(words), 300):
        chunk = ' '.join(words[i:i+300])
        if len(chunk.split()) > 50:  # Only keep substantial chunks
            chunks.append(chunk)

    examples = []
    for chunk in chunks[:10]:  # Limit to 10 chunks per file
        triples = extractor.extract_triples(chunk, source=source)

        if len(triples) >= 2:  # Only keep if we extracted multiple triples
            triple_strings = [f"S:{t.subject}|R:{t.relation}|O:{t.object}" for t in triples]

            example = TrainingExample(
                input_text=chunk,
                output_triples=triple_strings,
                metadata={'source': source}
            )
            examples.append(example)

    return examples


def build_dataset():
    """Build training dataset from collected content"""
    logger.info("Building training dataset from collected content...")

    # Load config
    config = load_config('configs/config.yaml')

    # Initialize extractor
    extractor = BaselineExtractor(config)

    # Initialize dataset builder
    builder = DatasetBuilder(config)

    # Paths
    data_dir = Path('data/raw/webpages')

    # Process YouTube transcripts
    logger.info("Processing YouTube transcripts...")
    youtube_files = list(data_dir.glob('youtube_*.md'))

    for file_path in youtube_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract transcript section
            if '## Transcript' in content:
                transcript = content.split('## Transcript')[1].strip()

                examples = extract_triples_from_text(
                    transcript,
                    extractor,
                    source=f'youtube:{file_path.stem}'
                )

                builder.examples.extend(examples)
                logger.info(f"Extracted {len(examples)} examples from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Process articles
    logger.info("Processing articles...")
    article_files = list(data_dir.glob('article_*.md'))

    for file_path in article_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract full content section
            if '## Full Content' in content:
                article_content = content.split('## Full Content')[1].strip()
            else:
                article_content = content

            examples = extract_triples_from_text(
                article_content,
                extractor,
                source=f'article:{file_path.stem}'
            )

            builder.examples.extend(examples)
            logger.info(f"Extracted {len(examples)} examples from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Process PDFs
    logger.info("Processing PDF extracts...")
    pdf_files = list(data_dir.glob('pdf_extracted_*.md'))

    for file_path in pdf_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            examples = extract_triples_from_text(
                content,
                extractor,
                source=f'pdf:{file_path.stem}'
            )

            builder.examples.extend(examples)
            logger.info(f"Extracted {len(examples)} examples from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # Add synthetic examples to augment the dataset
    logger.info("Generating synthetic examples...")
    builder.build_synthetic_examples(num_examples=50)

    # Add Wikipedia examples for common ML/AI topics
    logger.info("Adding Wikipedia examples...")
    topics = [
        'Machine Learning', 'Deep Learning', 'Natural Language Processing',
        'Knowledge Graph', 'Large Language Model', 'Retrieval Augmented Generation',
        'Neural Network', 'Transformer (machine learning model)',
        'Fine-tuning (machine learning)', 'Parameter-efficient fine-tuning'
    ]
    builder.build_from_wikipedia(topics, max_per_topic=5)

    logger.info(f"Total examples: {len(builder.examples)}")

    # Split into train/val/test (80/10/10)
    random.shuffle(builder.examples)

    total = len(builder.examples)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_examples = builder.examples[:train_size]
    val_examples = builder.examples[train_size:train_size + val_size]
    test_examples = builder.examples[train_size + val_size:]

    logger.info(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    # Save datasets
    output_dir = Path('data/datasets')
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_split(examples, split_name):
        output_path = output_dir / f'{split_name}.json'
        data = [ex.to_dict() for ex in examples]
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(examples)} examples to {output_path}")

    save_split(train_examples, 'train')
    save_split(val_examples, 'val')
    save_split(test_examples, 'test')

    logger.success(f"Dataset creation complete! Total examples: {total}")

    # Show sample
    logger.info("\nSample training example:")
    sample = train_examples[0]
    logger.info(f"Input: {sample.input_text[:200]}...")
    logger.info(f"Output: {sample.output_triples[:3]}")


if __name__ == '__main__':
    build_dataset()
