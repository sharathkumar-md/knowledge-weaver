"""Script to prepare training dataset for LoRA fine-tuning"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dataset_builder import DatasetBuilder
from src.utils import load_config, setup_logging


def main():
    print(f"\n{'='*60}")
    print("Knowledge Weaver - Dataset Preparation")
    print(f"{'='*60}\n")

    # Load config
    config = load_config('configs/config.yaml')
    setup_logging(config)

    # Initialize dataset builder
    builder = DatasetBuilder(config)

    # Wikipedia topics to fetch
    wikipedia_topics = [
        # Machine Learning
        'Artificial neural network',
        'Deep learning',
        'Machine learning',
        'Supervised learning',
        'Reinforcement learning',

        # Computer Science
        'Algorithm',
        'Data structure',
        'Binary search tree',
        'Graph (abstract data type)',
        'Hash table',

        # Mathematics
        'Linear algebra',
        'Calculus',
        'Probability theory',
        'Statistics',
        'Optimization (mathematics)',

        # Physics
        'Classical mechanics',
        'Quantum mechanics',
        'Thermodynamics',

        # Biology
        'Cell (biology)',
        'DNA',
        'Evolution',

        # General knowledge
        'Scientific method',
        'Logic',
        'Philosophy of science',
    ]

    print("Step 1: Fetching Wikipedia examples...")
    print(f"Topics: {len(wikipedia_topics)}")
    builder.build_from_wikipedia(wikipedia_topics, max_per_topic=5)

    print("\nStep 2: Generating synthetic examples...")
    builder.build_synthetic_examples(num_examples=200)

    print("\nStep 3: Loading manual annotations (if available)...")
    annotation_file = './data/datasets/manual_annotations.json'
    builder.add_manual_annotations(annotation_file)

    print(f"\nTotal examples collected: {len(builder.examples)}")

    # Save datasets
    print("\nStep 4: Saving datasets...")
    output_dir = './data/datasets'

    # Save in multiple formats
    builder.save_datasets(output_dir, format='json')
    print(f"  - Saved JSON format to {output_dir}/")

    builder.save_datasets(output_dir + '/instruction', format='instruction')
    print(f"  - Saved instruction format to {output_dir}/instruction/")

    # Print summary
    print(f"\n{'='*60}")
    print("Dataset Summary")
    print(f"{'='*60}\n")

    train_path = Path(output_dir) / 'train.json'
    val_path = Path(output_dir) / 'val.json'
    test_path = Path(output_dir) / 'test.json'

    import json
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    with open(test_path, 'r') as f:
        test_data = json.load(f)

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Test examples: {len(test_data)}")
    print(f"Total: {len(train_data) + len(val_data) + len(test_data)}")

    # Show sample
    print(f"\n{'='*60}")
    print("Sample Training Example")
    print(f"{'='*60}\n")

    sample = train_data[0]
    print("Input:")
    print(sample['input'][:300] + "..." if len(sample['input']) > 300 else sample['input'])
    print("\nOutput triples:")
    print(sample['output'])
    print("\nMetadata:")
    print(sample['metadata'])

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}\n")

    print("Next steps:")
    print("1. Review the dataset in ./data/datasets/")
    print("2. Optionally add manual annotations to manual_annotations.json")
    print("3. Run training script: python scripts/train_lora.py")


if __name__ == '__main__':
    main()
