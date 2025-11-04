"""Script to train LoRA model"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train_lora import train_lora
from src.utils import load_config, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train LoRA model for knowledge extraction")

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='./data/datasets',
        help='Path to dataset directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./models/lora/extractor_v1',
        help='Output directory for trained model'
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Knowledge Weaver - LoRA Training")
    print(f"{'='*60}\n")

    # Load config
    config = load_config(args.config)
    setup_logging(config)

    print(f"Configuration: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print()

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset directory not found: {args.dataset}")
        print("Please run: python scripts/prepare_dataset.py")
        return

    # Train
    train_lora(config, args.dataset, args.output)

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")

    print(f"Model saved to: {args.output}/final")
    print("\nNext steps:")
    print("1. Update config.yaml to set use_lora: true")
    print("2. Set LORA_MODEL_PATH environment variable")
    print("3. Run extraction: python main.py extract --input ./data/processed")


if __name__ == '__main__':
    main()
