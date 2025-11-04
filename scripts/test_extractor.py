"""Test script for the baseline extractor"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.extractor import BaselineExtractor
from src.utils import load_config, setup_logging


def main():
    # Load config
    config = load_config('configs/config.yaml')
    setup_logging(config)

    # Initialize extractor
    extractor = BaselineExtractor(config)

    # Test text
    test_text = """
    Neural networks are computational models inspired by biological neural networks.
    They consist of interconnected nodes organized in layers. Deep learning refers
    to neural networks with multiple hidden layers. Backpropagation is an algorithm
    used for training neural networks. The perceptron is the simplest form of a
    neural network. ReLU is an activation function used in neural networks.
    Neural networks are used for image classification and natural language processing.
    Overfitting is a challenge in deep learning that can be addressed using regularization.
    """

    print(f"\n{'='*60}")
    print("Testing Baseline Extractor")
    print(f"{'='*60}\n")

    print("Input text:")
    print(test_text)

    print(f"\n{'='*60}")
    print("Extracted Triples")
    print(f"{'='*60}\n")

    # Extract triples
    triples = extractor.extract_triples(test_text, source='test')

    if not triples:
        print("No triples extracted!")
    else:
        for idx, triple in enumerate(triples, 1):
            print(f"{idx}. {triple}")
            print(f"   Confidence: {triple.confidence:.2f}")
            print(f"   Method: {triple.provenance.get('method', 'unknown')}")
            print()

    print(f"Total triples extracted: {len(triples)}")

    print(f"\n{'='*60}")
    print("Test Complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
