"""Main entry point for Knowledge Weaver"""

import argparse
from pathlib import Path
from src.utils import load_config, setup_logging
from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Knowledge Weaver - Personal Knowledge Graph Builder")

    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('--input', type=str, required=True, help='Input directory')
    ingest_parser.add_argument('--output', type=str, help='Output directory')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract knowledge graph')
    extract_parser.add_argument('--input', type=str, required=True, help='Input data directory')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train LoRA model')
    train_parser.add_argument('--dataset', type=str, required=True, help='Training dataset path')
    train_parser.add_argument('--output', type=str, required=True, help='Output model directory')

    # UI command
    ui_parser = subparsers.add_parser('ui', help='Launch UI')
    ui_parser.add_argument('--port', type=int, help='Port to run on')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Run evaluation')
    eval_parser.add_argument('--gold-standard', type=str, help='Gold standard data path')

    args = parser.parse_args()

    # Load config and setup logging
    config = load_config(args.config)
    setup_logging(config)

    logger.info("Knowledge Weaver starting...")
    logger.info(f"Command: {args.command}")

    if args.command == 'ingest':
        from src.ingestion.pipeline import IngestionPipeline
        pipeline = IngestionPipeline(config)
        pipeline.run(args.input, args.output)

    elif args.command == 'extract':
        from src.agents.extractor import ExtractorAgent
        extractor = ExtractorAgent(config)
        extractor.run(args.input)

    elif args.command == 'train':
        from src.models.train_lora import train_lora
        train_lora(config, args.dataset, args.output)

    elif args.command == 'ui':
        import subprocess
        port = args.port or config['ui']['port']
        subprocess.run(['streamlit', 'run', 'src/ui/app.py', '--server.port', str(port)])

    elif args.command == 'evaluate':
        from src.evaluation.evaluator import Evaluator
        evaluator = Evaluator(config)
        evaluator.run(args.gold_standard)

    else:
        parser.print_help()

    logger.info("Knowledge Weaver finished.")


if __name__ == '__main__':
    main()
