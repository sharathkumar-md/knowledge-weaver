"""Dataset builder for LoRA fine-tuning"""

from typing import List, Dict, Any, Tuple
import json
import random
from pathlib import Path
from dataclasses import dataclass
import requests
from tqdm import tqdm
from loguru import logger


@dataclass
class TrainingExample:
    """A single training example for triple extraction"""
    input_text: str
    output_triples: List[str]  # List of "S:entity|R:relation|O:entity" strings
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'input': self.input_text,
            'output': '\n'.join(self.output_triples),
            'metadata': self.metadata
        }

    def to_instruction_format(self) -> Dict[str, Any]:
        """Convert to instruction-following format"""
        instruction = """Extract knowledge graph triples from the following text.
Output format: S:subject|R:relation|O:object (one per line)"""

        return {
            'instruction': instruction,
            'input': self.input_text,
            'output': '\n'.join(self.output_triples)
        }


class DatasetBuilder:
    """Build training dataset from multiple sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.examples: List[TrainingExample] = []

    def add_example(self, text: str, triples: List[Tuple[str, str, str]], metadata: Dict = None):
        """Add a training example"""
        # Format triples
        triple_strings = [f"S:{s}|R:{r}|O:{o}" for s, r, o in triples]

        example = TrainingExample(
            input_text=text,
            output_triples=triple_strings,
            metadata=metadata or {}
        )

        self.examples.append(example)

    def build_from_wikipedia(self, topics: List[str], max_per_topic: int = 10) -> None:
        """Build dataset from Wikipedia articles"""
        logger.info(f"Building dataset from {len(topics)} Wikipedia topics")

        for topic in tqdm(topics, desc="Fetching Wikipedia"):
            try:
                examples = self._fetch_wikipedia_examples(topic, max_per_topic)
                self.examples.extend(examples)
            except Exception as e:
                logger.warning(f"Failed to fetch {topic}: {e}")

        logger.info(f"Added {len(self.examples)} examples from Wikipedia")

    def _fetch_wikipedia_examples(self, topic: str, max_examples: int) -> List[TrainingExample]:
        """Fetch examples from a Wikipedia article"""
        # Get Wikipedia article
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'titles': topic,
            'prop': 'extracts',
            'explaintext': True,
            'exsectionformat': 'plain'
        }

        response = requests.get(url, params=params)
        data = response.json()

        # Extract text
        pages = data['query']['pages']
        page = next(iter(pages.values()))

        if 'extract' not in page:
            logger.warning(f"No content found for {topic}")
            return []

        text = page['extract']

        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]

        # Extract triples using baseline extractor
        from src.agents.extractor import BaselineExtractor
        extractor = BaselineExtractor(self.config)

        examples = []

        for para in paragraphs[:max_examples]:
            triples = extractor.extract_triples(para, source=f'wikipedia:{topic}')

            if len(triples) >= 2:  # Only keep if we extracted multiple triples
                triple_tuples = [(t.subject, t.relation, t.object) for t in triples]

                example = TrainingExample(
                    input_text=para,
                    output_triples=[f"S:{s}|R:{r}|O:{o}" for s, r, o in triple_tuples],
                    metadata={'source': 'wikipedia', 'topic': topic}
                )
                examples.append(example)

        return examples

    def build_synthetic_examples(self, num_examples: int = 100) -> None:
        """Generate synthetic training examples"""
        logger.info(f"Generating {num_examples} synthetic examples")

        templates = self._get_synthetic_templates()

        for _ in range(num_examples):
            template = random.choice(templates)
            example = self._generate_from_template(template)
            if example:
                self.examples.append(example)

        logger.info(f"Generated {num_examples} synthetic examples")

    def _get_synthetic_templates(self) -> List[Dict[str, Any]]:
        """Get templates for synthetic data generation"""
        return [
            {
                'text': '{concept1} is a type of {concept2}. It is used for {purpose}.',
                'triples': [
                    ('concept1', 'IS_A', 'concept2'),
                    ('concept1', 'USED_FOR', 'purpose')
                ]
            },
            {
                'text': '{concept1} consists of {part1} and {part2}. The {part1} {action} the {part2}.',
                'triples': [
                    ('concept1', 'HAS_PART', 'part1'),
                    ('concept1', 'HAS_PART', 'part2'),
                    ('part1', 'INTERACTS_WITH', 'part2')
                ]
            },
            {
                'text': '{concept1} causes {effect}. This can lead to {result}.',
                'triples': [
                    ('concept1', 'CAUSES', 'effect'),
                    ('effect', 'LEADS_TO', 'result')
                ]
            },
            {
                'text': '{tool} is used for {task}. It is similar to {alternative}.',
                'triples': [
                    ('tool', 'USED_FOR', 'task'),
                    ('tool', 'SIMILAR_TO', 'alternative')
                ]
            },
        ]

    def _generate_from_template(self, template: Dict[str, Any]) -> TrainingExample:
        """Generate an example from a template"""
        # Sample vocabulary
        concepts = ['Neural Network', 'Algorithm', 'Data Structure', 'Model', 'Function']
        parts = ['Input Layer', 'Hidden Layer', 'Output Layer', 'Node', 'Connection']
        actions = ['processes', 'transforms', 'feeds into', 'connects to']
        tools = ['Framework', 'Library', 'Tool', 'System']

        # Fill template
        text = template['text']
        replacements = {
            'concept1': random.choice(concepts),
            'concept2': random.choice(concepts),
            'part1': random.choice(parts),
            'part2': random.choice(parts),
            'action': random.choice(actions),
            'tool': random.choice(tools),
            'purpose': 'data processing',
            'effect': 'improved performance',
            'result': 'better accuracy',
            'task': 'machine learning',
            'alternative': 'similar tool'
        }

        for key, value in replacements.items():
            text = text.replace(f'{{{key}}}', value)

        # Generate triples
        triple_strings = []
        for subj_key, rel, obj_key in template['triples']:
            subj = replacements.get(subj_key, subj_key)
            obj = replacements.get(obj_key, obj_key)
            triple_strings.append(f"S:{subj}|R:{rel}|O:{obj}")

        return TrainingExample(
            input_text=text,
            output_triples=triple_strings,
            metadata={'source': 'synthetic'}
        )

    def add_manual_annotations(self, annotation_file: str) -> None:
        """Add manually annotated examples from file"""
        logger.info(f"Loading manual annotations from {annotation_file}")

        path = Path(annotation_file)
        if not path.exists():
            logger.warning(f"Annotation file not found: {annotation_file}")
            return

        with open(path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)

        for ann in annotations:
            example = TrainingExample(
                input_text=ann['text'],
                output_triples=ann['triples'],
                metadata={'source': 'manual', **ann.get('metadata', {})}
            )
            self.examples.append(example)

        logger.info(f"Added {len(annotations)} manual annotations")

    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List, List, List]:
        """Split dataset into train/val/test sets"""
        random.shuffle(self.examples)

        total = len(self.examples)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train = self.examples[:train_end]
        val = self.examples[train_end:val_end]
        test = self.examples[val_end:]

        logger.info(f"Dataset split: {len(train)} train, {len(val)} val, {len(test)} test")

        return train, val, test

    def save_datasets(self, output_dir: str, format: str = 'json'):
        """Save datasets to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train, val, test = self.split_dataset()

        # Save in specified format
        if format == 'json':
            self._save_json(train, output_path / 'train.json')
            self._save_json(val, output_path / 'val.json')
            self._save_json(test, output_path / 'test.json')
        elif format == 'jsonl':
            self._save_jsonl(train, output_path / 'train.jsonl')
            self._save_jsonl(val, output_path / 'val.jsonl')
            self._save_jsonl(test, output_path / 'test.jsonl')
        elif format == 'instruction':
            self._save_instruction(train, output_path / 'train.json')
            self._save_instruction(val, output_path / 'val.json')
            self._save_instruction(test, output_path / 'test.json')

        # Save summary
        summary = {
            'total_examples': len(self.examples),
            'train_size': len(train),
            'val_size': len(val),
            'test_size': len(test),
            'sources': self._count_sources()
        }

        with open(output_path / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved datasets to {output_dir}")

    def _save_json(self, examples: List[TrainingExample], path: Path):
        """Save as JSON array"""
        data = [ex.to_dict() for ex in examples]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_jsonl(self, examples: List[TrainingExample], path: Path):
        """Save as JSONL (one JSON per line)"""
        with open(path, 'w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + '\n')

    def _save_instruction(self, examples: List[TrainingExample], path: Path):
        """Save in instruction-following format"""
        data = [ex.to_instruction_format() for ex in examples]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _count_sources(self) -> Dict[str, int]:
        """Count examples by source"""
        sources = {}
        for ex in self.examples:
            source = ex.metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        return sources
