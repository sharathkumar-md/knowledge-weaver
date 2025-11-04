"""Extractor Agent - extracts concepts and relations from text"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import spacy
from loguru import logger
import re


@dataclass
class Triple:
    """Represents a knowledge graph triple"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    provenance: Dict[str, Any] = None

    def __post_init__(self):
        if self.provenance is None:
            self.provenance = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'subject': self.subject,
            'relation': self.relation,
            'object': self.object,
            'confidence': self.confidence,
            'provenance': self.provenance
        }

    def __str__(self) -> str:
        return f"S:{self.subject}|R:{self.relation}|O:{self.object}"


class BaselineExtractor:
    """Baseline extractor using NER and dependency parsing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = config.get('agents', {}).get('extractor', {})
        self.confidence_threshold = self.agent_config.get('confidence_threshold', 0.5)
        self.max_triples = self.agent_config.get('max_triples_per_chunk', 10)

        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise

        # Define relation patterns
        self.relation_patterns = self._build_relation_patterns()

    def _build_relation_patterns(self) -> List[Dict[str, Any]]:
        """Build patterns for relation extraction"""
        return [
            # is-a relations
            {'pattern': r'(\w+) is a (?:type of |kind of )?(\w+)', 'relation': 'IS_A'},
            {'pattern': r'(\w+) are (?:types of |kinds of )?(\w+)', 'relation': 'IS_A'},

            # has-part relations
            {'pattern': r'(\w+) (?:consists of|contains|includes?) (\w+)', 'relation': 'HAS_PART'},
            {'pattern': r'(\w+) (?:has|have) (\w+)', 'relation': 'HAS_PROPERTY'},

            # definition relations
            {'pattern': r'(\w+)[:,] (\w+(?:\s+\w+){1,5})', 'relation': 'DEFINED_AS'},

            # causation
            {'pattern': r'(\w+) causes? (\w+)', 'relation': 'CAUSES'},
            {'pattern': r'(\w+) leads? to (\w+)', 'relation': 'LEADS_TO'},
            {'pattern': r'(\w+) results? in (\w+)', 'relation': 'RESULTS_IN'},

            # application
            {'pattern': r'(\w+) (?:is )?used (?:for|in) (\w+)', 'relation': 'USED_FOR'},
            {'pattern': r'(\w+) applies? to (\w+)', 'relation': 'APPLIES_TO'},

            # comparison
            {'pattern': r'(\w+) (?:is )?similar to (\w+)', 'relation': 'SIMILAR_TO'},
            {'pattern': r'(\w+) differs? from (\w+)', 'relation': 'DIFFERS_FROM'},

            # temporal
            {'pattern': r'(\w+) (?:comes? )?before (\w+)', 'relation': 'BEFORE'},
            {'pattern': r'(\w+) (?:comes? )?after (\w+)', 'relation': 'AFTER'},
        ]

    def extract_triples(self, text: str, source: str = None) -> List[Triple]:
        """Extract triples from text using NER and pattern matching"""
        doc = self.nlp(text)
        triples = []

        # Method 1: Pattern-based extraction
        pattern_triples = self._extract_by_patterns(text, source)
        triples.extend(pattern_triples)

        # Method 2: Dependency parsing
        dep_triples = self._extract_by_dependencies(doc, source)
        triples.extend(dep_triples)

        # Method 3: Entity co-occurrence
        cooc_triples = self._extract_by_cooccurrence(doc, source)
        triples.extend(cooc_triples)

        # Filter by confidence and deduplicate
        triples = self._filter_and_deduplicate(triples)

        return triples[:self.max_triples]

    def _extract_by_patterns(self, text: str, source: str) -> List[Triple]:
        """Extract triples using regex patterns"""
        triples = []

        for pattern_dict in self.relation_patterns:
            pattern = pattern_dict['pattern']
            relation = pattern_dict['relation']

            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                subject = self._normalize_entity(match.group(1))
                obj = self._normalize_entity(match.group(2))

                if subject and obj and subject != obj:
                    triple = Triple(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        confidence=0.7,
                        provenance={'source': source, 'method': 'pattern', 'pattern': pattern}
                    )
                    triples.append(triple)

        return triples

    def _extract_by_dependencies(self, doc, source: str) -> List[Triple]:
        """Extract triples using dependency parsing"""
        triples = []

        for sent in doc.sents:
            for token in sent:
                # Subject-verb-object patterns
                if token.pos_ == 'VERB':
                    subjects = [child for child in token.children if child.dep_ in ('nsubj', 'nsubjpass')]
                    objects = [child for child in token.children if child.dep_ in ('dobj', 'pobj', 'attr')]

                    for subj in subjects:
                        for obj in objects:
                            subject = self._get_noun_phrase(subj)
                            object_text = self._get_noun_phrase(obj)
                            relation = self._normalize_relation(token.lemma_)

                            if subject and object_text and relation:
                                triple = Triple(
                                    subject=subject,
                                    relation=relation,
                                    object=object_text,
                                    confidence=0.6,
                                    provenance={'source': source, 'method': 'dependency', 'verb': token.text}
                                )
                                triples.append(triple)

        return triples

    def _extract_by_cooccurrence(self, doc, source: str) -> List[Triple]:
        """Extract triples based on entity co-occurrence in sentences"""
        triples = []

        for sent in doc.sents:
            # Extract all entities in sentence
            entities = [(ent.text, ent.label_) for ent in sent.ents]

            # Create RELATED_TO triples for entities in same sentence
            if len(entities) >= 2:
                for i, (ent1, label1) in enumerate(entities):
                    for ent2, label2 in entities[i+1:]:
                        if ent1 != ent2:
                            triple = Triple(
                                subject=self._normalize_entity(ent1),
                                relation='RELATED_TO',
                                object=self._normalize_entity(ent2),
                                confidence=0.4,
                                provenance={
                                    'source': source,
                                    'method': 'cooccurrence',
                                    'entity_types': [label1, label2]
                                }
                            )
                            triples.append(triple)

        return triples

    def _get_noun_phrase(self, token) -> str:
        """Get the full noun phrase for a token"""
        # Get the subtree
        phrase_tokens = list(token.subtree)

        # Filter to keep relevant tokens
        phrase = ' '.join([t.text for t in phrase_tokens if not t.is_punct])

        return self._normalize_entity(phrase)

    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text"""
        # Remove extra whitespace
        text = ' '.join(text.split())

        # Remove articles
        text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE)

        # Capitalize
        text = text.strip().title()

        return text if len(text) > 1 else None

    def _normalize_relation(self, verb: str) -> str:
        """Normalize relation from verb"""
        # Convert to uppercase and replace spaces with underscores
        relation = verb.upper().replace(' ', '_')

        return relation if len(relation) > 1 else None

    def _filter_and_deduplicate(self, triples: List[Triple]) -> List[Triple]:
        """Filter by confidence and remove duplicates"""
        # Filter by confidence
        filtered = [t for t in triples if t.confidence >= self.confidence_threshold]

        # Deduplicate based on (subject, relation, object)
        seen = set()
        deduplicated = []

        for triple in filtered:
            key = (triple.subject.lower(), triple.relation, triple.object.lower())
            if key not in seen:
                seen.add(key)
                deduplicated.append(triple)

        # Sort by confidence descending
        deduplicated.sort(key=lambda t: t.confidence, reverse=True)

        return deduplicated


class ExtractorAgent:
    """Main extractor agent that can use baseline or fine-tuned models"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = config.get('agents', {}).get('extractor', {})
        self.use_lora = self.agent_config.get('use_lora', False)

        # Initialize baseline extractor
        self.baseline_extractor = BaselineExtractor(config)

        # Initialize fine-tuned extractor if enabled
        self.lora_extractor = None
        if self.use_lora:
            try:
                from src.models.lora_extractor import LoRAExtractor
                self.lora_extractor = LoRAExtractor(config)
                logger.info("LoRA extractor initialized")
            except Exception as e:
                logger.warning(f"Failed to load LoRA extractor: {e}")
                logger.info("Falling back to baseline extractor")
                self.use_lora = False

    def extract(self, text: str, source: str = None) -> List[Triple]:
        """Extract triples using configured extractor"""
        if self.use_lora and self.lora_extractor:
            return self.lora_extractor.extract_triples(text, source)
        else:
            return self.baseline_extractor.extract_triples(text, source)

    def run(self, input_dir: str) -> None:
        """Run extraction on processed documents"""
        from pathlib import Path
        import json

        input_path = Path(input_dir)
        output_path = input_path.parent / 'extracted'
        output_path.mkdir(exist_ok=True)

        # Process all JSON files
        all_triples = []

        for json_file in input_path.glob('*.json'):
            if json_file.name == '_summary.json':
                continue

            logger.info(f"Extracting from {json_file.name}")

            with open(json_file, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)

            # Extract from each chunk
            for chunk in doc_data['chunks']:
                triples = self.extract(chunk, source=doc_data['metadata']['source'])
                all_triples.extend([t.to_dict() for t in triples])

        # Save all triples
        output_file = output_path / 'triples.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_triples, f, indent=2, ensure_ascii=False)

        logger.info(f"Extracted {len(all_triples)} triples to {output_file}")
