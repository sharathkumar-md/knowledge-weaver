"""Verifier/Evaluator Agent - automated quality metrics and validation"""

from typing import Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import numpy as np
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Extraction metrics
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Graph quality metrics
    graph_density: float = 0.0
    avg_node_degree: float = 0.0
    coverage: float = 0.0
    connectivity: float = 0.0

    # Confidence metrics
    avg_confidence: float = 0.0
    high_confidence_ratio: float = 0.0

    # Additional metrics
    total_triples: int = 0
    unique_entities: int = 0
    total_relations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return f"""Evaluation Metrics:
  Extraction Quality:
    - Precision: {self.precision:.3f}
    - Recall: {self.recall:.3f}
    - F1 Score: {self.f1_score:.3f}

  Graph Quality:
    - Density: {self.graph_density:.3f}
    - Avg Degree: {self.avg_node_degree:.2f}
    - Coverage: {self.coverage:.3f}
    - Connectivity: {self.connectivity:.3f}

  Confidence:
    - Avg Confidence: {self.avg_confidence:.3f}
    - High Confidence Ratio: {self.high_confidence_ratio:.3f}

  Statistics:
    - Total Triples: {self.total_triples}
    - Unique Entities: {self.unique_entities}
    - Total Relations: {self.total_relations}
"""


class Evaluator:
    """Evaluator agent for quality assessment"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config.get('evaluation', {})

        self.metrics_to_compute = self.eval_config.get('metrics', [
            'precision', 'recall', 'f1', 'graph_density', 'coverage'
        ])

        logger.info("Evaluator initialized")

    def evaluate_extraction(
        self,
        predicted_triples: List[Dict[str, Any]],
        gold_triples: List[Dict[str, Any]]
    ) -> EvaluationMetrics:
        """Evaluate extraction quality against gold standard"""
        logger.info("Evaluating extraction quality...")

        # Convert to comparable format
        pred_set = self._triples_to_set(predicted_triples)
        gold_set = self._triples_to_set(gold_triples)

        # Calculate precision, recall, F1
        true_positives = len(pred_set & gold_set)
        false_positives = len(pred_set - gold_set)
        false_negatives = len(gold_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics = EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_triples=len(predicted_triples)
        )

        logger.info(f"Extraction metrics: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        return metrics

    def evaluate_graph(self, graph_store) -> EvaluationMetrics:
        """Evaluate overall graph quality"""
        logger.info("Evaluating graph quality...")

        # Get graph
        if hasattr(graph_store, 'store'):
            graph = graph_store.store.graph
            schema = graph_store.schema
        else:
            graph = graph_store
            schema = None

        metrics = EvaluationMetrics()

        # Basic statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        metrics.unique_entities = num_nodes
        metrics.total_triples = num_edges

        if num_nodes == 0:
            logger.warning("Empty graph")
            return metrics

        # Graph density
        if num_nodes > 1:
            max_edges = num_nodes * (num_nodes - 1)  # for directed graph
            metrics.graph_density = num_edges / max_edges if max_edges > 0 else 0

        # Average node degree
        degrees = [d for _, d in graph.degree()]
        metrics.avg_node_degree = np.mean(degrees) if degrees else 0

        # Connectivity (ratio of nodes in largest component)
        import networkx as nx
        if num_nodes > 0:
            largest_cc = max(nx.weakly_connected_components(graph), key=len)
            metrics.connectivity = len(largest_cc) / num_nodes

        # Coverage (ratio of nodes with at least one connection)
        connected_nodes = sum(1 for d in degrees if d > 0)
        metrics.coverage = connected_nodes / num_nodes if num_nodes > 0 else 0

        # Confidence metrics (if available from schema)
        if schema and hasattr(schema, 'edges'):
            confidences = [e.confidence for e in schema.edges if hasattr(e, 'confidence')]
            if confidences:
                metrics.avg_confidence = np.mean(confidences)
                metrics.high_confidence_ratio = sum(1 for c in confidences if c >= 0.8) / len(confidences)

        # Count unique relations
        if hasattr(graph, 'edges') and callable(graph.edges):
            relations = set()
            for _, _, data in graph.edges(data=True):
                if 'relation' in data:
                    relations.add(data['relation'])
            metrics.total_relations = len(relations)

        logger.info(f"Graph metrics: Density={metrics.graph_density:.3f}, Coverage={metrics.coverage:.3f}")

        return metrics

    def run_full_evaluation(
        self,
        graph_store,
        gold_standard_path: str = None
    ) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        logger.info("Running full evaluation...")

        results = {
            'graph_metrics': None,
            'extraction_metrics': None,
            'timestamp': None
        }

        # Evaluate graph quality
        graph_metrics = self.evaluate_graph(graph_store)
        results['graph_metrics'] = graph_metrics.to_dict()

        # Evaluate extraction if gold standard provided
        if gold_standard_path and Path(gold_standard_path).exists():
            logger.info(f"Loading gold standard from {gold_standard_path}")

            with open(gold_standard_path, 'r', encoding='utf-8') as f:
                gold_data = json.load(f)

            # Get predicted triples from graph
            predicted_triples = []
            if hasattr(graph_store, 'schema'):
                for edge in graph_store.schema.edges:
                    predicted_triples.append({
                        'subject': edge.source_id,
                        'relation': edge.relation,
                        'object': edge.target_id
                    })

            extraction_metrics = self.evaluate_extraction(predicted_triples, gold_data)
            results['extraction_metrics'] = extraction_metrics.to_dict()

        # Add timestamp
        from datetime import datetime
        results['timestamp'] = datetime.now().isoformat()

        return results

    def _triples_to_set(self, triples: List[Dict[str, Any]]) -> Set[Tuple]:
        """Convert list of triples to set of tuples for comparison"""
        triple_set = set()

        for triple in triples:
            # Normalize
            subject = triple.get('subject', '').lower().strip()
            relation = triple.get('relation', '').upper().strip()
            obj = triple.get('object', '').lower().strip()

            if subject and relation and obj:
                triple_set.add((subject, relation, obj))

        return triple_set

    def compare_extractors(
        self,
        extractor1_triples: List[Dict[str, Any]],
        extractor2_triples: List[Dict[str, Any]],
        extractor1_name: str = "Extractor 1",
        extractor2_name: str = "Extractor 2"
    ) -> Dict[str, Any]:
        """Compare two extractors"""
        logger.info(f"Comparing {extractor1_name} vs {extractor2_name}")

        set1 = self._triples_to_set(extractor1_triples)
        set2 = self._triples_to_set(extractor2_triples)

        # Agreement metrics
        agreement = len(set1 & set2)
        only_in_1 = len(set1 - set2)
        only_in_2 = len(set2 - set1)

        total_unique = len(set1 | set2)
        agreement_ratio = agreement / total_unique if total_unique > 0 else 0

        return {
            'extractor1': {
                'name': extractor1_name,
                'total_triples': len(set1),
                'unique_triples': only_in_1
            },
            'extractor2': {
                'name': extractor2_name,
                'total_triples': len(set2),
                'unique_triples': only_in_2
            },
            'agreement': agreement,
            'agreement_ratio': agreement_ratio
        }

    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        output_path: str = None
    ) -> str:
        """Generate human-readable evaluation report"""
        report_lines = []

        report_lines.append("=" * 60)
        report_lines.append("KNOWLEDGE WEAVER - EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")

        # Timestamp
        if 'timestamp' in evaluation_results:
            report_lines.append(f"Generated: {evaluation_results['timestamp']}")
            report_lines.append("")

        # Graph metrics
        if 'graph_metrics' in evaluation_results and evaluation_results['graph_metrics']:
            report_lines.append("GRAPH QUALITY METRICS")
            report_lines.append("-" * 60)

            gm = evaluation_results['graph_metrics']
            report_lines.append(f"  Total Entities: {gm.get('unique_entities', 0)}")
            report_lines.append(f"  Total Relations: {gm.get('total_relations', 0)}")
            report_lines.append(f"  Total Triples: {gm.get('total_triples', 0)}")
            report_lines.append(f"  Graph Density: {gm.get('graph_density', 0):.3f}")
            report_lines.append(f"  Avg Node Degree: {gm.get('avg_node_degree', 0):.2f}")
            report_lines.append(f"  Coverage: {gm.get('coverage', 0):.3f}")
            report_lines.append(f"  Connectivity: {gm.get('connectivity', 0):.3f}")
            report_lines.append("")

        # Extraction metrics
        if 'extraction_metrics' in evaluation_results and evaluation_results['extraction_metrics']:
            report_lines.append("EXTRACTION QUALITY METRICS")
            report_lines.append("-" * 60)

            em = evaluation_results['extraction_metrics']
            report_lines.append(f"  Precision: {em.get('precision', 0):.3f}")
            report_lines.append(f"  Recall: {em.get('recall', 0):.3f}")
            report_lines.append(f"  F1 Score: {em.get('f1_score', 0):.3f}")
            report_lines.append(f"  Avg Confidence: {em.get('avg_confidence', 0):.3f}")
            report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 60)

        if 'graph_metrics' in evaluation_results:
            gm = evaluation_results['graph_metrics']

            if gm.get('graph_density', 0) < 0.01:
                report_lines.append("  - Graph is sparse. Consider adding more relations.")

            if gm.get('connectivity', 0) < 0.5:
                report_lines.append("  - Graph has disconnected components. Add bridging concepts.")

            if gm.get('coverage', 0) < 0.7:
                report_lines.append("  - Many isolated nodes. Connect concepts to improve coverage.")

        report_lines.append("")
        report_lines.append("=" * 60)

        report_text = '\n'.join(report_lines)

        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text

    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save evaluation results to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {filepath}")

    def run(self, gold_standard_path: str = None) -> None:
        """Run evaluation from command line"""
        logger.info("Running evaluation...")

        # This would be called from main.py with a loaded graph
        # For now, just log
        logger.info("Evaluation complete. Use evaluate_graph() method with a graph store object.")
