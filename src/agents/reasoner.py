"""Reasoner/Inferencer Agent - finds gaps, contradictions, and knowledge threads"""

from typing import Dict, Any, List, Tuple, Set
from dataclasses import dataclass, field
import networkx as nx
from loguru import logger
from collections import defaultdict


@dataclass
class Gap:
    """Represents a knowledge gap"""
    type: str  # 'missing_link', 'incomplete_chain', 'unexplored_concept'
    description: str
    related_nodes: List[str] = field(default_factory=list)
    severity: float = 1.0
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'description': self.description,
            'related_nodes': self.related_nodes,
            'severity': self.severity,
            'suggestions': self.suggestions
        }


@dataclass
class Contradiction:
    """Represents a potential contradiction"""
    node1: str
    node2: str
    description: str
    confidence: float = 0.8
    evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'node1': self.node1,
            'node2': self.node2,
            'description': self.description,
            'confidence': self.confidence,
            'evidence': self.evidence
        }


@dataclass
class KnowledgeThread:
    """Represents a connected chain of concepts"""
    name: str
    nodes: List[str]
    edges: List[Tuple[str, str, str]]  # (source, relation, target)
    importance: float = 1.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'nodes': self.nodes,
            'edges': [{'source': s, 'relation': r, 'target': t} for s, r, t in self.edges],
            'importance': self.importance,
            'summary': self.summary
        }


class ReasonerAgent:
    """Agent for reasoning over the knowledge graph"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = config.get('agents', {}).get('reasoner', {})

        self.max_inference_depth = self.agent_config.get('max_inference_depth', 3)
        self.contradiction_threshold = self.agent_config.get('contradiction_threshold', 0.8)
        self.gap_detection_enabled = self.agent_config.get('gap_detection_enabled', True)

        self.gaps: List[Gap] = []
        self.contradictions: List[Contradiction] = []
        self.threads: List[KnowledgeThread] = []

        logger.info("Reasoner Agent initialized")

    def analyze_graph(self, graph_store) -> Dict[str, Any]:
        """Perform comprehensive analysis on the knowledge graph"""
        logger.info("Starting graph analysis...")

        # Get the NetworkX graph
        if hasattr(graph_store, 'store'):
            nx_graph = graph_store.store.graph
        else:
            nx_graph = graph_store.graph

        # Find gaps
        if self.gap_detection_enabled:
            self.gaps = self._find_gaps(nx_graph)
            logger.info(f"Found {len(self.gaps)} knowledge gaps")

        # Find contradictions
        self.contradictions = self._find_contradictions(nx_graph)
        logger.info(f"Found {len(self.contradictions)} potential contradictions")

        # Identify knowledge threads
        self.threads = self._identify_threads(nx_graph)
        logger.info(f"Identified {len(self.threads)} knowledge threads")

        # Infer new connections
        inferred_edges = self._infer_connections(nx_graph)
        logger.info(f"Inferred {len(inferred_edges)} new connections")

        return {
            'gaps': [g.to_dict() for g in self.gaps],
            'contradictions': [c.to_dict() for c in self.contradictions],
            'threads': [t.to_dict() for t in self.threads],
            'inferred_edges': inferred_edges
        }

    def _find_gaps(self, graph: nx.MultiDiGraph) -> List[Gap]:
        """Find knowledge gaps in the graph"""
        gaps = []

        # 1. Find isolated or poorly connected nodes
        for node in graph.nodes():
            degree = graph.degree(node)
            if degree == 0:
                gap = Gap(
                    type='isolated_concept',
                    description=f"'{node}' is not connected to any other concepts",
                    related_nodes=[node],
                    severity=0.8,
                    suggestions=[
                        f"Add relationships between '{node}' and related concepts",
                        f"Provide more context about '{node}'"
                    ]
                )
                gaps.append(gap)
            elif degree == 1:
                gap = Gap(
                    type='weakly_connected',
                    description=f"'{node}' has only one connection",
                    related_nodes=[node],
                    severity=0.5,
                    suggestions=[f"Explore more relationships for '{node}'"]
                )
                gaps.append(gap)

        # 2. Find incomplete chains (paths that could be longer)
        # Look for nodes that could bridge disconnected components
        components = list(nx.weakly_connected_components(graph))
        if len(components) > 1:
            for i, comp1 in enumerate(components):
                for comp2 in components[i+1:]:
                    # Sample nodes from each component
                    node1 = list(comp1)[0] if comp1 else None
                    node2 = list(comp2)[0] if comp2 else None

                    if node1 and node2:
                        gap = Gap(
                            type='disconnected_components',
                            description=f"No path between '{node1}' and '{node2}'",
                            related_nodes=[node1, node2],
                            severity=0.9,
                            suggestions=[
                                f"Find relationships connecting concepts in separate clusters",
                                "Add bridging concepts or relations"
                            ]
                        )
                        gaps.append(gap)

        # 3. Find nodes with missing expected relations
        # (e.g., if A IS_A B and B IS_A C, we might expect A IS_A C)
        for node in list(graph.nodes())[:50]:  # Limit to avoid performance issues
            outgoing = list(graph.successors(node))

            if len(outgoing) >= 2:
                # Check if there should be connections between successors
                for i, succ1 in enumerate(outgoing):
                    for succ2 in outgoing[i+1:]:
                        if not graph.has_edge(succ1, succ2) and not graph.has_edge(succ2, succ1):
                            gap = Gap(
                                type='missing_link',
                                description=f"'{succ1}' and '{succ2}' are both related to '{node}' but not to each other",
                                related_nodes=[node, succ1, succ2],
                                severity=0.4,
                                suggestions=[f"Consider if '{succ1}' and '{succ2}' should be related"]
                            )
                            gaps.append(gap)

        return gaps[:20]  # Return top 20 gaps

    def _find_contradictions(self, graph: nx.MultiDiGraph) -> List[Contradiction]:
        """Find potential contradictions in the graph"""
        contradictions = []

        # Look for contradictory relations
        contradiction_pairs = [
            ('SIMILAR_TO', 'DIFFERS_FROM'),
            ('IS_A', 'IS_NOT'),
            ('CAUSES', 'PREVENTS'),
        ]

        for node1 in list(graph.nodes())[:50]:  # Limit for performance
            for node2 in graph.successors(node1):
                # Get all relations between node1 and node2
                relations = set()
                if graph.has_edge(node1, node2):
                    for key in graph[node1][node2]:
                        relations.add(graph[node1][node2][key].get('relation', ''))

                # Check for contradictory relations
                for rel1, rel2 in contradiction_pairs:
                    if rel1 in relations and rel2 in relations:
                        contradiction = Contradiction(
                            node1=node1,
                            node2=node2,
                            description=f"Both '{rel1}' and '{rel2}' relations exist between '{node1}' and '{node2}'",
                            confidence=0.9,
                            evidence=[f"{rel1} relation", f"{rel2} relation"]
                        )
                        contradictions.append(contradiction)

        return contradictions[:10]  # Return top 10

    def _identify_threads(self, graph: nx.MultiDiGraph) -> List[KnowledgeThread]:
        """Identify important knowledge threads (chains of concepts)"""
        threads = []

        # Find strongly connected subgraphs
        if len(graph.nodes()) > 0:
            # Find central nodes (high betweenness centrality)
            try:
                centrality = nx.betweenness_centrality(graph)
                central_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]

                # Build threads around central nodes
                for node, centrality_score in central_nodes:
                    thread = self._build_thread_from_node(graph, node)
                    if thread and len(thread.nodes) >= 3:
                        thread.importance = centrality_score
                        threads.append(thread)

            except Exception as e:
                logger.warning(f"Could not compute centrality: {e}")

        # Find longest paths
        if len(threads) < 5:
            # Add some path-based threads
            for source in list(graph.nodes())[:10]:
                for target in list(graph.nodes())[:10]:
                    if source != target:
                        try:
                            path = nx.shortest_path(graph, source, target)
                            if len(path) >= 3:
                                thread = self._path_to_thread(graph, path)
                                threads.append(thread)
                        except nx.NetworkXNoPath:
                            pass

        # Deduplicate and sort by importance
        threads = self._deduplicate_threads(threads)
        threads.sort(key=lambda t: t.importance, reverse=True)

        return threads[:10]  # Return top 10 threads

    def _build_thread_from_node(self, graph: nx.MultiDiGraph, central_node: str) -> KnowledgeThread:
        """Build a knowledge thread centered on a node"""
        # Get neighbors within depth
        nodes = {central_node}
        edges = []

        # BFS to depth 2
        current_level = {central_node}
        for _ in range(2):
            next_level = set()
            for node in current_level:
                for neighbor in graph.successors(node):
                    next_level.add(neighbor)
                    nodes.add(neighbor)

                    # Get edge info
                    if graph.has_edge(node, neighbor):
                        for key in graph[node][neighbor]:
                            relation = graph[node][neighbor][key].get('relation', 'RELATED_TO')
                            edges.append((node, relation, neighbor))

            current_level = next_level

        thread = KnowledgeThread(
            name=f"Thread: {central_node}",
            nodes=list(nodes),
            edges=edges[:20],  # Limit edges
            importance=1.0,
            summary=f"Knowledge thread centered on '{central_node}' with {len(nodes)} related concepts"
        )

        return thread

    def _path_to_thread(self, graph: nx.MultiDiGraph, path: List[str]) -> KnowledgeThread:
        """Convert a path to a knowledge thread"""
        edges = []

        for i in range(len(path) - 1):
            source = path[i]
            target = path[i+1]

            if graph.has_edge(source, target):
                # Get first relation
                key = list(graph[source][target].keys())[0]
                relation = graph[source][target][key].get('relation', 'RELATED_TO')
                edges.append((source, relation, target))

        thread = KnowledgeThread(
            name=f"Path: {path[0]} → {path[-1]}",
            nodes=path,
            edges=edges,
            importance=0.5,
            summary=f"Path from '{path[0]}' to '{path[-1]}' through {len(path)-2} intermediate concepts"
        )

        return thread

    def _deduplicate_threads(self, threads: List[KnowledgeThread]) -> List[KnowledgeThread]:
        """Remove duplicate or very similar threads"""
        unique_threads = []
        seen_node_sets = []

        for thread in threads:
            node_set = set(thread.nodes)

            # Check if very similar to existing thread
            is_duplicate = False
            for existing_set in seen_node_sets:
                overlap = len(node_set & existing_set) / max(len(node_set), len(existing_set))
                if overlap > 0.8:  # 80% overlap
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_threads.append(thread)
                seen_node_sets.append(node_set)

        return unique_threads

    def _infer_connections(self, graph: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """Infer new connections based on graph patterns"""
        inferred = []

        # Transitive inference: if A→B and B→C, infer A→C
        for node_a in list(graph.nodes())[:30]:  # Limit for performance
            for node_b in graph.successors(node_a):
                for node_c in graph.successors(node_b):
                    if node_c != node_a and not graph.has_edge(node_a, node_c):
                        # Get relations
                        rel_ab = self._get_primary_relation(graph, node_a, node_b)
                        rel_bc = self._get_primary_relation(graph, node_b, node_c)

                        # Infer relation A→C
                        inferred_relation = self._infer_transitive_relation(rel_ab, rel_bc)

                        if inferred_relation:
                            inferred.append({
                                'source': node_a,
                                'target': node_c,
                                'relation': inferred_relation,
                                'confidence': 0.6,
                                'reasoning': f"Transitive: {node_a} --{rel_ab}--> {node_b} --{rel_bc}--> {node_c}",
                                'method': 'transitive_inference'
                            })

        return inferred[:20]  # Return top 20

    def _get_primary_relation(self, graph: nx.MultiDiGraph, source: str, target: str) -> str:
        """Get the primary relation between two nodes"""
        if not graph.has_edge(source, target):
            return None

        # Get first relation
        key = list(graph[source][target].keys())[0]
        return graph[source][target][key].get('relation', 'RELATED_TO')

    def _infer_transitive_relation(self, rel1: str, rel2: str) -> str:
        """Infer transitive relation from two relations"""
        # Define transitive rules
        transitive_rules = {
            ('IS_A', 'IS_A'): 'IS_A',
            ('HAS_PART', 'HAS_PART'): 'HAS_PART',
            ('CAUSES', 'CAUSES'): 'CAUSES',
            ('LEADS_TO', 'LEADS_TO'): 'LEADS_TO',
            ('BEFORE', 'BEFORE'): 'BEFORE',
        }

        return transitive_rules.get((rel1, rel2), 'RELATED_TO')

    def get_gap_summary(self) -> str:
        """Get human-readable summary of gaps"""
        if not self.gaps:
            return "No significant knowledge gaps detected."

        summary = f"Found {len(self.gaps)} knowledge gaps:\n"

        # Group by type
        by_type = defaultdict(list)
        for gap in self.gaps:
            by_type[gap.type].append(gap)

        for gap_type, gaps in by_type.items():
            summary += f"\n{gap_type.replace('_', ' ').title()}: {len(gaps)}\n"
            for gap in gaps[:3]:  # Show top 3
                summary += f"  - {gap.description}\n"

        return summary

    def get_thread_summary(self) -> str:
        """Get human-readable summary of threads"""
        if not self.threads:
            return "No knowledge threads identified."

        summary = f"Identified {len(self.threads)} knowledge threads:\n\n"

        for i, thread in enumerate(self.threads[:5], 1):
            summary += f"{i}. {thread.name}\n"
            summary += f"   {thread.summary}\n"
            summary += f"   Importance: {thread.importance:.2f}\n\n"

        return summary
