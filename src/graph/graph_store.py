"""Graph storage layer with NetworkX and Neo4j support"""

from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import networkx as nx
import json
from pathlib import Path
from loguru import logger

from .schema_manager import SchemaManager, NodeSchema, EdgeSchema


class GraphStore(ABC):
    """Abstract base class for graph storage"""

    @abstractmethod
    def add_node(self, node: NodeSchema) -> None:
        pass

    @abstractmethod
    def add_edge(self, edge: EdgeSchema) -> None:
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[NodeSchema]:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: str) -> List[str]:
        pass

    @abstractmethod
    def export(self, filepath: str) -> None:
        pass


class NetworkXStore(GraphStore):
    """Graph storage using NetworkX"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges
        logger.info("NetworkX graph store initialized")

    def add_node(self, node: NodeSchema) -> None:
        """Add node to graph"""
        self.graph.add_node(
            node.id,
            label=node.label,
            type=node.type,
            aliases=list(node.aliases),
            properties=node.properties,
            provenance=[p.to_dict() for p in node.provenance],
            created_at=node.created_at,
            updated_at=node.updated_at
        )

    def add_edge(self, edge: EdgeSchema) -> None:
        """Add edge to graph"""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.relation,
            relation=edge.relation,
            type=edge.type,
            confidence=edge.confidence,
            properties=edge.properties,
            provenance=[p.to_dict() for p in edge.provenance],
            created_at=edge.created_at,
            updated_at=edge.updated_at
        )

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data"""
        if node_id in self.graph:
            return dict(self.graph.nodes[node_id])
        return None

    def get_neighbors(self, node_id: str, direction: str = 'out') -> List[str]:
        """Get neighbors of a node"""
        if node_id not in self.graph:
            return []

        if direction == 'out':
            return list(self.graph.successors(node_id))
        elif direction == 'in':
            return list(self.graph.predecessors(node_id))
        else:  # 'both'
            return list(set(self.graph.successors(node_id)) | set(self.graph.predecessors(node_id)))

    def get_edges(self, source_id: str = None, target_id: str = None) -> List[Dict[str, Any]]:
        """Get edges matching criteria"""
        edges = []

        if source_id and target_id:
            # Get edges between specific nodes
            for key, edge_data in self.graph[source_id][target_id].items():
                edges.append({
                    'source': source_id,
                    'target': target_id,
                    'key': key,
                    **edge_data
                })
        elif source_id:
            # Get all outgoing edges from source
            for target in self.graph.successors(source_id):
                for key, edge_data in self.graph[source_id][target].items():
                    edges.append({
                        'source': source_id,
                        'target': target,
                        'key': key,
                        **edge_data
                    })
        else:
            # Get all edges
            for source, target, key, edge_data in self.graph.edges(keys=True, data=True):
                edges.append({
                    'source': source,
                    'target': target,
                    'key': key,
                    **edge_data
                })

        return edges

    def find_paths(self, source_id: str, target_id: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between two nodes"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph,
                source_id,
                target_id,
                cutoff=max_length
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []

    def get_subgraph(self, node_ids: List[str], radius: int = 1) -> 'NetworkXStore':
        """Get subgraph around given nodes"""
        # Start with given nodes
        nodes_to_include = set(node_ids)

        # Add neighbors within radius
        for _ in range(radius):
            new_nodes = set()
            for node in nodes_to_include:
                if node in self.graph:
                    new_nodes.update(self.graph.neighbors(node))
            nodes_to_include.update(new_nodes)

        # Create subgraph
        subgraph = self.graph.subgraph(nodes_to_include).copy()

        # Create new store with subgraph
        store = NetworkXStore(self.config)
        store.graph = subgraph

        return store

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
        }

        if self.graph.number_of_nodes() > 0:
            stats['is_connected'] = nx.is_weakly_connected(self.graph)
            # Average degree
            degrees = [d for _, d in self.graph.degree()]
            stats['avg_degree'] = sum(degrees) / len(degrees)
            stats['max_degree'] = max(degrees)

            # Connected components
            stats['num_components'] = nx.number_weakly_connected_components(self.graph)

            # Try to compute diameter (may be expensive for large graphs)
            if self.graph.number_of_nodes() < 1000:
                try:
                    if nx.is_weakly_connected(self.graph):
                        stats['diameter'] = nx.diameter(self.graph.to_undirected())
                except:
                    pass

        return stats

    def export(self, filepath: str, format: str = 'json') -> None:
        """Export graph to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            # Export as node-link JSON
            data = nx.node_link_data(self.graph)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == 'gexf':
            # Export as GEXF (for Gephi)
            nx.write_gexf(self.graph, filepath)

        elif format == 'graphml':
            # Export as GraphML
            nx.write_graphml(self.graph, filepath)

        logger.info(f"Graph exported to {filepath} ({format} format)")

    def load(self, filepath: str, format: str = 'json') -> None:
        """Load graph from file"""
        if format == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.graph = nx.node_link_graph(data, directed=True, multigraph=True)

        elif format == 'gexf':
            self.graph = nx.read_gexf(filepath)

        elif format == 'graphml':
            self.graph = nx.read_graphml(filepath)

        logger.info(f"Graph loaded from {filepath}")

    def visualize(self, output_path: str = None, layout: str = 'spring') -> None:
        """Create visualization of graph"""
        try:
            import matplotlib.pyplot as plt

            # Choose layout
            if layout == 'spring':
                pos = nx.spring_layout(self.graph)
            elif layout == 'circular':
                pos = nx.circular_layout(self.graph)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph)

            # Draw
            plt.figure(figsize=(15, 10))

            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                node_color='lightblue',
                node_size=500,
                alpha=0.8
            )

            # Draw edges
            nx.draw_networkx_edges(
                self.graph,
                pos,
                alpha=0.3,
                arrows=True,
                arrowsize=10
            )

            # Draw labels
            labels = {node: self.graph.nodes[node].get('label', node) for node in self.graph.nodes()}
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels,
                font_size=8
            )

            plt.axis('off')
            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib not available for visualization")


class KnowledgeGraphStore:
    """High-level knowledge graph store with schema management"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        backend = config.get('graph', {}).get('backend', 'networkx')

        # Initialize schema manager
        self.schema = SchemaManager(config)

        # Initialize graph store
        if backend == 'networkx':
            self.store = NetworkXStore(config)
        elif backend == 'neo4j':
            # TODO: Implement Neo4j store
            logger.warning("Neo4j not implemented yet, using NetworkX")
            self.store = NetworkXStore(config)
        else:
            raise ValueError(f"Unknown graph backend: {backend}")

        logger.info(f"Knowledge graph initialized with {backend} backend")

    def add_triples(self, triples: List[Dict[str, Any]]) -> None:
        """Add triples to graph"""
        # Add to schema
        self.schema.add_triples(triples)

        # Add to store
        for node in self.schema.nodes.values():
            self.store.add_node(node)

        for edge in self.schema.edges:
            self.store.add_edge(edge)

        logger.info(f"Added {len(triples)} triples to knowledge graph")

    def query_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Query node by ID"""
        return self.store.get_node(node_id)

    def query_neighbors(self, node_id: str, direction: str = 'out') -> List[str]:
        """Get neighbors of a node"""
        return self.store.get_neighbors(node_id, direction)

    def find_paths(self, source_label: str, target_label: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between two nodes"""
        source_node = self.schema.get_node_by_label(source_label)
        target_node = self.schema.get_node_by_label(target_label)

        if not source_node or not target_node:
            return []

        return self.store.find_paths(source_node.id, target_node.id, max_length)

    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics"""
        return {
            'schema': self.schema.get_statistics(),
            'graph': self.store.get_statistics()
        }

    def export(self, directory: str) -> None:
        """Export both schema and graph"""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Export schema
        self.schema.save(str(dir_path / 'schema.json'))

        # Export graph
        self.store.export(str(dir_path / 'graph.json'), format='json')
        # self.store.export(str(dir_path / 'graph.gexf'), format='gexf')  # Skip GEXF for now

        logger.info(f"Knowledge graph exported to {directory}")

    def load(self, directory: str) -> None:
        """Load both schema and graph"""
        dir_path = Path(directory)

        # Load schema
        self.schema.load(str(dir_path / 'schema.json'))

        # Load graph
        self.store.load(str(dir_path / 'graph.json'), format='json')

        logger.info(f"Knowledge graph loaded from {directory}")
