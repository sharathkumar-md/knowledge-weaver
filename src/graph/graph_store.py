"""Graph storage layer with NetworkX and Neo4j support"""

from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
import networkx as nx
import json
from pathlib import Path
from loguru import logger

from .schema_manager import SchemaManager, NodeSchema, EdgeSchema

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("neo4j driver not installed. Install with: pip install neo4j")


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


class Neo4jStore(GraphStore):
    """Graph storage using Neo4j database"""

    def __init__(self, config: Dict[str, Any]):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j driver not installed. Install with: pip install neo4j")

        self.config = config
        neo4j_config = config.get('graph', {}).get('neo4j', {})

        # Get connection details
        self.uri = neo4j_config.get('uri', 'bolt://localhost:7687')
        self.user = neo4j_config.get('user', 'neo4j')
        self.password = neo4j_config.get('password', 'password')

        # Initialize driver
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j graph store initialized (connected to {self.uri})")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j at {self.uri}: {e}")
            raise

        # Create constraints and indexes
        self._create_constraints()

    def _create_constraints(self):
        """Create uniqueness constraints and indexes"""
        with self.driver.session() as session:
            # Constraint for unique node IDs
            try:
                session.run("""
                    CREATE CONSTRAINT node_id_unique IF NOT EXISTS
                    FOR (n:Concept) REQUIRE n.id IS UNIQUE
                """)
                logger.debug("Created uniqueness constraint on Concept.id")
            except Exception as e:
                logger.warning(f"Could not create constraint: {e}")

            # Index for faster lookups
            try:
                session.run("""
                    CREATE INDEX node_label_index IF NOT EXISTS
                    FOR (n:Concept) ON (n.label)
                """)
                logger.debug("Created index on Concept.label")
            except Exception as e:
                logger.warning(f"Could not create index: {e}")

    def add_node(self, node: NodeSchema) -> None:
        """Add node to Neo4j"""
        with self.driver.session() as session:
            query = """
            MERGE (n:Concept {id: $id})
            SET n.label = $label,
                n.type = $type,
                n.aliases = $aliases,
                n.properties = $properties,
                n.provenance = $provenance,
                n.created_at = $created_at,
                n.updated_at = $updated_at
            """
            session.run(
                query,
                id=node.id,
                label=node.label,
                type=node.type,
                aliases=list(node.aliases),
                properties=json.dumps(node.properties),
                provenance=json.dumps([p.to_dict() for p in node.provenance]),
                created_at=node.created_at,
                updated_at=node.updated_at
            )

    def add_edge(self, edge: EdgeSchema) -> None:
        """Add edge to Neo4j"""
        with self.driver.session() as session:
            # First ensure both nodes exist
            session.run("""
                MERGE (s:Concept {id: $source_id})
                MERGE (t:Concept {id: $target_id})
            """, source_id=edge.source_id, target_id=edge.target_id)

            # Create relationship
            query = """
            MATCH (s:Concept {id: $source_id})
            MATCH (t:Concept {id: $target_id})
            CREATE (s)-[r:RELATES_TO {
                relation: $relation,
                type: $type,
                confidence: $confidence,
                properties: $properties,
                provenance: $provenance,
                created_at: $created_at,
                updated_at: $updated_at
            }]->(t)
            """
            session.run(
                query,
                source_id=edge.source_id,
                target_id=edge.target_id,
                relation=edge.relation,
                type=edge.type,
                confidence=edge.confidence,
                properties=json.dumps(edge.properties),
                provenance=json.dumps([p.to_dict() for p in edge.provenance]),
                created_at=edge.created_at,
                updated_at=edge.updated_at
            )

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node data from Neo4j"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Concept {id: $id})
                RETURN n
            """, id=node_id)

            record = result.single()
            if record:
                node = record['n']
                return dict(node)
            return None

    def get_neighbors(self, node_id: str, direction: str = 'out') -> List[str]:
        """Get neighbors of a node from Neo4j"""
        with self.driver.session() as session:
            if direction == 'out':
                query = """
                MATCH (n:Concept {id: $id})-[]->(neighbor:Concept)
                RETURN neighbor.id as neighbor_id
                """
            elif direction == 'in':
                query = """
                MATCH (n:Concept {id: $id})<-[]-(neighbor:Concept)
                RETURN neighbor.id as neighbor_id
                """
            else:  # 'both'
                query = """
                MATCH (n:Concept {id: $id})-[]-(neighbor:Concept)
                RETURN neighbor.id as neighbor_id
                """

            result = session.run(query, id=node_id)
            return [record['neighbor_id'] for record in result]

    def get_edges(self, source_id: str = None, target_id: str = None) -> List[Dict[str, Any]]:
        """Get edges matching criteria from Neo4j"""
        with self.driver.session() as session:
            if source_id and target_id:
                query = """
                MATCH (s:Concept {id: $source_id})-[r]->(t:Concept {id: $target_id})
                RETURN s.id as source, t.id as target, r
                """
                result = session.run(query, source_id=source_id, target_id=target_id)
            elif source_id:
                query = """
                MATCH (s:Concept {id: $source_id})-[r]->(t:Concept)
                RETURN s.id as source, t.id as target, r
                """
                result = session.run(query, source_id=source_id)
            else:
                query = """
                MATCH (s:Concept)-[r]->(t:Concept)
                RETURN s.id as source, t.id as target, r
                """
                result = session.run(query)

            edges = []
            for record in result:
                edge_data = dict(record['r'])
                edge_data['source'] = record['source']
                edge_data['target'] = record['target']
                edges.append(edge_data)

            return edges

    def find_paths(self, source_id: str, target_id: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between two nodes in Neo4j"""
        with self.driver.session() as session:
            query = """
            MATCH path = (s:Concept {id: $source_id})-[*1..%d]->(t:Concept {id: $target_id})
            RETURN [node in nodes(path) | node.id] as path
            LIMIT 10
            """ % max_length

            result = session.run(query, source_id=source_id, target_id=target_id)
            return [record['path'] for record in result]

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics from Neo4j"""
        with self.driver.session() as session:
            # Count nodes
            num_nodes = session.run("MATCH (n:Concept) RETURN count(n) as count").single()['count']

            # Count edges
            num_edges = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']

            # Calculate density (for directed graph: density = E / (N * (N-1)))
            density = 0
            if num_nodes > 1:
                density = num_edges / (num_nodes * (num_nodes - 1))

            # Average degree
            avg_degree = 0
            if num_nodes > 0:
                result = session.run("""
                    MATCH (n:Concept)
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN avg(count(r)) as avg_degree
                """)
                avg_degree = result.single()['avg_degree'] or 0

            return {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': density,
                'avg_degree': avg_degree
            }

    def export(self, filepath: str, format: str = 'json') -> None:
        """Export Neo4j graph to file"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            # Export as JSON
            with self.driver.session() as session:
                # Get all nodes
                nodes_result = session.run("MATCH (n:Concept) RETURN n")
                nodes = [dict(record['n']) for record in nodes_result]

                # Get all edges
                edges_result = session.run("""
                    MATCH (s:Concept)-[r]->(t:Concept)
                    RETURN s.id as source, t.id as target, r
                """)
                edges = []
                for record in edges_result:
                    edge_data = dict(record['r'])
                    edge_data['source'] = record['source']
                    edge_data['target'] = record['target']
                    edges.append(edge_data)

                data = {'nodes': nodes, 'edges': edges}

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Neo4j graph exported to {filepath}")

    def clear(self):
        """Clear all data from Neo4j database"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j graph cleared")

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def __del__(self):
        """Cleanup on deletion"""
        try:
            self.close()
        except:
            pass


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
            if NEO4J_AVAILABLE:
                try:
                    self.store = Neo4jStore(config)
                except Exception as e:
                    logger.error(f"Failed to initialize Neo4j, falling back to NetworkX: {e}")
                    self.store = NetworkXStore(config)
            else:
                logger.warning("Neo4j driver not available, using NetworkX fallback")
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
