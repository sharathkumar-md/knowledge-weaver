"""Schema Manager - manages node/edge types, provenance, metadata"""

from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import json
from loguru import logger


class NodeType(Enum):
    """Predefined node types"""
    CONCEPT = "concept"
    ENTITY = "entity"
    EVENT = "event"
    ATTRIBUTE = "attribute"
    CATEGORY = "category"


class EdgeType(Enum):
    """Predefined edge types"""
    IS_A = "IS_A"
    HAS_PART = "HAS_PART"
    HAS_PROPERTY = "HAS_PROPERTY"
    CAUSES = "CAUSES"
    LEADS_TO = "LEADS_TO"
    USED_FOR = "USED_FOR"
    SIMILAR_TO = "SIMILAR_TO"
    RELATED_TO = "RELATED_TO"
    BEFORE = "BEFORE"
    AFTER = "AFTER"


@dataclass
class Provenance:
    """Provenance information for a node or edge"""
    source: str
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeSchema:
    """Schema for a graph node"""
    id: str
    label: str
    type: str = NodeType.CONCEPT.value
    aliases: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Provenance] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['aliases'] = list(data['aliases'])
        data['provenance'] = [p.to_dict() if isinstance(p, Provenance) else p for p in data['provenance']]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeSchema':
        data = data.copy()
        data['aliases'] = set(data.get('aliases', []))
        provs = data.get('provenance', [])
        data['provenance'] = [Provenance(**p) if isinstance(p, dict) else p for p in provs]
        return cls(**data)


@dataclass
class EdgeSchema:
    """Schema for a graph edge"""
    source_id: str
    target_id: str
    relation: str
    type: str = EdgeType.RELATED_TO.value
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: List[Provenance] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['provenance'] = [p.to_dict() if isinstance(p, Provenance) else p for p in data['provenance']]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EdgeSchema':
        data = data.copy()
        provs = data.get('provenance', [])
        data['provenance'] = [Provenance(**p) if isinstance(p, dict) else p for p in provs]
        return cls(**data)


class SchemaManager:
    """Manages schema, types, and provenance for the knowledge graph"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph_config = config.get('graph', {})

        self.provenance_tracking = self.graph_config.get('provenance_tracking', True)
        self.confidence_threshold = self.graph_config.get('confidence_threshold', 0.6)

        self.nodes: Dict[str, NodeSchema] = {}
        self.edges: List[EdgeSchema] = []

        logger.info("Schema Manager initialized")

    def create_node(
        self,
        label: str,
        node_type: str = None,
        properties: Dict[str, Any] = None,
        provenance: Provenance = None
    ) -> NodeSchema:
        """Create a new node with schema"""
        # Generate ID from label
        node_id = self._generate_node_id(label)

        # Check if node already exists
        if node_id in self.nodes:
            node = self.nodes[node_id]
            # Update existing node
            if properties:
                node.properties.update(properties)
            if provenance:
                node.provenance.append(provenance)
            node.updated_at = datetime.now().isoformat()
            return node

        # Create new node
        node = NodeSchema(
            id=node_id,
            label=label,
            type=node_type or NodeType.CONCEPT.value,
            properties=properties or {},
            provenance=[provenance] if provenance else []
        )

        self.nodes[node_id] = node
        return node

    def create_edge(
        self,
        source_label: str,
        target_label: str,
        relation: str,
        edge_type: str = None,
        confidence: float = 1.0,
        properties: Dict[str, Any] = None,
        provenance: Provenance = None
    ) -> Optional[EdgeSchema]:
        """Create a new edge with schema"""
        # Get or create nodes
        source_node = self.create_node(source_label)
        target_node = self.create_node(target_label)

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            logger.debug(f"Skipping low-confidence edge: {source_label} -> {target_label} ({confidence})")
            return None

        # Check if edge already exists
        existing = self._find_edge(source_node.id, target_node.id, relation)
        if existing:
            # Update existing edge
            if properties:
                existing.properties.update(properties)
            if provenance:
                existing.provenance.append(provenance)
            existing.confidence = max(existing.confidence, confidence)
            existing.updated_at = datetime.now().isoformat()
            return existing

        # Create new edge
        edge = EdgeSchema(
            source_id=source_node.id,
            target_id=target_node.id,
            relation=relation,
            type=edge_type or self._infer_edge_type(relation),
            confidence=confidence,
            properties=properties or {},
            provenance=[provenance] if provenance else []
        )

        self.edges.append(edge)
        return edge

    def add_triples(self, triples: List[Dict[str, Any]]) -> None:
        """Add multiple triples to the graph"""
        logger.info(f"Adding {len(triples)} triples to schema...")

        added_nodes = 0
        added_edges = 0

        for triple in triples:
            subject = triple.get('subject')
            relation = triple.get('relation')
            obj = triple.get('object')
            confidence = triple.get('confidence', 1.0)

            if not all([subject, relation, obj]):
                continue

            # Create provenance
            prov = None
            if self.provenance_tracking and 'provenance' in triple:
                prov_data = triple['provenance']
                prov = Provenance(
                    source=prov_data.get('source', 'unknown'),
                    method=prov_data.get('method', 'unknown'),
                    confidence=confidence,
                    metadata=prov_data
                )

            # Create edge (will create nodes if needed)
            edge = self.create_edge(
                source_label=subject,
                target_label=obj,
                relation=relation,
                confidence=confidence,
                provenance=prov
            )

            if edge:
                added_edges += 1

        added_nodes = len(self.nodes)

        logger.info(f"Added {added_nodes} nodes and {added_edges} edges")

    def _generate_node_id(self, label: str) -> str:
        """Generate a unique node ID from label"""
        # Normalize label for ID
        node_id = label.lower().replace(' ', '_')
        # Remove special characters
        node_id = ''.join(c for c in node_id if c.isalnum() or c == '_')
        return node_id

    def _infer_edge_type(self, relation: str) -> str:
        """Infer edge type from relation name"""
        relation_upper = relation.upper()

        # Check if it matches a predefined type
        for edge_type in EdgeType:
            if edge_type.value == relation_upper:
                return edge_type.value

        # Default to RELATED_TO
        return EdgeType.RELATED_TO.value

    def _find_edge(self, source_id: str, target_id: str, relation: str) -> Optional[EdgeSchema]:
        """Find an existing edge"""
        for edge in self.edges:
            if (edge.source_id == source_id and
                edge.target_id == target_id and
                edge.relation == relation):
                return edge
        return None

    def get_node(self, node_id: str) -> Optional[NodeSchema]:
        """Get node by ID"""
        return self.nodes.get(node_id)

    def get_node_by_label(self, label: str) -> Optional[NodeSchema]:
        """Get node by label"""
        node_id = self._generate_node_id(label)
        return self.nodes.get(node_id)

    def get_edges_from_node(self, node_id: str) -> List[EdgeSchema]:
        """Get all edges from a node"""
        return [e for e in self.edges if e.source_id == node_id]

    def get_edges_to_node(self, node_id: str) -> List[EdgeSchema]:
        """Get all edges to a node"""
        return [e for e in self.edges if e.target_id == node_id]

    def get_statistics(self) -> Dict[str, Any]:
        """Get schema statistics"""
        node_types = {}
        for node in self.nodes.values():
            node_types[node.type] = node_types.get(node.type, 0) + 1

        edge_types = {}
        for edge in self.edges:
            edge_types[edge.type] = edge_types.get(edge.type, 0) + 1

        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'node_types': node_types,
            'edge_types': edge_types,
            'avg_confidence': sum(e.confidence for e in self.edges) / len(self.edges) if self.edges else 0,
            'nodes_with_provenance': sum(1 for n in self.nodes.values() if n.provenance),
            'edges_with_provenance': sum(1 for e in self.edges if e.provenance)
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export schema to dictionary"""
        return {
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            'edges': [edge.to_dict() for edge in self.edges],
            'statistics': self.get_statistics()
        }

    def save(self, filepath: str) -> None:
        """Save schema to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Schema saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load schema from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load nodes
        self.nodes = {}
        for node_id, node_data in data['nodes'].items():
            self.nodes[node_id] = NodeSchema.from_dict(node_data)

        # Load edges
        self.edges = []
        for edge_data in data['edges']:
            self.edges.append(EdgeSchema.from_dict(edge_data))

        logger.info(f"Schema loaded from {filepath}")
