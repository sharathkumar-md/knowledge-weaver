"""Linker/Deduper Agent - entity resolution and clustering"""

from typing import Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from loguru import logger


@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    name: str
    aliases: Set[str] = field(default_factory=set)
    mentions: List[str] = field(default_factory=list)
    canonical_form: str = None
    cluster_id: int = None
    embedding: np.ndarray = None

    def __post_init__(self):
        if self.canonical_form is None:
            self.canonical_form = self.name
        self.aliases.add(self.name)
        if self.name not in self.mentions:
            self.mentions.append(self.name)


class LinkerAgent:
    """Agent for entity linking and deduplication"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agent_config = config.get('agents', {}).get('linker', {})

        self.similarity_threshold = self.agent_config.get('similarity_threshold', 0.85)
        self.clustering_algorithm = self.agent_config.get('clustering_algorithm', 'agglomerative')
        self.min_cluster_size = self.agent_config.get('min_cluster_size', 2)

        # Load embedding model
        embedding_model = config.get('models', {}).get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        self.entities: Dict[str, Entity] = {}
        self.clusters: Dict[int, List[str]] = defaultdict(list)

    def add_entities_from_triples(self, triples: List[Dict[str, Any]]) -> None:
        """Extract entities from triples"""
        logger.info(f"Adding entities from {len(triples)} triples")

        for triple in triples:
            subject = triple['subject']
            obj = triple['object']

            # Add subject
            if subject not in self.entities:
                self.entities[subject] = Entity(name=subject)
            else:
                self.entities[subject].mentions.append(subject)

            # Add object
            if obj not in self.entities:
                self.entities[obj] = Entity(name=obj)
            else:
                self.entities[obj].mentions.append(obj)

        logger.info(f"Total unique entities: {len(self.entities)}")

    def compute_embeddings(self) -> None:
        """Compute embeddings for all entities"""
        logger.info("Computing entity embeddings...")

        entity_names = list(self.entities.keys())
        embeddings = self.embedder.encode(entity_names, show_progress_bar=True)

        for name, embedding in zip(entity_names, embeddings):
            self.entities[name].embedding = embedding

        logger.info("Embeddings computed")

    def cluster_entities(self) -> Dict[int, List[str]]:
        """Cluster similar entities together"""
        logger.info(f"Clustering entities using {self.clustering_algorithm}")

        if not self.entities:
            logger.warning("No entities to cluster")
            return {}

        # Get embeddings
        entity_names = list(self.entities.keys())
        embeddings = np.array([self.entities[name].embedding for name in entity_names])

        # Cluster
        if self.clustering_algorithm == 'agglomerative':
            clusters = self._cluster_agglomerative(embeddings)
        elif self.clustering_algorithm == 'dbscan':
            clusters = self._cluster_dbscan(embeddings)
        else:
            logger.error(f"Unknown clustering algorithm: {self.clustering_algorithm}")
            return {}

        # Assign cluster IDs to entities
        cluster_map = defaultdict(list)

        for entity_name, cluster_id in zip(entity_names, clusters):
            self.entities[entity_name].cluster_id = int(cluster_id)
            cluster_map[int(cluster_id)].append(entity_name)

        self.clusters = cluster_map

        # Log statistics
        num_clusters = len(set(clusters))
        logger.info(f"Created {num_clusters} clusters from {len(entity_names)} entities")

        # Merge entities in same cluster
        self._merge_clusters()

        return self.clusters

    def _cluster_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using Agglomerative Clustering"""
        distance_threshold = 1 - self.similarity_threshold

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage='average',
            metric='cosine'
        )

        return clustering.fit_predict(embeddings)

    def _cluster_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster using DBSCAN"""
        # Convert similarity threshold to epsilon (distance)
        eps = 1 - self.similarity_threshold

        clustering = DBSCAN(
            eps=eps,
            min_samples=self.min_cluster_size,
            metric='cosine'
        )

        return clustering.fit_predict(embeddings)

    def _merge_clusters(self) -> None:
        """Merge entities within each cluster"""
        logger.info("Merging entities in clusters...")

        merged_count = 0

        for cluster_id, entity_names in self.clusters.items():
            if len(entity_names) < 2:
                continue  # No merging needed

            # Choose canonical form (most frequent or longest name)
            canonical = self._choose_canonical(entity_names)

            # Merge all aliases
            aliases = set()
            mentions = []

            for name in entity_names:
                entity = self.entities[name]
                aliases.update(entity.aliases)
                mentions.extend(entity.mentions)

            # Update canonical entity
            if canonical in self.entities:
                self.entities[canonical].aliases = aliases
                self.entities[canonical].mentions = mentions
                self.entities[canonical].canonical_form = canonical

            merged_count += len(entity_names) - 1

        logger.info(f"Merged {merged_count} duplicate entities")

    def _choose_canonical(self, entity_names: List[str]) -> str:
        """Choose the canonical form from a list of entity names"""
        # Choose the most common form, or the longest if tied
        mention_counts = defaultdict(int)

        for name in entity_names:
            entity = self.entities[name]
            for mention in entity.mentions:
                mention_counts[mention] += 1

        # Sort by count (descending) then length (descending)
        sorted_names = sorted(
            entity_names,
            key=lambda n: (mention_counts[n], len(n)),
            reverse=True
        )

        return sorted_names[0]

    def link_entity(self, entity_name: str) -> str:
        """Link an entity name to its canonical form"""
        if entity_name in self.entities:
            entity = self.entities[entity_name]
            return entity.canonical_form

        # Try fuzzy matching
        closest = self._find_closest_entity(entity_name)
        if closest:
            return self.entities[closest].canonical_form

        # No match found, return original
        return entity_name

    def _find_closest_entity(self, entity_name: str) -> str:
        """Find the closest matching entity using embeddings"""
        if not self.entities:
            return None

        # Compute embedding for query
        query_embedding = self.embedder.encode([entity_name])[0]

        # Find most similar
        best_match = None
        best_similarity = 0

        for name, entity in self.entities.items():
            if entity.embedding is None:
                continue

            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                entity.embedding.reshape(1, -1)
            )[0][0]

            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = name

        return best_match

    def update_triples(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update triples with canonical entity forms"""
        logger.info("Updating triples with canonical forms...")

        updated_triples = []

        for triple in triples:
            updated_triple = triple.copy()

            # Link subject and object
            updated_triple['subject'] = self.link_entity(triple['subject'])
            updated_triple['object'] = self.link_entity(triple['object'])

            # Only keep if subject and object are different
            if updated_triple['subject'] != updated_triple['object']:
                updated_triples.append(updated_triple)

        # Deduplicate
        seen = set()
        final_triples = []

        for triple in updated_triples:
            key = (triple['subject'], triple['relation'], triple['object'])
            if key not in seen:
                seen.add(key)
                final_triples.append(triple)

        logger.info(f"Reduced {len(triples)} triples to {len(final_triples)} after linking")

        return final_triples

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about entity linking"""
        total_entities = len(self.entities)
        total_clusters = len(self.clusters)

        cluster_sizes = [len(names) for names in self.clusters.values()]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

        return {
            'total_entities': total_entities,
            'total_clusters': total_clusters,
            'avg_cluster_size': avg_cluster_size,
            'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
            'singletons': sum(1 for size in cluster_sizes if size == 1)
        }
