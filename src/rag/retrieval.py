"""RAG (Retrieval-Augmented Generation) pipeline"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
import json


@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    text: str
    metadata: Dict[str, Any]
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'metadata': self.metadata,
            'score': self.score,
            'rank': self.rank
        }


class VectorStore:
    """Vector database wrapper for Chroma"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_config = config.get('rag', {})

        # Get configuration
        persist_directory = self.rag_config.get('chroma', {}).get('persist_directory', './data/chroma_db')
        collection_name = self.rag_config.get('chroma', {}).get('collection_name', 'knowledge_base')

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Personal knowledge base"}
        )

        # Load embedding model
        embedding_model = config.get('models', {}).get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)

        self.embedding_dim = self.rag_config.get('embedding_dimension', 384)

        logger.info(f"Vector store initialized with collection: {collection_name}")

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]] = None,
        ids: List[str] = None
    ) -> None:
        """Add documents to vector store"""
        if not texts:
            return

        # Generate IDs if not provided
        if ids is None:
            start_id = self.collection.count()
            ids = [f"doc_{start_id + i}" for i in range(len(texts))]

        # Ensure metadatas is provided
        if metadatas is None:
            metadatas = [{}] * len(texts)

        # Convert metadata values to strings (Chroma requirement)
        metadatas = [self._serialize_metadata(m) for m in metadatas]

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(texts)} documents...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))

            self.collection.add(
                documents=texts[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )

        logger.info(f"Added {len(texts)} documents to vector store")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        """Search for similar documents"""
        # Compute query embedding
        query_embedding = self.embedder.encode([query])[0].tolist()

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )

        # Format results
        retrieval_results = []

        if results['documents'] and results['documents'][0]:
            for rank, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                # Convert distance to similarity score (cosine similarity)
                score = 1 - distance

                result = RetrievalResult(
                    text=doc,
                    metadata=self._deserialize_metadata(metadata),
                    score=score,
                    rank=rank + 1
                )
                retrieval_results.append(result)

        return retrieval_results

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Convert metadata values to strings"""
        serialized = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = str(value)
        return serialized

    def _deserialize_metadata(self, metadata: Dict[str, str]) -> Dict[str, Any]:
        """Convert metadata strings back to original types"""
        deserialized = {}
        for key, value in metadata.items():
            try:
                # Try to parse as JSON
                deserialized[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Keep as string
                deserialized[key] = value
        return deserialized

    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name,
            'embedding_dimension': self.embedding_dim
        }

    def clear(self) -> None:
        """Clear all documents from collection"""
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"description": "Personal knowledge base"}
        )
        logger.info("Vector store cleared")


class RAGPipeline:
    """Complete RAG pipeline for retrieval-augmented querying"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_config = config.get('rag', {})

        self.top_k = self.rag_config.get('top_k', 5)
        self.similarity_threshold = self.rag_config.get('similarity_threshold', 0.7)

        # Initialize vector store
        self.vector_store = VectorStore(config)

        logger.info("RAG pipeline initialized")

    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents into vector store"""
        logger.info(f"Indexing {len(documents)} documents...")

        texts = []
        metadatas = []

        for doc in documents:
            # Use content as text
            text = doc.get('content', '')
            if not text:
                continue

            # Add chunks if available
            if 'chunks' in doc and doc['chunks']:
                for i, chunk in enumerate(doc['chunks']):
                    texts.append(chunk)

                    # Create metadata
                    metadata = doc.get('metadata', {}).copy()
                    metadata['chunk_index'] = i
                    metadata['total_chunks'] = len(doc['chunks'])
                    metadatas.append(metadata)
            else:
                texts.append(text)
                metadatas.append(doc.get('metadata', {}))

        # Add to vector store
        self.vector_store.add_documents(texts, metadatas)

        logger.info(f"Indexed {len(texts)} text chunks")

    def index_from_directory(self, directory: str) -> None:
        """Index all processed documents from a directory"""
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return

        # Load all JSON documents
        documents = []
        for json_file in dir_path.glob('*.json'):
            if json_file.name == '_summary.json':
                continue

            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
                documents.append(doc)

        if documents:
            self.index_documents(documents)
        else:
            logger.warning(f"No documents found in {directory}")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        if top_k is None:
            top_k = self.top_k

        logger.info(f"Retrieving top {top_k} results for query: {query[:100]}...")

        # Search vector store
        results = self.vector_store.search(query, top_k=top_k, filter=filter_metadata)

        # Filter by similarity threshold
        filtered_results = [r for r in results if r.score >= self.similarity_threshold]

        logger.info(f"Retrieved {len(filtered_results)} results above threshold")

        return filtered_results

    def retrieve_with_context(self, query: str, graph_store = None) -> Dict[str, Any]:
        """Retrieve documents with additional context from knowledge graph"""
        # Get retrieval results
        results = self.retrieve(query)

        response = {
            'query': query,
            'results': [r.to_dict() for r in results],
            'context': {}
        }

        # Add graph context if available
        if graph_store and results:
            # Extract key concepts from top results
            concepts = set()
            for result in results[:3]:
                # Simple concept extraction (can be improved)
                words = result.text.split()
                for word in words:
                    if len(word) > 4 and word[0].isupper():
                        concepts.add(word)

            # Get graph information about concepts
            graph_context = []
            if hasattr(graph_store, 'store'):
                graph = graph_store.store.graph
            else:
                graph = graph_store.graph

            for concept in list(concepts)[:5]:
                concept_id = concept.lower().replace(' ', '_')
                if concept_id in graph.nodes():
                    neighbors = list(graph.neighbors(concept_id))[:3]
                    graph_context.append({
                        'concept': concept,
                        'related': neighbors
                    })

            response['context']['graph'] = graph_context

        return response

    def answer_question(
        self,
        question: str,
        graph_store = None,
        use_llm: bool = False
    ) -> str:
        """Answer a question using retrieved context"""
        # Retrieve relevant context
        context = self.retrieve_with_context(question, graph_store)

        if not context['results']:
            return "I don't have enough information to answer that question."

        # Build answer from context
        answer_parts = ["Based on your knowledge base:\n"]

        for i, result in enumerate(context['results'][:3], 1):
            text = result['text'][:300]  # Truncate long text
            score = result['score']
            answer_parts.append(f"\n{i}. (Relevance: {score:.2f}) {text}...")

        # Add graph context
        if 'graph' in context.get('context', {}):
            answer_parts.append("\n\nRelated concepts:")
            for item in context['context']['graph']:
                related = ', '.join(item['related'])
                answer_parts.append(f"- {item['concept']}: related to {related}")

        return ''.join(answer_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics"""
        return {
            'vector_store': self.vector_store.get_statistics(),
            'config': {
                'top_k': self.top_k,
                'similarity_threshold': self.similarity_threshold
            }
        }
