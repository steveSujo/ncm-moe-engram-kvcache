"""
Retriever for psudo-Engine Context Subsystem.

Hybrid BM25 + embedding-based retrieval for context assembly.
Uses all-MiniLM-L6-v2 for semantic search and BM25 for keyword matching.
"""

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


@dataclass
class Document:
    """A document in the retrieval index."""
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    bm25_terms: dict[str, int] = field(default_factory=dict)


class BM25Index:
    """BM25 (Okapi BM25) index for keyword-based retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lens: list[int] = []
        self.avgdl: float = 0
        self._built = False

    def add_document(self, doc: Document):
        """Add a document to the index."""
        self.documents.append(doc)
        self.doc_lens.append(len(doc.content.split()))

        # Update document frequencies
        terms = set(doc.content.lower().split())
        for term in terms:
            self.doc_freqs[term] += 1

        self._built = False

    def build(self):
        """Build the BM25 index statistics."""
        if self.documents:
            self.avgdl = sum(self.doc_lens) / len(self.doc_lens)
        self._built = True

    def _idf(self, term: str) -> float:
        """Compute inverse document frequency."""
        n = self.doc_freqs.get(term, 0)
        if n == 0:
            return 0.0
        N = len(self.documents)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a query against a document."""
        if not self._built:
            self.build()

        query_terms = query.lower().split()
        doc = self.documents[doc_idx]
        doc_len = self.doc_lens[doc_idx]

        score = 0.0
        for term in query_terms:
            tf = doc.content.lower().split().count(term)
            if tf == 0:
                continue

            idf = self._idf(term)
            score += idf * (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            )

        return score

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, int]]:
        """Search for top-k documents by BM25 score."""
        if not self._built:
            self.build()

        scores = [(self.score(query, i), i) for i in range(len(self.documents))]
        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]


class EmbeddingIndex:
    """Embedding-based semantic search index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if __import__('torch').cuda.is_available() else 'cpu')
        self.documents: list[Document] = []
        self.embeddings: np.ndarray | None = None
        self.model: Any | None = None
        self._built = False

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self._init_model()

    def _init_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception:
            self.model = None

    def add_document(self, doc: Document):
        """Add a document to the index."""
        self.documents.append(doc)
        self._built = False

    def build(self, batch_size: int = 32):
        """Build embeddings for all documents."""
        if not self.documents or self.model is None:
            self._built = True
            return

        texts = [doc.content for doc in self.documents]

        # Encode in batches
        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
            embeddings_list.append(emb)

        self.embeddings = np.vstack(embeddings_list)

        # Store embeddings in documents
        for i, doc in enumerate(self.documents):
            doc.embedding = self.embeddings[i]

        self._built = True

    def search(self, query: str, top_k: int = 10) -> list[tuple[float, int]]:
        """Search for top-k documents by cosine similarity."""
        if not self._built or self.model is None or self.embeddings is None:
            return []

        # Encode query
        query_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        # Compute cosine similarities
        similarities = np.dot(self.embeddings, query_emb.T).flatten()

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(float(similarities[i]), i) for i in top_indices]


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and embedding search.

    Uses reciprocal rank fusion (RRF) to combine results from both indices.
    """

    def __init__(
        self,
        bm25_weight: float = 0.5,
        embedding_weight: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None
    ):
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight

        self.bm25 = BM25Index()
        self.embeddings = EmbeddingIndex(embedding_model, device)

        self.documents: list[Document] = []
        self._id_map: dict[str, int] = {}  # doc_id -> index

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> Document:
        """Add a document to both indices."""
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {}
        )

        idx = len(self.documents)
        self.documents.append(doc)
        self._id_map[doc_id] = idx

        self.bm25.add_document(doc)
        self.embeddings.add_document(doc)

        return doc

    def add_documents(self, docs: list[tuple[str, str, dict[str, Any]]]):
        """Add multiple documents at once."""
        for doc_id, content, metadata in docs:
            self.add_document(doc_id, content, metadata)

    def build(self):
        """Build both indices."""
        self.bm25.build()
        self.embeddings.build()

    def search(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60
    ) -> list[tuple[float, Document]]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).

        Args:
            query: Search query
            top_k: Number of results to return
            rrf_k: RRF constant (typically 60)

        Returns:
            List of (fused_score, Document) tuples
        """
        # Get results from both indices
        bm25_results = self.bm25.search(query, top_k * 2)
        emb_results = self.embeddings.search(query, top_k * 2)

        # RRF fusion
        rrf_scores: dict[int, float] = defaultdict(float)

        # BM25 ranks
        for rank, (_score, idx) in enumerate(bm25_results):
            rrf_scores[idx] += self.bm25_weight / (rrf_k + rank + 1)

        # Embedding ranks
        for rank, (_score, idx) in enumerate(emb_results):
            rrf_scores[idx] += self.embedding_weight / (rrf_k + rank + 1)

        # Sort by fused score
        fused = sorted(rrf_scores.items(), key=lambda x: -x[1])

        # Return top-k with documents
        results = []
        for idx, score in fused[:top_k]:
            if idx < len(self.documents):
                results.append((score, self.documents[idx]))

        return results

    def search_bm25_only(self, query: str, top_k: int = 10) -> list[tuple[float, Document]]:
        """Search using BM25 only."""
        results = self.bm25.search(query, top_k)
        return [(score, self.documents[idx]) for score, idx in results]

    def search_embedding_only(self, query: str, top_k: int = 10) -> list[tuple[float, Document]]:
        """Search using embeddings only."""
        results = self.embeddings.search(query, top_k)
        return [(score, self.documents[idx]) for score, idx in results]

    def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID."""
        idx = self._id_map.get(doc_id)
        if idx is not None and idx < len(self.documents):
            return self.documents[idx]
        return None

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index (requires rebuild)."""
        if doc_id not in self._id_map:
            return False

        idx = self._id_map[doc_id]
        # Mark as removed (lazy deletion)
        self.documents[idx] = None
        del self._id_map[doc_id]
        return True

    def save(self, path: str):
        """Save index to disk."""
        data = {
            'documents': [
                {
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                }
                for doc in self.documents if doc is not None
            ],
            'bm25_weight': self.bm25_weight,
            'embedding_weight': self.embedding_weight,
            'embedding_model': self.embeddings.model_name,
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

        # Save embeddings separately if they exist
        if self.embeddings.embeddings is not None:
            emb_path = path.replace('.pkl', '_embeddings.npy')
            np.save(emb_path, self.embeddings.embeddings)

    @classmethod
    def load(cls, path: str, device: str | None = None) -> 'HybridRetriever':
        """Load index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        retriever = cls(
            bm25_weight=data.get('bm25_weight', 0.5),
            embedding_weight=data.get('embedding_weight', 0.5),
            embedding_model=data.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"),
            device=device
        )

        for doc_data in data['documents']:
            retriever.add_document(
                doc_data['id'],
                doc_data['content'],
                doc_data.get('metadata', {})
            )

        retriever.build()

        # Load embeddings if available
        emb_path = path.replace('.pkl', '_embeddings.npy')
        if Path(emb_path).exists():
            retriever.embeddings.embeddings = np.load(emb_path)
            for i, doc in enumerate(retriever.documents):
                if doc is not None:
                    doc.embedding = retriever.embeddings.embeddings[i]
            retriever.embeddings._built = True

        return retriever

    def get_stats(self) -> dict[str, Any]:
        """Get retriever statistics."""
        return {
            'num_documents': len([d for d in self.documents if d is not None]),
            'bm25_built': self.bm25._built,
            'embeddings_built': self.embeddings._built,
            'embedding_model': self.embeddings.model_name,
            'device': self.embeddings.device,
        }


class SyntaxRuleRetriever:
    """
    Specialized retriever for syntax rules from SyntaxHub.
    """

    def __init__(self, syntax_hub: Any):
        self.syntax_hub = syntax_hub
        self.retriever = HybridRetriever(
            bm25_weight=0.4,
            embedding_weight=0.6
        )
        self._indexed = False

    def index_syntax_hub(self):
        """Index all rules from the syntax hub."""
        if self._indexed:
            return

        for rule in self.syntax_hub.rules_by_id.values():
            content = f"{rule.title}\n{rule.description}\n{rule.pattern}"
            self.retriever.add_document(
                doc_id=rule.id,
                content=content,
                metadata={
                    'category': rule.category,
                    'language': rule.language,
                    'tags': rule.tags,
                    'priority': rule.priority,
                    'file_source': rule.file_source,
                }
            )

        self.retriever.build()
        self._indexed = True

    def retrieve_for_task(
        self,
        task: str,
        language: str | None = None,
        top_k: int = 10
    ) -> list[tuple[float, Any]]:
        """Retrieve syntax rules relevant to a task."""
        if not self._indexed:
            self.index_syntax_hub()

        results = self.retriever.search(task, top_k=top_k * 2)

        # Filter by language if specified
        if language:
            filtered = []
            for score, doc in results:
                if doc.metadata.get('language') == language or doc.metadata.get('language') == 'generic':
                    filtered.append((score, doc))
                if len(filtered) >= top_k:
                    break
            results = filtered

        # Convert back to syntax rules
        rule_results = []
        for score, doc in results[:top_k]:
            rule = self.syntax_hub.rules_by_id.get(doc.id)
            if rule:
                rule_results.append((score, rule))

        return rule_results


def create_retriever(config: dict | None = None) -> HybridRetriever:
    """Create retriever from config."""
    config = config or {}
    return HybridRetriever(
        bm25_weight=config.get('bm25_weight', 0.5),
        embedding_weight=config.get('embedding_weight', 0.5),
        embedding_model=config.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2"),
        device=config.get('device')
    )


if __name__ == "__main__":
    # Quick test

    retriever = HybridRetriever(
        bm25_weight=0.5,
        embedding_weight=0.5
    )

    # Add test documents
    docs = [
        ("func_1", "def read_file(path: str) -> str:\n    with open(path) as f:\n        return f.read()",
         {"language": "python", "type": "builtin"}),
        ("func_2", "fn read_file(path: &str) -> String {\n    std::fs::read_to_string(path)\n}",
         {"language": "rust", "type": "builtin"}),
        ("func_3", "read_file(path: string) -> string",
         {"language": "psudo", "type": "builtin"}),
        ("func_4", "def write_file(path: str, content: str) -> None:\n    with open(path, 'w') as f:\n        f.write(content)",
         {"language": "python", "type": "builtin"}),
        ("func_5", "for item in items:\n    print(item)",
         {"language": "python", "type": "control_flow"}),
    ]

    retriever.add_documents(docs)
    retriever.build()

    # Search
    results = retriever.search("read file", top_k=3)
    for _score, _doc in results:
        pass

    # Test BM25 only
    bm25_results = retriever.search_bm25_only("read file", top_k=3)
    for _score, _doc in bm25_results:
        pass

    # Test embedding only
    emb_results = retriever.search_embedding_only("read file", top_k=3)
    for _score, _doc in emb_results:
        pass

