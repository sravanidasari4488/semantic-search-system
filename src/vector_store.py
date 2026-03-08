"""Vector database storage using FAISS."""

from pathlib import Path

import faiss
import numpy as np
from sklearn.preprocessing import normalize


class VectorStore:
    """FAISS-based vector store for similarity search.

    Uses IndexFlatIP (inner product). Embeddings are L2-normalized before indexing
    so inner product equals cosine similarity. Higher scores = more similar.
    """

    def __init__(self, dimension: int | None = None) -> None:
        """Initialize the store.

        Args:
            dimension: Embedding dimension. Required for load_index(); optional
                when building from scratch (inferred from embeddings).
        """
        self.dimension = dimension
        self.index: faiss.IndexFlatIP | None = None

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings.

        Embeddings are L2-normalized before indexing so IndexFlatIP returns
        cosine similarity (inner product of unit vectors).

        Args:
            embeddings: Array of shape (n_docs, dim), dtype float32.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D (n_docs, dim)")

        self.dimension = embeddings.shape[1]

        # Ensure unit vectors for correct inner-product-as-cosine behavior
        embeddings = normalize(embeddings, norm="l2", axis=1).astype(np.float32)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for most similar documents.

        Args:
            query_embedding: Query vector of shape (dim,) or (1, dim).
            top_k: Number of neighbors to return.

        Returns:
            Tuple of (indices, scores). indices: doc indices, scores: similarity
            (inner product = cosine similarity for normalized vectors).
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index or load_index first.")

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        query = normalize(query, norm="l2", axis=1).astype(np.float32)

        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        return indices[0], scores[0]

    def save_index(self, path: str | Path) -> None:
        """Save FAISS index to disk.

        Args:
            path: Output path (e.g. data/faiss.index).
        """
        if self.index is None:
            raise ValueError("No index to save. Call build_index first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load_index(self, path: str | Path) -> None:
        """Load FAISS index from disk.

        Args:
            path: Path to saved index file.
        """
        self.index = faiss.read_index(str(path))
        self.dimension = self.index.d
