"""Embedding pipeline for generating vector representations."""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

if TYPE_CHECKING:
    from collections.abc import Sequence


BATCH_SIZE = 64
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingPipeline:
    """Generates and manages L2-normalized embeddings for semantic search."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        """Initialize the pipeline with a sentence-transformers model.

        Args:
            model_name: HuggingFace model id (e.g. all-MiniLM-L6-v2).
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._embeddings: np.ndarray | None = None

    def generate_embeddings(self, texts: "Sequence[str]") -> np.ndarray:
        """Generate L2-normalized embeddings for a list of texts.

        Normalization is done because:
        - Cosine similarity = dot product when vectors are unit-length. L2-normalizing
          lets us use fast inner-product search (e.g. FAISS) instead of full cosine.
        - It makes similarity scores comparable across documents of different length.
        - Many vector indexes (FAISS, Annoy) expect or perform better with unit vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            numpy array of shape (n_texts, embedding_dim), L2-normalized.
        """
        if not texts:
            self._embeddings = np.array([]).reshape(
                0, self.model.get_sentence_embedding_dimension()
            )
            return self._embeddings

        embeddings = self.model.encode(
            list(texts),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # L2-normalize so dot product equals cosine similarity
        embeddings = normalize(embeddings, norm="l2", axis=1)
        self._embeddings = embeddings.astype(np.float32)

        return self._embeddings

    def save_embeddings(self, path: str | Path, embeddings: np.ndarray) -> None:
        """Save embeddings to a .npy file.

        Args:
            path: Output file path (e.g. data/embeddings.npy).
            embeddings: Array of shape (n_docs, embedding_dim) to save.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    def load_embeddings(self, path: str | Path) -> np.ndarray:
        """Load embeddings from a .npy file.

        Args:
            path: Path to saved .npy file.

        Returns:
            numpy array of shape (n_docs, embedding_dim).
        """
        return np.load(path)
