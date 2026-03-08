"""Fuzzy clustering for document grouping using Gaussian Mixture Models."""

from typing import TypedDict

import numpy as np
from sklearn.mixture import GaussianMixture


class ClusterResult(TypedDict):
    """Result for a single document: distribution and dominant cluster."""

    distribution: dict[int, float]
    dominant_cluster: int


class FuzzyClusterer:
    """Fuzzy clustering via Gaussian Mixture Models.

    GMM provides soft assignments: each document gets a probability distribution
    over clusters instead of a single hard assignment.
    """

    def __init__(self, n_clusters: int = 20, random_state: int | None = 42) -> None:
        """Initialize the clusterer.

        Args:
            n_clusters: Number of mixture components.
            random_state: Seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: GaussianMixture | None = None

    def fit(self, embeddings: np.ndarray, n_clusters: int = 20) -> "FuzzyClusterer":
        """Fit GMM on embeddings.

        Args:
            embeddings: Array of shape (n_docs, dim).
            n_clusters: Number of components (overrides __init__ if provided).

        Returns:
            self for method chaining.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        self.n_clusters = n_clusters

        self.model = GaussianMixture(
            n_components=n_clusters,
            random_state=self.random_state,
            covariance_type="diag",
        )
        self.model.fit(embeddings)

        return self

    def get_cluster_distribution(self, embedding: np.ndarray) -> dict[int, float]:
        """Return probability distribution over clusters for one embedding.

        Args:
            embedding: Single vector of shape (dim,) or (1, dim).

        Returns:
            Dict mapping cluster_id (0-indexed) to probability.
            Example: {0: 0.65, 3: 0.25, 8: 0.10}
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        embedding = np.asarray(embedding, dtype=np.float64)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        probs = self.model.predict_proba(embedding)[0]
        return {i: float(p) for i, p in enumerate(probs)}

    def cluster_documents(self, embeddings: np.ndarray) -> list[ClusterResult]:
        """Return probability distributions and dominant cluster per document.

        Args:
            embeddings: Array of shape (n_docs, dim).

        Returns:
            List of dicts, each with:
            - distribution: {cluster_id: probability}
            - dominant_cluster: cluster with highest probability
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        probs = self.model.predict_proba(embeddings)
        dominant = np.argmax(probs, axis=1)

        results: list[ClusterResult] = []
        for i in range(len(embeddings)):
            results.append(
                {
                    "distribution": {j: float(probs[i, j]) for j in range(self.n_clusters)},
                    "dominant_cluster": int(dominant[i]),
                }
            )

        return results
