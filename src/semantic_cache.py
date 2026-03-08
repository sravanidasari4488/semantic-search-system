"""Semantic caching for query result reuse."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.preprocessing import normalize

from src.embedding_pipeline import EmbeddingPipeline

if TYPE_CHECKING:
    from src.clustering import FuzzyClusterer


DEFAULT_THRESHOLD = 0.9


class SemanticCache:
    """In-memory cache that returns cached results for semantically similar queries.

    When a clusterer is provided, cache entries are organized by the query's
    dominant cluster. Lookup only searches within the same cluster, reducing
    search from O(n) to O(n/k) when the cache grows large.
    """

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        threshold: float = DEFAULT_THRESHOLD,
        clusterer: FuzzyClusterer | None = None,
    ) -> None:
        """Initialize the cache.

        Args:
            embedding_pipeline: Used to embed queries for similarity comparison.
            threshold: Minimum cosine similarity for a cache hit (default 0.9).
            clusterer: Optional. When provided, organizes entries by cluster for
                faster lookup when cache grows large.
        """
        self.embedding_pipeline = embedding_pipeline
        self.threshold = threshold
        self.clusterer = clusterer

        self._entries: list[dict[str, Any]] = []
        self._entries_by_cluster: dict[int, list[int]] = {}
        self.hit_count = 0
        self.miss_count = 0

    @property
    def total_entries(self) -> int:
        """Number of cached query-result pairs."""
        return len(self._entries)

    def _get_query_cluster(self, query_emb: np.ndarray) -> int:
        """Return dominant cluster for a query embedding."""
        if self.clusterer is None:
            return 0
        dist = self.clusterer.get_cluster_distribution(query_emb)
        return max(dist, key=dist.get)

    def _get_candidate_indices(self, dominant_cluster: int) -> list[int]:
        """Return indices of entries to search. Uses cluster to narrow when available."""
        if self.clusterer is None:
            return list(range(len(self._entries)))
        if dominant_cluster in self._entries_by_cluster:
            indices = self._entries_by_cluster[dominant_cluster]
            if indices:
                return indices
        return list(range(len(self._entries)))

    def get(self, query: str) -> dict[str, Any]:
        """Return cached result if a semantically similar query exists.

        1. Embed the query and get its dominant cluster.
        2. Only compare against cached entries in the same cluster (when using
           clusterer), reducing lookup cost when cache is large.
        3. If max similarity >= threshold, return cached result (hit).
        4. Otherwise return miss; caller should compute and call set().

        Args:
            query: Search query text.

        Returns:
            Dict with result, matched_query, similarity_score. On hit: result is
            the cached list, matched_query and similarity_score are set. On miss:
            result, matched_query, similarity_score are None.
        """
        miss = {"result": None, "matched_query": None, "similarity_score": None}

        if not self._entries:
            self.miss_count += 1
            return miss

        query_emb = self.embedding_pipeline.generate_embeddings([query])[0]
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        query_emb = normalize(query_emb, norm="l2", axis=1)

        dominant_cluster = self._get_query_cluster(query_emb[0])
        candidate_indices = self._get_candidate_indices(dominant_cluster)

        if not candidate_indices:
            self.miss_count += 1
            return miss

        cached_embeddings = np.vstack(
            [self._entries[i]["embedding"] for i in candidate_indices]
        )
        similarities = np.dot(cached_embeddings, query_emb.T).flatten()
        best_local_idx = int(np.argmax(similarities))
        best_idx = candidate_indices[best_local_idx]
        best_sim = float(similarities[best_local_idx])

        if best_sim >= self.threshold:
            self.hit_count += 1
            return {
                "result": self._entries[best_idx]["result"],
                "matched_query": self._entries[best_idx]["query"],
                "similarity_score": best_sim,
            }

        self.miss_count += 1
        return miss

    def set(self, query: str, result: list[dict[str, Any]]) -> None:
        """Store a query-result pair in the cache.

        When clusterer is set, assigns the entry to its dominant cluster for
        efficient cluster-scoped lookup.

        Args:
            query: Search query text.
            result: Search result to cache (e.g. list of {doc_id, text, score}).
        """
        emb = self.embedding_pipeline.generate_embeddings([query])[0]
        emb = np.asarray(emb, dtype=np.float32)
        emb = normalize(emb.reshape(1, -1), norm="l2", axis=1)[0]

        idx = len(self._entries)
        self._entries.append(
            {
                "query": query,
                "embedding": emb,
                "result": result,
            }
        )

        if self.clusterer is not None:
            cluster = self._get_query_cluster(emb)
            if cluster not in self._entries_by_cluster:
                self._entries_by_cluster[cluster] = []
            self._entries_by_cluster[cluster].append(idx)

    def clear(self) -> None:
        """Clear all cached entries and reset hit/miss counts."""
        self._entries.clear()
        self._entries_by_cluster.clear()
        self.hit_count = 0
        self.miss_count = 0
