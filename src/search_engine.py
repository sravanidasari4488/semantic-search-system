"""Search engine combining embedding pipeline, vector store, and dataset."""

from typing import Any

import pandas as pd

from src.embedding_pipeline import EmbeddingPipeline
from src.semantic_cache import SemanticCache
from src.vector_store import VectorStore


class SemanticSearchEngine:
    """Orchestrates semantic search over indexed documents."""

    def __init__(
        self,
        embedding_pipeline: EmbeddingPipeline,
        vector_store: VectorStore,
        dataset_dataframe: pd.DataFrame,
        semantic_cache: SemanticCache | None = None,
    ) -> None:
        """Initialize the search engine.

        Args:
            embedding_pipeline: Pipeline for embedding queries.
            vector_store: FAISS index with document embeddings.
            dataset_dataframe: DataFrame with document_id, category, text.
                Row order must match the order of embeddings in the vector store.
            semantic_cache: Optional cache for semantically similar queries.
        """
        self.embedding_pipeline = embedding_pipeline
        self.vector_store = vector_store
        self.dataset = dataset_dataframe
        self.semantic_cache = semantic_cache

    def _do_vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Perform vector search and return result list."""
        query_embedding = self.embedding_pipeline.generate_embeddings([query])[0]
        indices, scores = self.vector_store.search(query_embedding, top_k=top_k)

        results = []
        for idx, score in zip(indices, scores, strict=True):
            if idx < 0 or idx >= len(self.dataset):
                continue
            row = self.dataset.iloc[idx]
            results.append(
                {
                    "doc_id": str(row["document_id"]),
                    "text": str(row["text"]),
                    "score": float(score),
                    "index": int(idx),
                }
            )
        return results

    def search(
        self, query: str, top_k: int = 5, use_cache: bool = True
    ) -> dict[str, Any]:
        """Run semantic search for a query.

        Flow: check cache -> if hit return cached; if miss do vector search,
        store in cache, return result. Response includes cache metadata.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            use_cache: Whether to use semantic cache (if configured).

        Returns:
            Dict with results (list of {doc_id, text, score}), cache_hit,
            matched_query, similarity_score.
        """
        metadata: dict[str, Any] = {
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": None,
        }

        if use_cache and self.semantic_cache is not None:
            cached = self.semantic_cache.get(query)
            if cached["result"] is not None:
                metadata["cache_hit"] = True
                metadata["matched_query"] = cached["matched_query"]
                metadata["similarity_score"] = cached["similarity_score"]
                return {"results": cached["result"], **metadata}

        results = self._do_vector_search(query, top_k)

        if use_cache and self.semantic_cache is not None:
            self.semantic_cache.set(query, results)

        return {"results": results, **metadata}
