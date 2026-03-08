"""FastAPI application for semantic search service."""

import logging
from pathlib import Path

from fastapi import FastAPI

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

from src.clustering import FuzzyClusterer
from src.data_loader import load_documents
from src.embedding_pipeline import EmbeddingPipeline
from src.semantic_cache import SemanticCache
from src.search_engine import SemanticSearchEngine
from src.vector_store import VectorStore


# --- Pydantic models ---


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    query: str = Field(..., description="Search query text")


PREVIEW_LENGTH = 200


class SearchHit(BaseModel):
    """Single search result document (preview only; full text not sent)."""

    doc_id: str
    preview: str = Field(..., description="First 200 chars of document")
    score: float


class QueryResponse(BaseModel):
    """Response for POST /query."""

    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str = Field(..., description="Top result preview")
    results: list[SearchHit] = Field(..., description="All search result documents")
    dominant_cluster: int | None = Field(
        ..., description="Dominant cluster of top result"
    )


class CacheStatsResponse(BaseModel):
    """Response for GET /cache/stats."""

    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


# --- App ---

app = FastAPI(title="Semantic Search API", version="0.1.0")

# Initialized on startup
_search_engine: SemanticSearchEngine | None = None
_index_to_dominant_cluster: dict[int, int] | None = None


def _get_dominant_cluster(top_result_index: int | None) -> int | None:
    """Look up dominant cluster for the top result by index."""
    if _index_to_dominant_cluster is None or top_result_index is None:
        return None
    return _index_to_dominant_cluster.get(top_result_index)


@app.on_event("startup")
def startup() -> None:
    """Load data, build index, and initialize search engine."""
    global _search_engine, _index_to_dominant_cluster

    logger.info("Starting Semantic Search API...")

    data_path = Path("data/20_newsgroups/20_newsgroups")
    if not data_path.exists():
        logger.warning("Dataset path not found, skipping initialization.")
        return

    df = load_documents(str(data_path))
    if df.empty:
        logger.warning("No documents loaded, skipping initialization.")
        return

    pipeline = EmbeddingPipeline()
    embeddings_path = Path("data/embeddings.npy")

    if embeddings_path.exists():
        logger.info("Loading embeddings from disk...")
        embeddings = pipeline.load_embeddings(embeddings_path)
    else:
        logger.info("Generating embeddings (this may take a few minutes)...")
        texts = df["text"].tolist()
        embeddings = pipeline.generate_embeddings(texts)
        pipeline.save_embeddings(embeddings_path, embeddings)
        logger.info("Embeddings saved to disk.")

    vector_store = VectorStore()
    index_path = Path("data/faiss.index")

    if index_path.exists():
        logger.info("Loading FAISS index...")
        vector_store.load_index(index_path)
    else:
        logger.info("Building FAISS index...")
        vector_store.build_index(embeddings)
        vector_store.save_index(index_path)
        logger.info("FAISS index saved to disk.")

    logger.info("Fitting clustering model...")
    clusterer = FuzzyClusterer(n_clusters=20)
    clusterer.fit(embeddings)
    cluster_results = clusterer.cluster_documents(embeddings)
    _index_to_dominant_cluster = {
        i: r["dominant_cluster"] for i, r in enumerate(cluster_results)
    }

    cache = SemanticCache(
        embedding_pipeline=pipeline,
        clusterer=clusterer,
    )
    _search_engine = SemanticSearchEngine(
        embedding_pipeline=pipeline,
        vector_store=vector_store,
        dataset_dataframe=df,
        semantic_cache=cache,
    )

    logger.info("System ready.")


@app.get("/health")
def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Semantic search for a query. Uses cache when available."""
    if _search_engine is None:
        return QueryResponse(
            query=req.query,
            cache_hit=False,
            matched_query=None,
            similarity_score=None,
            result="",
            results=[],
            dominant_cluster=None,
        )

    out = _search_engine.search(req.query, top_k=5, use_cache=True)

    top_index: int | None = None
    if out["results"]:
        top_index = out["results"][0].get("index")

    hits = [
        SearchHit(
            doc_id=r["doc_id"],
            preview=r["text"][:PREVIEW_LENGTH],
            score=r["score"],
        )
        for r in out["results"]
    ]

    top_preview = hits[0].preview if hits else ""
    return QueryResponse(
        query=req.query,
        cache_hit=out["cache_hit"],
        matched_query=out["matched_query"],
        similarity_score=out["similarity_score"],
        result=top_preview,
        results=hits,
        dominant_cluster=_get_dominant_cluster(top_index),
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats() -> CacheStatsResponse:
    """Return cache hit/miss and size statistics."""
    if _search_engine is None or _search_engine.semantic_cache is None:
        return CacheStatsResponse(
            total_entries=0, hit_count=0, miss_count=0, hit_rate=0.0
        )
    c = _search_engine.semantic_cache
    total_requests = c.hit_count + c.miss_count
    hit_rate = c.hit_count / total_requests if total_requests > 0 else 0.0
    return CacheStatsResponse(
        total_entries=c.total_entries,
        hit_count=c.hit_count,
        miss_count=c.miss_count,
        hit_rate=round(hit_rate, 3),
    )


@app.delete("/cache")
def clear_cache() -> dict:
    """Clear the semantic cache."""
    if _search_engine is None or _search_engine.semantic_cache is None:
        return {"status": "ok", "message": "No cache configured"}

    _search_engine.semantic_cache.clear()
    return {"status": "ok", "message": "Cache cleared"}
