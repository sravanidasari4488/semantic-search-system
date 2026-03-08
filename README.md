# Semantic Search System

## Project Overview

A Python-based semantic search system for the 20 Newsgroups dataset. It combines embedding-based retrieval, vector similarity search, fuzzy clustering, and semantic caching. The system is served via FastAPI and runs locally without external services.

**Key components:**

- **Embedding Pipeline** – Sentence Transformers (all-MiniLM-L6-v2) for text embeddings
- **Vector Store** – FAISS (IndexFlatIP) for similarity search
- **Fuzzy Clustering** – Gaussian Mixture Model for document grouping
- **Semantic Cache** – Reuses results for semantically similar queries

---

## System Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  FastAPI API    │
│  POST /query    │
└──────┬──────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Embedding Model                │
│  (Sentence Transformers MiniLM) │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────┐
│ FAISS Vector    │
│ Search          │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Fuzzy Clustering│
│ (GMM)           │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Semantic Cache  │
│ (hit/miss)      │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Return Results  │
│ (preview+score) │
└─────────────────┘
```

---

## How to Run Locally

### Prerequisites

- Python 3.10+
- 20 Newsgroups dataset at `data/20_newsgroups/20_newsgroups/`

### Setup

```bash
python -m venv venv

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

### Start the API

```bash
python -m uvicorn api.main:app --reload
```

On first run, embeddings and the FAISS index are generated and cached to disk. Subsequent starts load from cache for faster startup.

API docs: http://127.0.0.1:8000/docs

---

## How to Run with Docker

### Build

```bash
docker build -t semantic-search-api .
```

### Run

```bash
docker run -p 8000:8000 -v /path/to/data:/app/data semantic-search-api
```

Mount your local `data/` directory (with `20_newsgroups/20_newsgroups/`) so the container can load the dataset and use cached embeddings/FAISS index if present.

---

## API Endpoints

| Method | Endpoint        | Description                    |
|--------|-----------------|--------------------------------|
| GET    | /health         | Health check                   |
| POST   | /query          | Semantic search (JSON body: `{"query": "..."}`) |
| GET    | /cache/stats    | Cache hit/miss and size stats  |
| DELETE | /cache          | Clear the semantic cache       |

---
## Example Query

Request:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{"query": "space shuttle launch"}'

## Project Structure

```
semantic-search-system/
├── data/                  # Data files and cached artifacts
├── src/
│   ├── data_loader.py     # Document loading and preprocessing
│   ├── embedding_pipeline.py  # Embedding generation
│   ├── vector_store.py    # FAISS vector storage
│   ├── clustering.py      # Fuzzy clustering (GMM)
│   ├── semantic_cache.py  # Semantic query caching
│   └── search_engine.py   # Search orchestration
├── api/
│   └── main.py            # FastAPI service
├── notebooks/
│   └── cluster_analysis.ipynb  # Cluster justification, boundary cases, threshold exploration
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```
Note: On the first run, the system generates embeddings and builds the FAISS index for the dataset (~20k documents). This may take several minutes. The embeddings and index are cached to disk, so subsequent startups are much faster.