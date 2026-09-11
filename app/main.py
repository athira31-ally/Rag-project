from __future__ import annotations

from fastapi import FastAPI

from app.routes import documents, health, query

app = FastAPI(
    title="Talkument RAG (portfolio rebuild)",
    description="A from-scratch RAG document Q&A service demonstrating the "
    "same architecture as the original Talkument project: chunking, "
    "pluggable embeddings, Qdrant retrieval with HNSW tuning, citation "
    "mapping, cache-stampede-safe Redis caching, and async ingestion.",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
