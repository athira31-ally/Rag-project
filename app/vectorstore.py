"""Qdrant wrapper: collection lifecycle, batched upserts, filtered search.

Design choices worth calling out (these are the knobs that actually move
recall/latency in production, not just defaults left untouched):

* HNSW `m` / `ef_construct` are exposed via settings rather than hardcoded --
  higher values trade index build time and memory for recall.
* `ef` (search-time) is set per query, independent of the build-time
  parameter, so it can be tuned without rebuilding the index.
* workspace_id is stored as an indexed payload field so multi-tenant
  filtering happens inside Qdrant (server-side), not by over-fetching and
  filtering in Python.
"""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.config import settings


class VectorStore:
    def __init__(self, dim: int, client: QdrantClient | None = None):
        self.dim = dim
        self.collection = settings.qdrant_collection
        self.client = client or self._build_client()
        self._ensure_collection()

    @staticmethod
    def _build_client() -> QdrantClient:
        if settings.qdrant_url:
            return QdrantClient(url=settings.qdrant_url)
        # Local, on-disk mode: no server required. Good for tests/demos;
        # a real deployment sets QDRANT_URL to point at the EKS-hosted cluster.
        return QdrantClient(path=settings.qdrant_local_path)

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.dim,
                distance=qm.Distance.COSINE,
                hnsw_config=qm.HnswConfigDiff(
                    m=settings.qdrant_hnsw_m,
                    ef_construct=settings.qdrant_hnsw_ef_construct,
                ),
            ),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="workspace_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="document_id",
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    def upsert_chunks(
        self,
        document_id: str,
        title: str,
        workspace_id: str,
        chunk_texts: list[str],
        vectors: list[list[float]],
    ) -> list[str]:
        chunk_ids = [str(uuid.uuid4()) for _ in chunk_texts]
        points = [
            qm.PointStruct(
                id=chunk_id,
                vector=vector,
                payload={
                    "document_id": document_id,
                    "title": title,
                    "workspace_id": workspace_id,
                    "chunk_index": i,
                    "text": text,
                },
            )
            for i, (chunk_id, text, vector) in enumerate(zip(chunk_ids, chunk_texts, vectors))
        ]
        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return chunk_ids

    def search(
        self,
        query_vector: list[float],
        workspace_id: str,
        top_k: int = 5,
    ) -> list[qm.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="workspace_id", match=qm.MatchValue(value=workspace_id))]
            ),
            limit=top_k,
            search_params=qm.SearchParams(hnsw_ef=settings.qdrant_search_ef),
        )
