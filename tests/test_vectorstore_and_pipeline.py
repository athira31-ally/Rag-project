"""Exercises the full ingest -> query loop against a real (local, on-disk)
Qdrant instance and the hashing embedding provider -- no network, no API
keys, no external services required.
"""
import shutil
import uuid

import pytest

from app.embeddings import HashingEmbeddingProvider
from app.llm import ExtractiveAnswerGenerator
from app.rag_pipeline import RagPipeline
from app.vectorstore import VectorStore


@pytest.fixture
def pipeline(tmp_path):
    # Each test gets its own on-disk Qdrant storage folder -- the local mode
    # file-locks its storage path, so sharing one across tests/processes
    # would collide.
    monkeypatch_path = str(tmp_path / "qdrant")
    from qdrant_client import QdrantClient

    embedder = HashingEmbeddingProvider(dim=64)
    store = VectorStore(dim=64, client=QdrantClient(path=monkeypatch_path))
    store.collection = f"test_{uuid.uuid4().hex}"
    store._ensure_collection()
    yield RagPipeline(embedder=embedder, store=store, generator=ExtractiveAnswerGenerator(), cache=None)


def test_ingest_then_query_returns_relevant_citation(pipeline):
    pipeline.ingest(
        document_id="doc-1",
        title="Refund Policy",
        workspace_id="acme",
        text="Refunds are issued within five business days of a return request. "
        "Store credit is available immediately at checkout.",
    )
    pipeline.ingest(
        document_id="doc-2",
        title="Shipping Policy",
        workspace_id="acme",
        text="Standard shipping takes seven to ten business days across the region.",
    )

    result = pipeline.query(question="How long do refunds take?", workspace_id="acme", top_k=2)

    assert result.citations, "expected at least one citation"
    assert any(c.title == "Refund Policy" for c in result.citations)
    assert result.used_llm is False


def test_query_is_scoped_to_workspace(pipeline):
    pipeline.ingest(document_id="a", title="Tenant A Doc", workspace_id="tenant-a", text="secret tenant A content")
    pipeline.ingest(document_id="b", title="Tenant B Doc", workspace_id="tenant-b", text="secret tenant B content")

    result = pipeline.query(question="secret content", workspace_id="tenant-a", top_k=5)

    assert all(c.title == "Tenant A Doc" for c in result.citations)
