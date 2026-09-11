import tempfile
import uuid

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from app.deps import get_pipeline
from app.embeddings import HashingEmbeddingProvider
from app.llm import ExtractiveAnswerGenerator
from app.main import app
from app.rag_pipeline import RagPipeline
from app.vectorstore import VectorStore


def _make_test_pipeline() -> RagPipeline:
    local_path = tempfile.mkdtemp(prefix="qdrant_api_test_")
    store = VectorStore(dim=64, client=QdrantClient(path=local_path))
    store.collection = f"api_test_{uuid.uuid4().hex}"
    store._ensure_collection()
    return RagPipeline(
        embedder=HashingEmbeddingProvider(dim=64),
        store=store,
        generator=ExtractiveAnswerGenerator(),
        cache=None,
    )


_test_pipeline = _make_test_pipeline()
app.dependency_overrides[get_pipeline] = lambda: _test_pipeline
client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ingest_and_query_roundtrip():
    ingest_resp = client.post(
        "/documents",
        json={"title": "Vacation Policy", "text": "Employees accrue two days of leave per month.", "workspace_id": "hr"},
    )
    assert ingest_resp.status_code == 200
    body = ingest_resp.json()
    assert body["status"] == "completed"
    assert body["chunk_count"] >= 1

    query_resp = client.post("/query", json={"question": "How much leave do employees accrue?", "workspace_id": "hr"})
    assert query_resp.status_code == 200
    result = query_resp.json()
    assert result["citations"]
    assert result["citations"][0]["title"] == "Vacation Policy"
