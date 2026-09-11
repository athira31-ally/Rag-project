"""Async ingestion task.

Idempotency note: the task is keyed by document_id, and re-running it for
the same id simply re-upserts the same chunk set (Qdrant upsert is
overwrite-by-id), so a retried/duplicated task delivery -- the classic
at-least-once Celery failure mode -- doesn't create duplicate chunks or a
duplicated streamed response downstream.
"""
from __future__ import annotations

from app.celery_app import celery_app
from app.rag_pipeline import RagPipeline

_pipeline: RagPipeline | None = None


def _get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


@celery_app.task(name="app.tasks.ingest_document", bind=True, max_retries=3, default_retry_delay=5)
def ingest_document(self, document_id: str, title: str, workspace_id: str, text: str) -> dict:
    try:
        chunk_count = _get_pipeline().ingest(
            document_id=document_id, title=title, workspace_id=workspace_id, text=text
        )
        return {"document_id": document_id, "chunk_count": chunk_count, "status": "completed"}
    except Exception as exc:  # pragma: no cover - retry path
        raise self.retry(exc=exc)
