from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends

from app.deps import get_pipeline
from app.models import DocumentIn, DocumentStatusOut, IngestionStatus
from app.rag_pipeline import RagPipeline

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentStatusOut)
def ingest_document(payload: DocumentIn, pipeline: RagPipeline = Depends(get_pipeline)) -> DocumentStatusOut:
    """Synchronous ingestion for the demo/local flow.

    In a deployment with Celery running, swap this for enqueuing
    app.tasks.ingest_document.delay(...) and returning PENDING immediately --
    the pipeline logic itself doesn't change, only who calls it.
    """
    document_id = str(uuid.uuid4())
    chunk_count = pipeline.ingest(
        document_id=document_id,
        title=payload.title,
        workspace_id=payload.workspace_id,
        text=payload.text,
    )
    return DocumentStatusOut(
        document_id=document_id,
        title=payload.title,
        status=IngestionStatus.COMPLETED,
        chunk_count=chunk_count,
    )
