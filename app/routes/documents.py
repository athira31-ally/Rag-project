from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Form, UploadFile

from app.deps import get_pipeline
from app.models import DocumentIn, DocumentStatusOut, IngestionStatus
from app.ocr import get_ocr_pipeline
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


@router.post("/ocr", response_model=DocumentStatusOut)
async def ingest_scanned_document(
    file: UploadFile,
    title: str = Form(...),
    workspace_id: str = Form(default="default"),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> DocumentStatusOut:
    """Ingests a scanned image: OCR first, then the same chunk/embed/index
    path as a plain-text document."""
    image_bytes = await file.read()
    text = get_ocr_pipeline().extract_text(image_bytes)

    document_id = str(uuid.uuid4())
    if not text:
        return DocumentStatusOut(
            document_id=document_id,
            title=title,
            status=IngestionStatus.FAILED,
            chunk_count=0,
            error="OCR extracted no text from the uploaded image",
        )

    chunk_count = pipeline.ingest(document_id=document_id, title=title, workspace_id=workspace_id, text=text)
    return DocumentStatusOut(
        document_id=document_id, title=title, status=IngestionStatus.COMPLETED, chunk_count=chunk_count
    )
