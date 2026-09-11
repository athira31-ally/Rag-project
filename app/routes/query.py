from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_pipeline
from app.models import QueryRequest, QueryResponse
from app.rag_pipeline import RagPipeline

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> QueryResponse:
    return pipeline.query(
        question=payload.question,
        workspace_id=payload.workspace_id,
        top_k=payload.top_k,
    )
