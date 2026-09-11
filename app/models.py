"""Pydantic schemas shared across the API and pipeline layers."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentIn(BaseModel):
    """Payload for submitting raw text for ingestion (upload route wraps this)."""

    title: str = Field(..., min_length=1, max_length=256)
    text: str = Field(..., min_length=1)
    workspace_id: str = Field(
        default="default",
        description="Tenant / workspace scope used for payload filtering in Qdrant.",
    )


class DocumentStatusOut(BaseModel):
    document_id: str
    title: str
    status: IngestionStatus
    chunk_count: int | None = None
    error: str | None = None


class Citation(BaseModel):
    document_id: str
    title: str
    chunk_id: str
    snippet: str
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    workspace_id: str = "default"
    top_k: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    used_llm: bool
