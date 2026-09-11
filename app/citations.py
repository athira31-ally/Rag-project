"""Maps retrieved Qdrant hits into citation objects returned to the caller.

Kept as its own module because citation formatting is a product decision
(how much snippet context, how scores are surfaced) independent of retrieval
mechanics -- easy to evolve without touching the vector store or pipeline.
"""
from __future__ import annotations

from app.models import Citation

_SNIPPET_MAX_CHARS = 320


def to_citations(hits) -> list[Citation]:
    citations = []
    for hit in hits:
        text = hit.payload.get("text", "")
        snippet = text if len(text) <= _SNIPPET_MAX_CHARS else text[: _SNIPPET_MAX_CHARS - 1] + "…"
        citations.append(
            Citation(
                document_id=hit.payload["document_id"],
                title=hit.payload["title"],
                chunk_id=str(hit.id),
                snippet=snippet,
                score=round(float(hit.score), 4),
            )
        )
    return citations
