"""Text chunking for ingestion.

Splits on whitespace-token boundaries with configurable size/overlap so
neighbouring chunks share context -- the same trade-off that matters for
RAG recall: too small loses context, too large hurts embedding precision
and citation granularity.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    start_token: int
    end_token: int


def chunk_text(text: str, chunk_size_tokens: int = 300, overlap_tokens: int = 50) -> list[Chunk]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = chunk_size_tokens - overlap_tokens
    index = 0
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(
            Chunk(
                index=index,
                text=" ".join(chunk_tokens),
                start_token=start,
                end_token=end,
            )
        )
        index += 1
        if end == len(tokens):
            break
        start += step

    return chunks
