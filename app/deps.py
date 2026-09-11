"""FastAPI dependency wiring, kept separate from main.py so tests can
override get_pipeline() without importing the ASGI app module.
"""
from __future__ import annotations

from functools import lru_cache

from app.rag_pipeline import RagPipeline


@lru_cache
def get_pipeline() -> RagPipeline:
    return RagPipeline()
