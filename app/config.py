"""Centralized, typed application settings.

Everything is read from the environment (see .env.example). Nothing here
defaults to a real endpoint or credential -- the defaults are chosen so the
app runs fully offline out of the box (local Qdrant storage, hashing
embeddings, no LLM key required, extractive answers).
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Vector store ---
    qdrant_url: str | None = None
    qdrant_collection: str = "documents"
    qdrant_local_path: str = "./local_qdrant_data"
    qdrant_hnsw_m: int = 32
    qdrant_hnsw_ef_construct: int = 256
    qdrant_search_ef: int = 128

    # --- Embeddings ---
    embedding_provider: str = "hashing"  # hashing | openai | sentence-transformers
    embedding_dim: int = 384
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"

    # --- LLM answer generation ---
    llm_provider: str = "none"  # none | openai
    openai_chat_model: str = "gpt-4o-mini"

    # --- Redis cache ---
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300
    cache_lock_timeout_seconds: int = 10

    # --- Celery ---
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # --- Retrieval ---
    default_top_k: int = 5
    chunk_size_tokens: int = 300
    chunk_overlap_tokens: int = 50


settings = Settings()
