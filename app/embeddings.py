"""Pluggable embedding backends.

A production deployment swaps in a real model (OpenAI/Azure OpenAI-compatible,
or a local sentence-transformers model) via EMBEDDING_PROVIDER. The default,
zero-dependency "hashing" provider makes the whole app runnable offline with
no API key and no model download -- useful for local dev, tests, and demos --
while keeping the same interface a real embedding call would use.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from app.config import settings

# A small, fixed stopword list -- no external NLP dependency. Skipping these
# before hashing matters a lot for a bag-of-words-style embedding: without
# it, high-frequency function words ("to", "the", "of") can dominate the
# similarity signal over the low-frequency content words ("refund",
# "shipping") that actually distinguish one document from another.
_STOPWORDS = frozenset(
    """
    a an the of to in on for at by from with as is are was were be been being
    this that these those it its it's and or but if then than so not no nor
    do does did doing have has had having will would shall should can could
    may might must i you he she we they what which who whom my your his her
    our their about into over under again further once here there when where
    why how all any both each few more most other some such only own same
    just don't should've now
    """.split()
)


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbeddingProvider:
    """Deterministic, dependency-free pseudo-embedding.

    Hashes overlapping word shingles into a fixed-size vector and L2-normalizes
    it. Not semantically meaningful like a real model, but deterministic,
    fast, and good enough to exercise the full retrieval pipeline (chunking,
    Qdrant indexing, HNSW search, citation mapping) without any external
    dependency -- which is what the test suite and the local quickstart run
    against by default.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        words = text.lower().split()
        content_words = [w for w in words if w not in _STOPWORDS] or words
        shingles = content_words if content_words else [text.lower()]
        for shingle in shingles:
            digest = hashlib.sha256(shingle.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class OpenAIEmbeddingProvider:
    """OpenAI/Azure-OpenAI-compatible embeddings (any base_url override works)."""

    def __init__(self, model: str, dim: int, api_key: str | None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the 'openai' embedding provider")
        from openai import OpenAI  # imported lazily so it's an optional dependency at runtime

        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


def get_embedding_provider() -> EmbeddingProvider:
    provider = settings.embedding_provider
    if provider == "hashing":
        return HashingEmbeddingProvider(dim=settings.embedding_dim)
    if provider == "openai":
        return OpenAIEmbeddingProvider(
            model=settings.openai_embedding_model,
            dim=settings.embedding_dim,
            api_key=settings.openai_api_key,
        )
    if provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer  # optional dependency

        model = SentenceTransformer("all-MiniLM-L6-v2")

        class _STProvider:
            dim = model.get_sentence_embedding_dimension()

            def embed(self, texts: list[str]) -> list[list[float]]:
                return model.encode(texts, normalize_embeddings=True).tolist()

        return _STProvider()

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")
