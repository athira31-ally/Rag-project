"""Ties chunking + embeddings + vector search + citations + generation
together into the two operations the API exposes: ingest and query.
"""
from __future__ import annotations

from app.cache import Cache
from app.chunking import chunk_text
from app.citations import to_citations
from app.config import settings
from app.embeddings import EmbeddingProvider, get_embedding_provider
from app.llm import AnswerGenerator, get_answer_generator
from app.models import Citation, QueryResponse
from app.vectorstore import VectorStore


class RagPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        generator: AnswerGenerator | None = None,
        cache: Cache | None = None,
    ):
        self.embedder = embedder or get_embedding_provider()
        self.store = store or VectorStore(dim=self.embedder.dim)
        self.generator = generator or get_answer_generator()
        self.cache = cache

    def ingest(self, document_id: str, title: str, workspace_id: str, text: str) -> int:
        chunks = chunk_text(
            text,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.upsert_chunks(
            document_id=document_id,
            title=title,
            workspace_id=workspace_id,
            chunk_texts=[c.text for c in chunks],
            vectors=vectors,
        )
        return len(chunks)

    def query(self, question: str, workspace_id: str, top_k: int | None = None) -> QueryResponse:
        k = top_k or settings.default_top_k

        def _search() -> list[Citation]:
            [vector] = self.embedder.embed([question])
            hits = self.store.search(query_vector=vector, workspace_id=workspace_id, top_k=k)
            return to_citations(hits)

        if self.cache is not None:
            cache_key = f"query:{workspace_id}:{k}:{hash(question)}"
            citations = self.cache.get_or_set(
                cache_key,
                _search,
                serialize=lambda cs: __import__("json").dumps([c.model_dump() for c in cs]),
                deserialize=lambda s: [Citation(**c) for c in __import__("json").loads(s)],
            )
        else:
            citations = _search()

        answer, used_llm = self.generator.generate(question, citations)
        return QueryResponse(answer=answer, citations=citations, used_llm=used_llm)
