"""Answer generation, with a graceful, dependency-free fallback.

LLM_PROVIDER=none (the default) returns an extractive answer built directly
from the top retrieved chunk -- no API key, no network call, fully
deterministic. Set LLM_PROVIDER=openai to generate a real synthesized answer
from an OpenAI/Azure-OpenAI-compatible chat model instead.
"""
from __future__ import annotations

from typing import Protocol

from app.config import settings
from app.models import Citation


class AnswerGenerator(Protocol):
    def generate(self, question: str, citations: list[Citation]) -> tuple[str, bool]: ...


class ExtractiveAnswerGenerator:
    def generate(self, question: str, citations: list[Citation]) -> tuple[str, bool]:
        if not citations:
            return "I couldn't find anything relevant to that question in the indexed documents.", False
        top = citations[0]
        return f"Based on “{top.title}”: {top.snippet}", False


class OpenAIAnswerGenerator:
    def __init__(self, model: str, api_key: str | None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the 'openai' LLM provider")
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, citations: list[Citation]) -> tuple[str, bool]:
        context = "\n\n".join(f"[{i+1}] {c.title}: {c.snippet}" for i, c in enumerate(citations))
        prompt = (
            "Answer the question using only the numbered context below. "
            "Cite sources inline like [1]. If the answer isn't in the context, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or "", True


def get_answer_generator() -> AnswerGenerator:
    if settings.llm_provider == "none":
        return ExtractiveAnswerGenerator()
    if settings.llm_provider == "openai":
        return OpenAIAnswerGenerator(model=settings.openai_chat_model, api_key=settings.openai_api_key)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
