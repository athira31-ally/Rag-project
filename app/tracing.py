"""Lightweight, pluggable tracing for pipeline calls.

Every ingest/query call is wrapped in a span so latency and inputs/outputs
are inspectable without an external dependency by default. Swapping in a
real observability backend (Langfuse here) is a config change, not a code
change -- the same pattern used for embeddings/LLM/OCR/voice providers
throughout this repo.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from app.config import settings

logger = logging.getLogger("talkument_rag.trace")


@dataclass
class Span:
    name: str
    metadata: dict = field(default_factory=dict)
    duration_ms: float | None = None
    error: str | None = None


class Tracer(Protocol):
    @contextmanager
    def span(self, name: str, **metadata): ...


class NoOpTracer:
    @contextmanager
    def span(self, name: str, **metadata):
        yield Span(name=name, metadata=metadata)


class ConsoleTracer:
    """Logs a structured line per span. Good enough default for local dev
    and for demonstrating the instrumentation without any external service."""

    @contextmanager
    def span(self, name: str, **metadata):
        span = Span(name=name, metadata=metadata)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.error = str(exc)
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                "trace span=%s duration_ms=%s metadata=%s error=%s",
                span.name,
                span.duration_ms,
                span.metadata,
                span.error,
            )


class InMemoryTracer:
    """Records spans in a list instead of logging -- used by tests to assert
    on what got traced without capturing stdout."""

    def __init__(self):
        self.spans: list[Span] = []

    @contextmanager
    def span(self, name: str, **metadata):
        span = Span(name=name, metadata=metadata)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.error = str(exc)
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.spans.append(span)


class LangfuseTracer:
    """Optional real backend -- only imported if selected."""

    def __init__(self, public_key: str, secret_key: str, host: str):
        from langfuse import Langfuse  # optional dependency

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    @contextmanager
    def span(self, name: str, **metadata):
        trace = self._client.trace(name=name, metadata=metadata)
        span = Span(name=name, metadata=metadata)
        start = time.perf_counter()
        try:
            yield span
        except Exception as exc:
            span.error = str(exc)
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            trace.update(output={"duration_ms": span.duration_ms, "error": span.error})


def get_tracer() -> Tracer:
    provider = settings.tracer_provider
    if provider == "none":
        return NoOpTracer()
    if provider == "console":
        return ConsoleTracer()
    if provider == "langfuse":
        return LangfuseTracer(
            public_key=settings.langfuse_public_key or "",
            secret_key=settings.langfuse_secret_key or "",
            host=settings.langfuse_host,
        )
    raise ValueError(f"Unknown TRACER_PROVIDER: {provider!r}")
