from app.tracing import InMemoryTracer, NoOpTracer


def test_in_memory_tracer_records_span_with_duration():
    tracer = InMemoryTracer()

    with tracer.span("unit.test", foo="bar"):
        pass

    assert len(tracer.spans) == 1
    span = tracer.spans[0]
    assert span.name == "unit.test"
    assert span.metadata == {"foo": "bar"}
    assert span.duration_ms is not None
    assert span.error is None


def test_in_memory_tracer_records_error():
    tracer = InMemoryTracer()

    try:
        with tracer.span("unit.failing"):
            raise ValueError("boom")
    except ValueError:
        pass

    assert tracer.spans[0].error == "boom"


def test_pipeline_ingest_and_query_are_traced():
    from app.embeddings import HashingEmbeddingProvider
    from app.llm import ExtractiveAnswerGenerator
    from app.rag_pipeline import RagPipeline
    from app.vectorstore import VectorStore

    tracer = InMemoryTracer()
    store = VectorStore(dim=32)
    store.collection = "tracing_test"
    store._ensure_collection()
    pipeline = RagPipeline(
        embedder=HashingEmbeddingProvider(dim=32),
        store=store,
        generator=ExtractiveAnswerGenerator(),
        cache=None,
        tracer=tracer,
    )

    pipeline.ingest(document_id="d1", title="Doc", workspace_id="ws", text="some content about refunds")
    pipeline.query(question="refunds?", workspace_id="ws")

    span_names = [s.name for s in tracer.spans]
    assert "rag.ingest" in span_names
    assert "rag.query" in span_names
    assert "rag.retrieve" in span_names
    assert "rag.generate" in span_names


def test_noop_tracer_is_a_no_op():
    tracer = NoOpTracer()
    with tracer.span("whatever"):
        pass  # should not raise
