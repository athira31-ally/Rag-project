# Talkument RAG (portfolio rebuild)

A from-scratch, original implementation of a RAG-based document Q&A
service — the same architecture and problems I worked on building a
production system (multi-tenant retrieval on Qdrant, deployed on EKS behind
a Kong gateway, with async ingestion and a citation system), rebuilt here
independently with generic sample data and no proprietary code or client
information from that work.

## What it does

1. **Ingest** — text is chunked with overlap, embedded, and upserted into
   Qdrant with tenant (`workspace_id`) and document metadata as indexed
   payload fields.
2. **Query** — the question is embedded, Qdrant returns the nearest chunks
   (scoped to the caller's workspace via a server-side filter), and the
   response includes both an answer and the citations it came from.

```
        POST /documents                      POST /query
             │                                     │
             ▼                                     ▼
      chunk_text()                          embed(question)
             │                                     │
             ▼                                     ▼
   embed(chunk texts)                    VectorStore.search()
             │                             (Qdrant, HNSW, filtered
             ▼                              by workspace_id)
   VectorStore.upsert_chunks()                     │
     (Qdrant, payload-indexed)                     ▼
                                            to_citations()
                                                    │
                                                    ▼
                                          AnswerGenerator.generate()
                                       (extractive, or LLM if configured)
```

## Design notes

**Pluggable embeddings, not a hard dependency on one API.** `EMBEDDING_PROVIDER`
switches between a zero-dependency deterministic hashing provider (default —
runs fully offline, which is what the test suite and local quickstart use),
an OpenAI/Azure-OpenAI-compatible provider, and a local `sentence-transformers`
provider. The interface is the same either way, so swapping models for a
real deployment is a one-line config change, not a code change.

**HNSW tuning is exposed, not buried.** `QDRANT_HNSW_M` / `QDRANT_HNSW_EF_CONSTRUCT`
control index build quality; `QDRANT_SEARCH_EF` controls search-time recall
independently, so it can be tuned without rebuilding the index. Multi-tenant
filtering happens server-side via an indexed `workspace_id` payload field
rather than over-fetching and filtering in application code.

**Cache-stampede-safe Redis caching.** `app/cache.py` implements the
standard fix for a real production failure mode: several API replicas
missing the same cache key at once and all recomputing (and potentially
racing to write back) simultaneously. A short-lived distributed lock
ensures only one caller recomputes; the rest wait briefly for the result or
fall back to computing locally if the lock holder stalls, so one slow
replica can't wedge the others. `tests/test_cache.py` verifies the expensive
compute path runs exactly once under concurrent load.

**Idempotent async ingestion.** `app/tasks.py` runs ingestion as a Celery
task keyed by `document_id`; Qdrant's upsert-by-id semantics mean a retried
or duplicated task delivery re-writes the same chunks rather than creating
duplicates — the same class of bug as a duplicated streamed response.

**Graceful degradation for answer generation.** With no LLM configured, the
API still returns a real answer (extractive, from the top citation) instead
of failing — useful for local dev and for keeping citations testable without
needing an API key.

## Project layout

```
app/
  main.py            FastAPI app + router registration
  config.py           typed settings (env-driven, see .env.example)
  chunking.py          overlapping token-window chunking
  embeddings.py        pluggable embedding providers
  vectorstore.py       Qdrant collection lifecycle, upsert, filtered search
  citations.py          maps Qdrant hits -> Citation objects
  cache.py              Redis cache with stampede protection
  llm.py                pluggable answer generation (extractive or LLM)
  rag_pipeline.py       ties the above together for ingest/query
  celery_app.py, tasks.py   async ingestion
  routes/               FastAPI route handlers
tests/                pytest suite (chunking, embeddings, cache, full
                       ingest->query roundtrip, API tests) — all run
                       offline, no external services needed
sample_data/          generic example documents for the quickstart
docker/               Dockerfile + docker-compose (api, worker, qdrant, redis)
k8s/                  illustrative EKS manifests (Deployment/Service/HPA)
                       and a Kong route/rate-limit example
```

## Quickstart (local, no external services)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --reload
```

In another shell:

```bash
curl -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{"title": "Refund Policy", "text": "Refunds are issued within five business days.", "workspace_id": "demo"}'

curl -X POST localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"question": "How long do refunds take?", "workspace_id": "demo"}'
```

By default this runs entirely offline: Qdrant in local on-disk mode, the
hashing embedding provider, and extractive answers. To use a real vector
DB/model/LLM, copy `.env.example` to `.env` and fill in `QDRANT_URL`,
`EMBEDDING_PROVIDER=openai` (or `sentence-transformers`), and
`LLM_PROVIDER=openai`.

## Running with Docker Compose (API + worker + Qdrant + Redis)

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Tests

```bash
pytest -q
```


