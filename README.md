# Talkument RAG (portfolio rebuild)

A from-scratch, original implementation of a RAG-based document Q&A
service — the same architecture and problems I worked on building a
production system (multi-tenant retrieval on Qdrant, deployed on EKS behind
a Kong gateway, with async ingestion and a citation system), rebuilt here
independently with generic sample data and no proprietary code or client
information from that work.

## What it does

1. **Ingest** — text (or a scanned image, via OCR) is chunked with overlap,
   embedded, and upserted into Qdrant with tenant (`workspace_id`) and
   document metadata as indexed payload fields.
2. **Query** — the question (typed or spoken) is embedded, Qdrant returns
   the nearest chunks (scoped to the caller's workspace via a server-side
   filter), and the response includes an answer, its citations, and —
   for the voice route — a synthesized audio reply.
3. **Auth** — Google OAuth2 login issues a JWT session; a `me`/dev-token
   flow makes local development and tests possible without real Google
   credentials.
4. **Observability** — every ingest/query call runs inside a trace span
   (console-logged by default, pluggable to Langfuse), and a small
   promptfoo-style YAML eval suite checks retrieval/answer quality.

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

**OCR with a fallback chain.** `app/ocr.py` runs a real local OCR engine
(Tesseract, via `pytesseract` — no API key, no network) as the default, and
falls back to a cloud vision model only when the local pass returns
suspiciously little text (a proxy for "this scan is too poor to read
locally"). `tests/test_ocr.py` renders text into a synthetic image and
verifies Tesseract actually reads it back — a real integration test, not a
mock.

**Voice Q&A with fully offline defaults.** Speech-to-text and text-to-speech
are pluggable the same way embeddings are: the default `stub` STT/TTS
providers make `/voice/query` genuinely exercisable end to end (upload →
transcript → RAG query → synthesized WAV reply) without a model download or
an API key, and swapping in Whisper/OpenAI TTS is a config change.

**Auth kept testable, not just "real."** `app/auth.py` implements a real
Google OAuth2 flow, but defaults to a deterministic fake provider so `/auth`
routes, JWT issuance, and the `get_current_user` dependency are covered by
tests without live Google credentials.

**Tracing and evals as part of the pipeline, not bolted on after.**
`RagPipeline.ingest`/`query` wrap each step in a trace span
(`app/tracing.py`); `scripts/run_evals.py` runs a small YAML suite of
question → expected-citation/answer assertions against the live pipeline
and is itself covered by `tests/test_evals.py`, so a retrieval-quality
regression shows up in `pytest`, not just when someone remembers to run it
by hand.

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
  ocr.py                OCR pipeline with local + cloud-fallback providers
  voice.py              speech-to-text / text-to-speech providers
  auth.py               Google OAuth2 + JWT session handling
  tracing.py            pluggable tracer (console / in-memory / Langfuse)
  rag_pipeline.py       ties the above together for ingest/query, traced
  celery_app.py, tasks.py   async ingestion
  routes/               FastAPI route handlers (documents, query, voice, auth, health)
tests/                pytest suite — chunking, embeddings, cache, OCR
                       (real Tesseract), voice, auth, tracing, evals, and a
                       full ingest->query API roundtrip — all run offline
scripts/
  run_evals.py          promptfoo-style eval runner (also run by pytest)
  evals/cases.yaml       eval suite: questions + expected citations/answers
sample_data/          generic example documents for the quickstart
docker/               Dockerfile (incl. tesseract-ocr) + docker-compose
                       (api, worker, qdrant, redis)
k8s/                  illustrative EKS manifests (Deployment/Service/HPA)
                       and a Kong route/rate-limit example
DEPLOYMENT.md         build/push/deploy walkthrough for this app on EKS + Kong
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

## Trying the new routes

```bash
# Auth: get a dev session token (no real Google credentials needed locally)
TOKEN=$(curl -s -X POST localhost:8000/auth/dev-token | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl localhost:8000/auth/me -H "Authorization: Bearer $TOKEN"

# OCR: ingest a scanned image
curl -X POST localhost:8000/documents/ocr \
  -F "file=@scan.png" -F "title=Scanned Note" -F "workspace_id=demo"

# Voice: upload audio (or, with the default stub STT, a text file standing
# in for a transcript) and get back an answer plus a synthesized WAV reply
curl -X POST localhost:8000/voice/query \
  -F "audio=@question.wav" -F "workspace_id=demo"
```

## Evals

```bash
python scripts/run_evals.py
```

Runs the suite in `scripts/evals/cases.yaml` against the live pipeline and
prints a pass/fail table — useful as a CI gate for retrieval-quality
regressions.

## Tests

```bash
pytest -q
```

27 tests, all offline — chunking edge cases, embedding determinism, a full
ingest → query roundtrip against a real (local) Qdrant instance, workspace
isolation, cache-stampede protection under concurrent load, real OCR via
Tesseract, voice provider behavior, auth/JWT flow, tracing spans, the eval
suite, and the FastAPI routes end to end.

See `DEPLOYMENT.md` for a walkthrough of building the image, deploying to
EKS, and putting it behind Kong.

