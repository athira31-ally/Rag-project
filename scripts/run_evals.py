#!/usr/bin/env python3
"""A small, self-contained eval runner in the spirit of promptfoo: load a
YAML suite of {question, expected assertions}, run it against the live
RagPipeline, and report a pass/fail table. Exits non-zero if anything
fails, so it's usable as a CI gate.

Usage:
    python scripts/run_evals.py [path/to/cases.yaml]
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.rag_pipeline import RagPipeline  # noqa: E402
from app.tracing import NoOpTracer  # noqa: E402


def load_suite(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def run(suite_path: Path) -> bool:
    suite = load_suite(suite_path)
    workspace_id = suite.get("workspace_id", "eval")

    pipeline = RagPipeline(tracer=NoOpTracer())

    # Use a small chunk size for the eval fixtures so each distinct claim in
    # a fixture document (e.g. the enterprise-contract carve-out at the end
    # of the refund policy) becomes its own retrievable, citable chunk
    # instead of being buried inside one large one. Restored afterwards so
    # running the suite doesn't change chunking behavior for anything else
    # in the same process (e.g. when pytest imports this module).
    original_size, original_overlap = settings.chunk_size_tokens, settings.chunk_overlap_tokens
    settings.chunk_size_tokens, settings.chunk_overlap_tokens = 40, 8
    try:
        for fixture in suite.get("fixtures", []):
            text = (suite_path.parent.parent.parent / fixture["file"]).read_text()
            pipeline.ingest(
                document_id=str(uuid.uuid4()),
                title=fixture["title"],
                workspace_id=workspace_id,
                text=text,
            )
    finally:
        settings.chunk_size_tokens, settings.chunk_overlap_tokens = original_size, original_overlap

    results = []
    for case in suite["cases"]:
        result = pipeline.query(question=case["question"], workspace_id=workspace_id)
        failures = []

        expected_title = case.get("expect_top_citation_title")
        if expected_title:
            top_title = result.citations[0].title if result.citations else None
            if top_title != expected_title:
                failures.append(f"expected top citation '{expected_title}', got '{top_title}'")

        for phrase in case.get("expect_answer_contains", []):
            if phrase.lower() not in result.answer.lower():
                failures.append(f"expected answer to contain '{phrase}'")

        results.append((case["name"], failures))

    print(f"\n{'CASE':<25} {'RESULT':<8} DETAILS")
    print("-" * 70)
    all_passed = True
    for name, failures in results:
        if failures:
            all_passed = False
            print(f"{name:<25} {'FAIL':<8} {'; '.join(failures)}")
        else:
            print(f"{name:<25} {'PASS':<8}")

    passed = sum(1 for _, f in results if not f)
    print("-" * 70)
    print(f"{passed}/{len(results)} passed\n")
    return all_passed


if __name__ == "__main__":
    suite_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "evals" / "cases.yaml"
    ok = run(suite_path)
    sys.exit(0 if ok else 1)
