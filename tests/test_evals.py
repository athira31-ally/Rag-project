"""Runs the eval suite as part of the regular test run so a regression in
retrieval quality shows up in `pytest`, not just when someone remembers to
run scripts/run_evals.py by hand."""
from pathlib import Path

from scripts.run_evals import run


def test_eval_suite_passes():
    suite_path = Path(__file__).resolve().parent.parent / "scripts" / "evals" / "cases.yaml"
    assert run(suite_path) is True
