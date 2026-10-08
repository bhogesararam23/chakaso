"""Tests for benchmark reports and a pinned-fingerprint regression.

A report must carry its own provenance and admit what it is not (development, no
significance claims), must be deterministic, and its machine form must round-trip as JSON.
The pinned content fingerprint is a regression gate: if a fixture or a case judgement
changes, the fingerprint moves and this test fails until the change is deliberate and the
dataset version is bumped.
"""

from __future__ import annotations

import json

from chakaso.benchmark import (
    DEVELOPMENT_BENCHMARK_NOTE,
    RetrievalBenchmarkRunner,
    build_corpus,
    development_benchmark,
    format_report,
    load_development_benchmark,
    report_json,
    report_to_dict,
)
from chakaso.retrieval import RetrievalService

# Pinned deliberately. Changing fixtures or case judgements changes this; the response is
# to bump the dataset version and re-pin, not to silently edit the constant.
DEVELOPMENT_FINGERPRINT = "36e65ab20468d141"


def _run():
    corpus, dataset = load_development_benchmark()
    return RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=5).run(dataset)


def test_report_is_labelled_a_development_benchmark() -> None:
    text = format_report(_run())

    assert DEVELOPMENT_BENCHMARK_NOTE in text
    assert report_to_dict(_run())["benchmark"]["kind"] == "development"  # type: ignore[index]
    assert DEVELOPMENT_FINGERPRINT in text


def test_report_carries_provenance_and_metrics() -> None:
    payload = report_to_dict(_run())

    assert payload["benchmark"]["dataset_id"] == "dev-retrieval"  # type: ignore[index]
    assert payload["benchmark"]["dataset_version"] == "1.0.0"  # type: ignore[index]
    assert payload["configuration"]["top_k"] == 5  # type: ignore[index]
    assert set(payload["metrics"]) >= {"recall_at_k", "precision_at_k", "mrr"}  # type: ignore[arg-type]


def test_json_report_round_trips_and_is_deterministic() -> None:
    run = _run()
    first = report_json(run)
    second = report_json(
        RetrievalBenchmarkRunner(RetrievalService(build_corpus()), top_k=5).run(
            development_benchmark(build_corpus())
        )
    )

    assert json.loads(first)["benchmark"]["kind"] == "development"
    assert first == second


def test_development_benchmark_fingerprint_is_pinned() -> None:
    assert development_benchmark(build_corpus()).content_fingerprint == DEVELOPMENT_FINGERPRINT


def test_human_report_lists_every_case() -> None:
    run = _run()
    text = format_report(run)

    for outcome in run.outcomes:
        assert str(outcome.case_id) in text
