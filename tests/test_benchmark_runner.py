"""Tests for the retrieval benchmark runner.

A focused two-document scenario fixes the metric arithmetic precisely; a run over the whole
development benchmark checks the things that must hold regardless of BM25's ordering: that
provenance is recorded, scored and abstention cases are separated, every gold id actually
exists in the corpus, and two identical runs are byte-for-byte equal (determinism, Section
46)."""

from __future__ import annotations

from chakaso.benchmark import (
    BenchmarkCategory,
    CaseId,
    DatasetId,
    RetrievalBenchmarkRunner,
    build_corpus,
    dataset_from_rows,
    development_benchmark,
    load_development_benchmark,
)
from chakaso.retrieval import Corpus, RetrievalService, ingest_text

CAP_GOLD_TEXT = "The capital of the realm is Paris."
BANANA_TEXT = "A banana is a yellow fruit."


def _focus_service_and_dataset():
    corpus = Corpus()
    cap = ingest_text(CAP_GOLD_TEXT, label="capital")
    banana = ingest_text(BANANA_TEXT, label="banana")
    corpus.add_document(cap)
    corpus.add_document(banana)
    dataset = dataset_from_rows(
        [
            {
                "case_id": "find-capital",
                "category": "direct-retrieval",
                "query": "what is the capital",
                "gold_chunk_ids": [str(cap.chunks[0].chunk_id)],
                "forbidden_chunk_ids": [str(banana.chunks[0].chunk_id)],
            },
            {
                "case_id": "no-evidence",
                "category": "missing-evidence",
                "query": "venus mars quasar distance",
            },
        ],
        dataset_id="focus",
        version="1.0.0",
    )
    return RetrievalService(corpus), dataset, cap


def test_a_direct_case_is_a_hit_with_full_recall() -> None:
    service, dataset, cap = _focus_service_and_dataset()
    run = RetrievalBenchmarkRunner(service, top_k=5).run(dataset)

    outcome = next(o for o in run.outcomes if o.case_id == CaseId("find-capital"))
    assert outcome.hit is True
    assert outcome.recall_at_k == 1.0
    assert str(cap.chunks[0].chunk_id) in outcome.retrieved_ids


def test_a_no_gold_case_is_an_abstention_not_a_zero_recall() -> None:
    service, dataset, _ = _focus_service_and_dataset()
    run = RetrievalBenchmarkRunner(service, top_k=5).run(dataset)

    outcome = next(o for o in run.outcomes if o.case_id == CaseId("no-evidence"))
    assert outcome.has_gold is False
    assert outcome.abstained is True
    # Scored and abstention cases are kept distinct: one of each here.
    assert run.metrics.scored_cases == 1
    assert run.metrics.abstention_cases == 1


def test_forbidden_evidence_hit_is_reported() -> None:
    banana_only = dataset_from_rows(
        [
            {
                "case_id": "banana-is-wrong",
                "category": "distractor",
                "query": "yellow fruit",
                "forbidden_chunk_ids": [
                    str(ingest_text(BANANA_TEXT, label="banana").chunks[0].chunk_id)
                ],
                "gold_chunk_ids": [],
            }
        ],
        dataset_id="focus",
        version="1.0.0",
    )
    # Give the distractor case a gold id so it is scored, and forbid the only matching chunk.
    corpus = Corpus()
    corpus.add_document(ingest_text(BANANA_TEXT, label="banana"))
    corpus.add_document(ingest_text("Something about a rocket launch.", label="other"))
    run = RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=5).run(banana_only)

    assert run.metrics.forbidden_hits == 1


def test_run_records_dataset_provenance() -> None:
    service, dataset, _ = _focus_service_and_dataset()
    run = RetrievalBenchmarkRunner(service, top_k=3, retriever_name="lexical").run(dataset)

    assert run.dataset_id == "focus"
    assert run.dataset_version == "1.0.0"
    assert run.top_k == 3
    assert run.retriever_name == "lexical"
    assert len(run.dataset_content_fingerprint) == 16


def test_two_runs_are_identical() -> None:
    corpus = build_corpus()
    dataset = development_benchmark(corpus)
    runner = RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=5)

    first = runner.run(dataset)
    second = runner.run(dataset)

    assert [(o.case_id, o.retrieved_ids, o.recall_at_k) for o in first.outcomes] == [
        (o.case_id, o.retrieved_ids, o.recall_at_k) for o in second.outcomes
    ]
    assert first.metrics == second.metrics


def test_every_scored_gold_exists_in_the_corpus() -> None:
    corpus, dataset = load_development_benchmark()
    available = {str(chunk.chunk_id) for chunk in corpus.chunks}
    run = RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=5).run(dataset)

    for outcome in run.outcomes:
        assert set(outcome.gold_ids) <= available
        assert set(outcome.forbidden_ids) <= available


def test_development_benchmark_separates_scored_from_abstention() -> None:
    corpus, dataset = load_development_benchmark()
    run = RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=5).run(dataset)

    assert run.metrics.total_cases == len(dataset)
    assert run.metrics.scored_cases == sum(1 for c in dataset if c.gold_chunk_ids)
    assert run.metrics.abstention_cases == sum(1 for c in dataset if not c.gold_chunk_ids)


def test_category_is_carried_on_the_outcome() -> None:
    corpus, dataset = load_development_benchmark()
    run = RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=5).run(dataset)

    distractor = next(o for o in run.outcomes if o.case_id == CaseId("distractor-parental-leave"))
    assert distractor.category is BenchmarkCategory.DISTRACTOR


def test_dataset_id_and_version_reach_the_run_unchanged() -> None:
    dataset = dataset_from_rows(
        [
            {
                "case_id": "x",
                "category": "direct-retrieval",
                "query": "q",
            }
        ],
        dataset_id=str(DatasetId("named")),
        version="2.1.0",
    )

    run = RetrievalBenchmarkRunner(RetrievalService(Corpus())).run(dataset)

    assert run.dataset_id == "named"
    assert run.dataset_version == "2.1.0"
