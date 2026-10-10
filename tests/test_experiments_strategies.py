"""Tests for the measured retrieval-strategy comparison.

The rule these guard is the honesty rule: the numbers must come from actually running each
retriever over the same synthetic dataset, the comparison must be deterministic and re-runnable,
a delta must equal the two values it claims to relate, and the tool must refuse to compare a
strategy it does not have or against a baseline it was not asked to run. No test here asserts a
particular winner — the point is that hybrid need not beat lexical, and must be reported either way
— only that the measurement is real, consistent, and correctly attributed.
"""

from __future__ import annotations

import pytest

from chakaso.benchmark import load_development_benchmark
from chakaso.experiments import (
    InvalidExperimentError,
    compare_retrieval_strategies,
    retrieval_strategy_experiment,
    run_experiment,
)


def test_compare_reports_every_strategy_in_order_with_deltas() -> None:
    corpus, dataset = load_development_benchmark()

    report = compare_retrieval_strategies(corpus, dataset, top_k=5)

    assert [row.strategy for row in report.results] == ["lexical", "dense", "hybrid"]
    assert report.baseline == "lexical"
    assert {delta.strategy for delta in report.deltas} == {"dense", "hybrid"}
    for row in report.results:
        assert 0.0 <= row.recall_at_k <= 1.0
        assert 0.0 <= row.hit_rate <= 1.0
    for delta in report.deltas:
        assert abs((delta.candidate - delta.baseline) - delta.delta) < 1e-12


def test_compare_is_deterministic_across_runs() -> None:
    corpus, dataset = load_development_benchmark()

    first = compare_retrieval_strategies(corpus, dataset)
    second = compare_retrieval_strategies(corpus, dataset)

    assert first.to_dict() == second.to_dict()


def test_compare_accepts_a_smaller_strategy_set() -> None:
    corpus, dataset = load_development_benchmark()

    report = compare_retrieval_strategies(
        corpus, dataset, strategies=("lexical", "hybrid"), baseline="lexical"
    )

    assert [row.strategy for row in report.results] == ["lexical", "hybrid"]
    assert {delta.strategy for delta in report.deltas} == {"hybrid"}


def test_unknown_strategy_is_rejected() -> None:
    corpus, dataset = load_development_benchmark()

    with pytest.raises(ValueError, match="unknown retrieval strategy"):
        compare_retrieval_strategies(corpus, dataset, strategies=("teleport",))


def test_baseline_must_be_one_of_the_compared_strategies() -> None:
    corpus, dataset = load_development_benchmark()

    with pytest.raises(ValueError, match="baseline"):
        compare_retrieval_strategies(
            corpus, dataset, strategies=("dense", "hybrid"), baseline="lexical"
        )


def test_a_strategy_experiment_runs_and_records_measured_metrics() -> None:
    result = run_experiment(retrieval_strategy_experiment("dense", top_k=3))

    assert result.evaluators == ("dense",)
    assert result.configuration["retriever"] == "dense"
    assert result.configuration["top_k"] == "3"
    assert result.metrics["recall_at_k"] is not None


def test_strategy_is_part_of_the_result_fingerprint() -> None:
    # Two strategies over the same dataset answer the same question differently, so they must not
    # share a result_id — the configuration difference is real, not noise.
    lexical = run_experiment(retrieval_strategy_experiment("lexical"))
    dense = run_experiment(retrieval_strategy_experiment("dense"))

    assert lexical.result_id != dense.result_id


def test_unknown_strategy_experiment_is_rejected() -> None:
    with pytest.raises(InvalidExperimentError, match="unknown retrieval strategy"):
        retrieval_strategy_experiment("quantum")
