"""Tests for the deterministic regression baseline comparison.

The comparison must (a) agree two identical runs changed nothing, (b) call a worse metric a
regression and a better one an improvement according to each metric's declared direction, (c) refuse
to compare results that were defined differently (different configuration, benchmark, evaluator or
experiment) rather than produce a misleading delta, and (d) stay deterministic. No statistics —
these are small synthetic benchmarks and a delta is all this claims to show.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from chakaso.experiments import (
    BenchmarkKind,
    ComparisonDirection,
    Experiment,
    ExperimentId,
    IncompatibleExperimentError,
    compare_results,
    run_experiment,
)

T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 8, 13, 0, tzinfo=UTC)


def _experiment(**overrides: object) -> Experiment:
    base: dict[str, object] = {
        "experiment_id": ExperimentId("corr-baseline"),
        "name": "correction rule baseline",
        "hypothesis": "the decision rule reaches every expected outcome",
        "benchmark_kind": BenchmarkKind.CORRECTION,
        "configuration": {"grounding": "structural+fixture-semantic"},
    }
    base.update(overrides)
    return Experiment(**base)  # type: ignore[arg-type]


def test_identical_runs_show_no_regression_or_improvement() -> None:
    baseline = run_experiment(_experiment(), implementation_version="v1", now=lambda: T0)
    candidate = run_experiment(_experiment(), implementation_version="v2", now=lambda: T1)

    comparison = compare_results(baseline, candidate)  # type: ignore[arg-type]

    assert comparison.regressions == ()
    assert comparison.improvements == ()
    assert all(c.delta == 0.0 for c in comparison.comparisons if c.baseline is not None)


def test_a_worse_higher_is_better_metric_is_a_regression() -> None:
    baseline = run_experiment(_experiment(), now=lambda: T0)
    worse = replace(baseline, metrics={**baseline.metrics, "decision_accuracy": 0.5})

    comparison = compare_results(baseline, worse)

    assert "decision_accuracy" in comparison.regressions


def test_a_worse_lower_is_better_metric_is_a_regression() -> None:
    baseline = run_experiment(_experiment(), now=lambda: T0)
    worse = replace(baseline, metrics={**baseline.metrics, "unnecessary_revision_rate": 0.5})

    comparison = compare_results(baseline, worse)

    assert "unnecessary_revision_rate" in comparison.regressions


def test_a_better_retrieval_metric_is_an_improvement() -> None:
    experiment = _experiment(
        experiment_id=ExperimentId("ret-baseline"),
        name="retrieval",
        hypothesis="lexical retrieval is measurable",
        benchmark_kind=BenchmarkKind.RETRIEVAL,
        configuration={"top_k": "5"},
    )
    baseline = run_experiment(experiment, now=lambda: T0)
    better = replace(
        baseline,
        metrics={**baseline.metrics, "recall_at_k": baseline.metrics["recall_at_k"] + 0.1},
    )

    comparison = compare_results(baseline, better)

    assert "recall_at_k" in comparison.improvements


def test_a_missing_metric_is_not_comparable_not_zero() -> None:
    baseline = run_experiment(_experiment(), now=lambda: T0)
    partial = replace(baseline, metrics={**baseline.metrics, "appropriate_review_rate": None})

    comparison = compare_results(baseline, partial)
    by_metric = {c.metric: c.direction for c in comparison.comparisons}

    assert by_metric["appropriate_review_rate"] is ComparisonDirection.NOT_COMPARABLE


def test_incompatible_definitions_are_refused() -> None:
    baseline = run_experiment(_experiment(), now=lambda: T0)

    reconfigured = replace(baseline, configuration={"grounding": "structural"})
    with pytest.raises(IncompatibleExperimentError, match="configuration"):
        compare_results(baseline, reconfigured)

    other_experiment = replace(baseline, experiment_id=ExperimentId("different"))
    with pytest.raises(IncompatibleExperimentError, match="different experiments"):
        compare_results(baseline, other_experiment)

    other_benchmark = replace(baseline, dataset_content_fingerprint="0" * 16)
    with pytest.raises(IncompatibleExperimentError, match="dataset_content_fingerprint"):
        compare_results(baseline, other_benchmark)


def test_comparison_is_deterministic_and_serializable() -> None:
    baseline = run_experiment(_experiment(), now=lambda: T0)
    candidate = run_experiment(_experiment(), now=lambda: T0)

    first = compare_results(baseline, candidate)
    second = compare_results(baseline, candidate)

    assert first == second
    payload = first.to_dict()
    assert payload["experiment_id"] == "corr-baseline"
    assert payload["regressions"] == []
