"""Tests for the experiment model and runner.

The reproducibility promise is the thing to pin: a result's fingerprint must ignore the timestamp
and the host, must move when the configuration, benchmark or metrics move, and must be stable for
the same experiment on the same code. The runner is checked to actually drive the existing
development benchmarks (so the numbers are real measurements of the rule, not fixtures of the test)
and to record the evaluator and dataset identity honestly.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from chakaso.experiments import (
    BenchmarkKind,
    Experiment,
    ExperimentId,
    ExperimentResult,
    InvalidExperimentError,
    run_experiment,
)

T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 8, 13, 0, tzinfo=UTC)


def _correction_experiment() -> Experiment:
    return Experiment(
        experiment_id=ExperimentId("corr-baseline"),
        name="correction rule baseline",
        hypothesis="the decision rule reaches every expected correction outcome",
        benchmark_kind=BenchmarkKind.CORRECTION,
        configuration={"grounding": "structural+fixture-semantic"},
    )


# --- identity / validation -------------------------------------------------


def test_experiment_id_must_be_a_slug() -> None:
    assert str(ExperimentId("corr-baseline")) == "corr-baseline"
    with pytest.raises(InvalidExperimentError):
        ExperimentId("Not A Slug")


def test_an_experiment_must_state_a_hypothesis() -> None:
    with pytest.raises(InvalidExperimentError, match="hypothesis"):
        Experiment(
            experiment_id=ExperimentId("x"),
            name="n",
            hypothesis="   ",
            benchmark_kind=BenchmarkKind.RETRIEVAL,
        )


def test_configuration_is_normalised_to_an_immutable_mapping() -> None:
    experiment = replace(_correction_experiment(), configuration={"a": "1", "b": "2"})
    assert experiment.configuration["a"] == "1"
    with pytest.raises(TypeError):
        experiment.configuration["a"] = "changed"  # type: ignore[index]


def test_a_naive_result_timestamp_is_refused() -> None:
    result = run_experiment(_correction_experiment(), now=lambda: T0)
    with pytest.raises(InvalidExperimentError, match="timezone-aware"):
        replace(result, recorded_at=datetime(2026, 10, 8))


# --- runner ---------------------------------------------------------------


def test_running_the_correction_experiment_measures_the_real_rule() -> None:
    result = run_experiment(_correction_experiment(), now=lambda: T0)

    assert isinstance(result, ExperimentResult)
    assert result.dataset_id == "dev-correction"
    assert result.metrics["decision_accuracy"] == 1.0
    assert set(result.evaluators) == {"structural-0.1", "fixture-semantic-0.1"}


def test_running_the_retrieval_experiment_measures_the_retriever() -> None:
    experiment = Experiment(
        experiment_id=ExperimentId("ret-baseline"),
        name="retrieval baseline",
        hypothesis="lexical retrieval is measurable",
        benchmark_kind=BenchmarkKind.RETRIEVAL,
        configuration={"top_k": "5"},
    )

    result = run_experiment(experiment, now=lambda: T0)

    assert result.dataset_id == "dev-retrieval"
    assert result.evaluators == ("lexical",)
    assert result.metrics["recall_at_k"] is not None


# --- reproducibility ------------------------------------------------------


def test_result_id_ignores_timestamp_and_environment() -> None:
    at_noon = run_experiment(_correction_experiment(), now=lambda: T0, environment={"host": "a"})
    at_one = run_experiment(_correction_experiment(), now=lambda: T1, environment={"host": "b"})

    assert at_noon.recorded_at != at_one.recorded_at
    assert at_noon.environment["host"] != at_one.environment["host"]
    assert at_noon.result_id == at_one.result_id


def test_result_id_moves_when_a_relevant_input_moves() -> None:
    base = run_experiment(_correction_experiment(), now=lambda: T0)

    reconfigured = run_experiment(
        replace(_correction_experiment(), configuration={"grounding": "structural"}),
        now=lambda: T0,
    )
    assert reconfigured.result_id != base.result_id

    metrics_changed = replace(base, metrics={**base.metrics, "decision_accuracy": 0.5})
    assert metrics_changed.result_id != base.result_id


def test_result_to_dict_is_complete_and_separates_environment() -> None:
    result = run_experiment(_correction_experiment(), now=lambda: T0, environment={"host": "a"})
    payload = result.to_dict()

    assert payload["result_id"] == result.result_id
    assert payload["experiment"]["hypothesis"] == _correction_experiment().hypothesis
    assert payload["benchmark"]["dataset_id"] == "dev-correction"
    assert payload["metrics"]["decision_accuracy"] == 1.0
    assert payload["environment"] == {"host": "a"}
    assert payload["recorded_at"] == T0.isoformat()
