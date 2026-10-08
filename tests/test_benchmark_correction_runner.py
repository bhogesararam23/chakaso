"""Tests for the correction benchmark runner.

The runner is where the correction decision rule is exercised against a labelled set, so the
central assertion is that the curated cases reach the decisions the rule is supposed to reach — and,
just as importantly, that the benchmark *can fail*: corrupting a case's expectation must turn a pass
into a failure, or the green result would prove nothing (Section 47). Determinism and the
unknown-fixture-label guard are checked too, because a benchmark that drifts or silently accepts a
bad reference is not a measurement.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from chakaso.benchmark.correction import CorrectionCase, development_correction_benchmark
from chakaso.benchmark.correction_runner import CorrectionBenchmarkRunner
from chakaso.benchmark.identity import CaseId
from chakaso.correction.decision import CorrectionDecision


def test_the_rule_reaches_every_curated_case_s_expected_decision() -> None:
    run = CorrectionBenchmarkRunner().run(development_correction_benchmark())

    assert run.passed is True
    assert run.metrics.decision_accuracy == 1.0
    assert run.metrics.unjustified_persistence_rate == 0.0
    assert run.metrics.unnecessary_revision_rate == 0.0
    assert run.metrics.appropriate_abstention_rate == 1.0
    assert run.metrics.appropriate_review_rate == 1.0


def test_a_corrupted_expectation_makes_the_benchmark_fail() -> None:
    dataset = development_correction_benchmark()
    # Flip the first case's expectation to something the rule does not produce. If the benchmark
    # still reported success, it would be measuring nothing.
    broken = replace(
        dataset,
        cases=(
            replace(dataset.cases[0], expected_decision=CorrectionDecision.CORRECT),
            *dataset.cases[1:],
        ),
    )

    run = CorrectionBenchmarkRunner().run(broken)

    assert run.passed is False
    assert run.outcomes[0].passed is False
    assert run.metrics.decision_accuracy is not None
    assert run.metrics.decision_accuracy < 1.0


def test_runs_are_deterministic() -> None:
    runner = CorrectionBenchmarkRunner()
    dataset = development_correction_benchmark()

    first = runner.run(dataset)
    second = runner.run(dataset)

    assert first.outcomes == second.outcomes
    assert first.dataset_content_fingerprint == second.dataset_content_fingerprint


def test_outcomes_report_the_evaluator_that_decided() -> None:
    run = CorrectionBenchmarkRunner().run(development_correction_benchmark())
    by_id = {str(outcome.case_id): outcome for outcome in run.outcomes}

    assert by_id["correct-contradiction"].evaluator_name == "fixture-semantic-0.1"
    assert by_id["qualify-lost-support"].evaluator_name == "structural-0.1"


def test_an_unknown_fixture_label_fails_loudly() -> None:
    case = CorrectionCase(
        case_id=CaseId("bogus"),
        description="names evidence that does not exist",
        claims=development_correction_benchmark().cases[0].claims,
        reassessment_labels=("this-label-does-not-exist",),
        expected_decision=CorrectionDecision.RETAIN,
    )

    with pytest.raises(LookupError, match="unknown fixture label"):
        CorrectionBenchmarkRunner().run(replace(development_correction_benchmark(), cases=(case,)))
