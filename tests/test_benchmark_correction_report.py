"""Tests for the correction benchmark report and a fingerprint regression.

The report must be deterministic, must label itself a development instrument that scores a decision
rule (not a capability), and must expose per-case expected-vs-actual so a failure is inspectable.
The pinned dataset fingerprint is the same regression guard the retrieval corpus uses: an
unintended edit to any correction case moves it and fails this test on purpose.
"""

from __future__ import annotations

import json
from dataclasses import replace

from chakaso.benchmark.correction import development_correction_benchmark
from chakaso.benchmark.correction_report import (
    DEVELOPMENT_CORRECTION_NOTE,
    correction_report_json,
    correction_report_to_dict,
    format_correction_report,
)
from chakaso.benchmark.correction_runner import CorrectionBenchmarkRunner
from chakaso.correction.decision import CorrectionDecision

# The curated correction dataset's fingerprint. Changing a case's judgement must be a deliberate,
# reviewed version bump — this constant is where that review is forced.
PINNED_CORRECTION_FINGERPRINT = "0693127d305a5ad6"


def _run():
    return CorrectionBenchmarkRunner().run(development_correction_benchmark())


def test_development_dataset_fingerprint_is_pinned() -> None:
    assert development_correction_benchmark().content_fingerprint == PINNED_CORRECTION_FINGERPRINT


def test_report_json_is_deterministic_and_labels_itself_development() -> None:
    first = correction_report_json(_run())
    second = correction_report_json(_run())

    assert first == second
    payload = json.loads(first)
    assert payload["benchmark"]["kind"] == "development"
    assert payload["benchmark"]["subject"] == "correction-decision-rule"
    assert payload["benchmark"]["note"] == DEVELOPMENT_CORRECTION_NOTE
    assert payload["metrics"]["decision_accuracy"] == 1.0
    assert len(payload["cases"]) == 6


def test_report_preserves_per_case_expected_and_actual() -> None:
    payload = correction_report_to_dict(_run())
    by_id = {case["case_id"]: case for case in payload["cases"]}

    contradiction = by_id["correct-contradiction"]
    assert contradiction["expected_decision"] == "correct"
    assert contradiction["actual_decision"] == "correct"
    assert contradiction["passed"] is True
    assert contradiction["evaluator"] == "fixture-semantic-0.1"


def test_text_report_shows_pass_and_the_development_note() -> None:
    text = format_correction_report(_run())

    assert "Correction benchmark report" in text
    assert DEVELOPMENT_CORRECTION_NOTE in text
    assert "[PASS] retain-supported" in text


def test_a_failing_case_is_visible_in_the_report() -> None:
    dataset = development_correction_benchmark()
    broken = replace(
        dataset,
        cases=(
            replace(dataset.cases[0], expected_decision=CorrectionDecision.CORRECT),
            *dataset.cases[1:],
        ),
    )
    text = format_correction_report(CorrectionBenchmarkRunner().run(broken))

    assert "[FAIL] retain-supported" in text
