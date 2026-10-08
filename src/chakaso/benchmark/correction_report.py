"""Rendering a correction benchmark run, for a person and for a machine.

Same discipline as the retrieval report (``report.py``): every report carries the dataset identity,
version and content fingerprint and the evaluator, and states in its own words what it is — here, a
development instrument that scores a *decision rule* over synthetic cases, runs no model, and
produces no answer. The JSON is deterministic (sorted keys, fixed separators) so the same run always
yields the same bytes, and metrics whose denominator had no cases serialise as ``null`` rather than
a fabricated number.
"""

from __future__ import annotations

import json
from typing import Final

from chakaso.benchmark.correction_runner import CorrectionBenchmarkRun, CorrectionCaseOutcome

__all__ = [
    "DEVELOPMENT_CORRECTION_NOTE",
    "correction_report_json",
    "correction_report_to_dict",
    "format_correction_report",
]

#: Stated in every correction report so a perfect score cannot be read as a capability claim.
DEVELOPMENT_CORRECTION_NOTE: Final[str] = (
    "Development benchmark of the correction decision rule over a small, synthetic, "
    "hand-built case set. It scores a rule against a caller's expected decision, runs no "
    "language model, produces no revised answer, and is far too small to support any claim "
    "about real-world correction quality. Contradictions and uncertainties in these cases are "
    "supplied by a fixture judge, never inferred."
)


def correction_report_to_dict(run: CorrectionBenchmarkRun) -> dict[str, object]:
    """Convert a run into a deterministic, JSON-serializable mapping."""
    metrics = run.metrics
    return {
        "benchmark": {
            "kind": "development",
            "subject": "correction-decision-rule",
            "dataset_id": run.dataset_id,
            "dataset_version": run.dataset_version,
            "dataset_content_fingerprint": run.dataset_content_fingerprint,
            "note": DEVELOPMENT_CORRECTION_NOTE,
        },
        "configuration": {"runner": "correction", **run.metadata},
        "metrics": {
            "total_cases": metrics.total_cases,
            "passed_cases": metrics.passed_cases,
            "decision_accuracy": metrics.decision_accuracy,
            "unjustified_persistence_rate": metrics.unjustified_persistence_rate,
            "unnecessary_revision_rate": metrics.unnecessary_revision_rate,
            "appropriate_abstention_rate": metrics.appropriate_abstention_rate,
            "appropriate_review_rate": metrics.appropriate_review_rate,
            "expected_by_decision": {
                decision.value: count
                for decision, count in sorted(
                    metrics.expected_by_decision.items(), key=lambda item: item[0].value
                )
            },
            "correct_by_decision": {
                decision.value: count
                for decision, count in sorted(
                    metrics.correct_by_decision.items(), key=lambda item: item[0].value
                )
            },
        },
        "cases": [_case_dict(outcome) for outcome in run.outcomes],
    }


def _case_dict(outcome: CorrectionCaseOutcome) -> dict[str, object]:
    return {
        "case_id": str(outcome.case_id),
        "case_version": outcome.case_version,
        "description": outcome.description,
        "expected_decision": outcome.expected_decision.value,
        "actual_decision": outcome.actual_decision.value,
        "passed": outcome.passed,
        "evaluator": outcome.evaluator_name,
        "reassessment_labels": list(outcome.reassessment_labels),
        "affected_claim_ids": list(outcome.affected_claim_ids),
    }


def correction_report_json(run: CorrectionBenchmarkRun) -> str:
    """Render the run as deterministic JSON."""
    return json.dumps(correction_report_to_dict(run), sort_keys=True, indent=2, ensure_ascii=False)


def format_correction_report(run: CorrectionBenchmarkRun) -> str:
    """Render the run as human-readable text."""
    lines = [
        f"Correction benchmark report — {run.dataset_id} v{run.dataset_version}",
        f"content fingerprint: {run.dataset_content_fingerprint}",
        DEVELOPMENT_CORRECTION_NOTE,
        "",
        _format_metrics(run),
        "",
        "Per case:",
    ]
    for outcome in run.outcomes:
        verdict = "PASS" if outcome.passed else "FAIL"
        lines.append(
            f"  [{verdict}] {outcome.case_id} ({outcome.evaluator_name}): "
            f"expected={outcome.expected_decision.value} actual={outcome.actual_decision.value}"
        )
    return "\n".join(lines)


def _format_metrics(run: CorrectionBenchmarkRun) -> str:
    metrics = run.metrics
    return "\n".join(
        [
            f"Cases: {metrics.total_cases}   Passed: {metrics.passed_cases}",
            f"Decision accuracy: {_fmt(metrics.decision_accuracy)}",
            f"Unjustified persistence: {_fmt(metrics.unjustified_persistence_rate)}   "
            f"Unnecessary revision: {_fmt(metrics.unnecessary_revision_rate)}",
            f"Appropriate abstention: {_fmt(metrics.appropriate_abstention_rate)}   "
            f"Appropriate review: {_fmt(metrics.appropriate_review_rate)}",
        ]
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"
