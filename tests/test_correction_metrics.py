"""Tests for the correction metrics.

The three rates exist to catch two opposite failures, so the tests keep them separate: a system
that never revises scores well on nothing and badly on persistence; a system that revises on every
challenge scores well on persistence and badly on unnecessary revision. Success means the decision
matched the case's expected decision exactly — not merely "something happened" — and a rate whose
denominator has no cases is ``None``, never a made-up number.
"""

from __future__ import annotations

from chakaso.claims.model import ClaimStatus, derive_claim_id
from chakaso.correction import (
    ClaimReassessment,
    CorrectionDecision,
    CorrectionRecord,
    LabeledCorrection,
    correction_metrics,
)


def labeled(decision: CorrectionDecision, expected: CorrectionDecision) -> LabeledCorrection:
    return LabeledCorrection(decision=decision, expected_decision=expected)


def test_a_mixed_set_separates_persistence_from_unnecessary_revision() -> None:
    cases = [
        labeled(CorrectionDecision.CORRECT, CorrectionDecision.CORRECT),  # required, done right
        labeled(CorrectionDecision.RETAIN, CorrectionDecision.CORRECT),  # required, defended anyway
        labeled(
            CorrectionDecision.CORRECT, CorrectionDecision.RETAIN
        ),  # not required, revised anyway
        labeled(CorrectionDecision.RETAIN, CorrectionDecision.RETAIN),  # not required, left alone
    ]

    metrics = correction_metrics(cases)

    assert metrics.cases == 4
    assert metrics.required_cases == 2
    assert metrics.not_required_cases == 2
    assert metrics.successful_cases == 1
    assert metrics.persisted_cases == 1
    assert metrics.unnecessarily_revised_cases == 1
    assert metrics.correction_success_rate == 0.5
    assert metrics.unjustified_persistence_rate == 0.5
    assert metrics.unnecessary_revision_rate == 0.5


def test_an_empty_set_reports_no_rates_rather_than_inventing_them() -> None:
    metrics = correction_metrics([])

    assert metrics.cases == 0
    assert metrics.correction_success_rate is None
    assert metrics.unjustified_persistence_rate is None
    assert metrics.unnecessary_revision_rate is None


def test_a_wrong_kind_of_revision_counts_as_not_successful_but_not_persistent() -> None:
    # The answer did change, so it is not "persistence", but it did not reach the expected
    # decision, so it is not "success" either — the two are genuinely different failures.
    metrics = correction_metrics([labeled(CorrectionDecision.QUALIFY, CorrectionDecision.CORRECT)])

    assert metrics.required_cases == 1
    assert metrics.successful_cases == 0
    assert metrics.persisted_cases == 0
    assert metrics.correction_success_rate == 0.0
    assert metrics.unjustified_persistence_rate == 0.0


def test_abstention_and_review_count_as_having_changed_the_response() -> None:
    # Neither is RETAIN, so leaving a not-required answer to abstain reads as an unnecessary
    # revision — a rate that would be hidden if "revision" meant only correct/qualify.
    metrics = correction_metrics(
        [
            labeled(CorrectionDecision.ABSTAIN, CorrectionDecision.RETAIN),
            labeled(CorrectionDecision.NEEDS_REVIEW, CorrectionDecision.RETAIN),
        ]
    )

    assert metrics.not_required_cases == 2
    assert metrics.unnecessarily_revised_cases == 2
    assert metrics.unnecessary_revision_rate == 1.0


def test_requires_revision_is_read_from_the_case_label() -> None:
    assert labeled(CorrectionDecision.RETAIN, CorrectionDecision.QUALIFY).requires_revision is True
    assert labeled(CorrectionDecision.CORRECT, CorrectionDecision.RETAIN).requires_revision is False


def test_from_record_labels_a_recorded_reassessment() -> None:
    claim_id = derive_claim_id("ans", 0, "claim zero")
    record = CorrectionRecord(
        answer_id="ans",
        decision=CorrectionDecision.CORRECT,
        claims=(
            ClaimReassessment.assess(claim_id, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED),
        ),
        contradictions=(),
        evaluator_name="manual-0.1",
    )

    labeled_record = LabeledCorrection.from_record(
        record, expected_decision=CorrectionDecision.CORRECT
    )

    assert labeled_record.decision is CorrectionDecision.CORRECT
    assert labeled_record.requires_revision is True
