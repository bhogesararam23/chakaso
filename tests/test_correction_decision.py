"""Tests for the per-claim correction decision.

The whole point of writing the rule down is that correction is not a mood: a claim is
corrected *only* when the supplied evidence contradicts it, qualified only when support it
previously had falls away, and retained otherwise. These tests pin every branch of that table
and, crucially, the honesty invariants — new evidence that merely re-confirms or says nothing
about a claim must never trigger a change, and losing support never silently becomes "true."
"""

from __future__ import annotations

import pytest

from chakaso.claims.model import ClaimStatus, derive_claim_id
from chakaso.correction import (
    ClaimDisposition,
    ClaimReassessment,
    CorrectionDecision,
    decide_claim,
)

CID = derive_claim_id("ans", 0, "claim zero")


@pytest.mark.parametrize(
    ("prior", "new", "expected"),
    [
        # Contradiction by the supplied evidence is the only route to a correction.
        (ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED, ClaimDisposition.CORRECT),
        (ClaimStatus.NOT_EVALUATED, ClaimStatus.CONTRADICTED, ClaimDisposition.CORRECT),
        # Support previously held and now lost (or left unsettled) is softened, not deleted.
        (ClaimStatus.SUPPORTED, ClaimStatus.UNSUPPORTED, ClaimDisposition.QUALIFY),
        (ClaimStatus.SUPPORTED, ClaimStatus.UNCERTAIN, ClaimDisposition.QUALIFY),
        # A claim that never read as supported is unchanged by evidence that still is not.
        (ClaimStatus.UNSUPPORTED, ClaimStatus.UNSUPPORTED, ClaimDisposition.RETAIN),
        (ClaimStatus.NOT_EVALUATED, ClaimStatus.UNCERTAIN, ClaimDisposition.RETAIN),
        # Re-confirmation and silence are both reasons to leave a claim alone.
        (ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED, ClaimDisposition.RETAIN),
        (ClaimStatus.SUPPORTED, ClaimStatus.NOT_EVALUATED, ClaimDisposition.RETAIN),
        (ClaimStatus.UNSUPPORTED, ClaimStatus.SUPPORTED, ClaimDisposition.RETAIN),
    ],
)
def test_decide_claim_follows_the_stated_table(
    prior: ClaimStatus, new: ClaimStatus, expected: ClaimDisposition
) -> None:
    assert decide_claim(prior, new) is expected


def test_a_reconfirmed_claim_is_never_corrected_for_being_older() -> None:
    # The rule inspects only the evidence status, never which answer came first or how old it
    # is: nothing here can decide a contest on recency, which is ADR-0015's boundary inherited.
    assert decide_claim(ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED) is ClaimDisposition.RETAIN


def test_assess_derives_the_disposition_from_the_shared_rule() -> None:
    reassessment = ClaimReassessment.assess(
        CID, ClaimStatus.SUPPORTED, ClaimStatus.UNSUPPORTED, required=True
    )

    assert reassessment.disposition is ClaimDisposition.QUALIFY
    assert reassessment.required is True
    assert reassessment.changed is True


def test_a_retained_claim_is_recorded_as_unchanged() -> None:
    reassessment = ClaimReassessment.assess(CID, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED)

    assert reassessment.disposition is ClaimDisposition.RETAIN
    assert reassessment.changed is False


def test_decision_vocabulary_is_the_documented_set() -> None:
    assert {member.value for member in CorrectionDecision} == {
        "retain",
        "qualify",
        "correct",
        "abstain",
        "needs_review",
    }
    assert {member.value for member in ClaimDisposition} == {"retain", "qualify", "correct"}
