"""The per-claim correction decision: what reassessment decides to do with a claim.

Correction is specified in ``docs/correction.md``; this is the smallest part of it that can
be written honestly today. It maps the status a claim *had* onto the status the supplied
evidence supports *now*, into one of three dispositions — retain, qualify, correct — under
rules stated in full so that no outcome is a guess:

* a claim is **corrected** only when the supplied evidence *contradicts* it — never because it
  was produced first, has aged, or came from a less-preferred source (ADR-0015 refuses to rank
  sources automatically, and correction inherits that refusal);
* a claim is **qualified** when it previously read as supported and the evidence no longer
  supports it, or bears on it without settling it — this is the form of correction that does
  not require anyone to have been wrong;
* otherwise the claim is **retained**: evidence that merely re-confirms a claim, or is silent
  about it, is not a reason to touch it.

The answer-level disposition (including ``abstain`` and ``needs_review``) is derived from these
per-claim decisions by the reassessment engine, because abstention and review describe a whole
answer, not one claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from chakaso.claims.model import ClaimId, ClaimStatus

__all__ = [
    "ClaimDisposition",
    "ClaimReassessment",
    "CorrectionDecision",
    "decide_claim",
]


class ClaimDisposition(StrEnum):
    """What reassessment decides to do with one claim."""

    RETAIN = "retain"
    QUALIFY = "qualify"
    CORRECT = "correct"


class CorrectionDecision(StrEnum):
    """The disposition of a whole answer after reassessment.

    ``ABSTAIN`` and ``NEEDS_REVIEW`` are answer-level rather than per-claim: they name the
    cases where retain/qualify/correct is the wrong response — the evidence no longer supports
    anything the answer asserted, or a required claim is left unsettled and a human must decide
    it. Neither is produced by guessing at a winner.
    """

    RETAIN = "retain"
    QUALIFY = "qualify"
    CORRECT = "correct"
    ABSTAIN = "abstain"
    NEEDS_REVIEW = "needs_review"


def decide_claim(prior_status: ClaimStatus, new_status: ClaimStatus) -> ClaimDisposition:
    """Decide one claim's disposition from its prior status and its status under new evidence.

    ``new_status`` is what the supplied evidence establishes *now* (ADR-0014/0015); a
    contradiction can only have been asserted by a judgement, never inferred from recency here.
    """
    if new_status is ClaimStatus.CONTRADICTED:
        return ClaimDisposition.CORRECT
    # Losing (or failing to settle) support a claim previously had is a reason to soften it,
    # not to keep asserting it as though it were evidenced; a claim that never read as
    # supported is unchanged by evidence that still does not support it.
    unsettled = new_status in (ClaimStatus.UNSUPPORTED, ClaimStatus.UNCERTAIN)
    if unsettled and prior_status is ClaimStatus.SUPPORTED:
        return ClaimDisposition.QUALIFY
    return ClaimDisposition.RETAIN


@dataclass(frozen=True, slots=True)
class ClaimReassessment:
    """One claim before and after reassessment, with the disposition the rules decided."""

    claim_id: ClaimId
    prior_status: ClaimStatus
    new_status: ClaimStatus
    required: bool
    disposition: ClaimDisposition

    @classmethod
    def assess(
        cls,
        claim_id: ClaimId,
        prior_status: ClaimStatus,
        new_status: ClaimStatus,
        *,
        required: bool = False,
    ) -> ClaimReassessment:
        """Build a reassessment, deriving the disposition from the shared ``decide_claim`` rule."""
        return cls(
            claim_id=claim_id,
            prior_status=prior_status,
            new_status=new_status,
            required=required,
            disposition=decide_claim(prior_status, new_status),
        )

    @property
    def changed(self) -> bool:
        """Whether reassessment decides to do anything other than leave the claim as it was."""
        return self.disposition is not ClaimDisposition.RETAIN
