"""Reassessment: re-evaluate an answer's claims against evidence and decide what to do.

This is the orchestration the correction contract in ``docs/correction.md`` calls for, built
on the boundaries that already exist: the claims carry the statuses they had when the answer
was made, a ``GroundingEvaluator`` re-establishes their statuses against the evidence supplied
now (ADR-0015), and ``decide_claim`` turns each before/after pair into a per-claim disposition.
What ``reassess`` adds is the answer-level decision — the step that recognises a contradiction
rather than defending the earlier answer because it was there first.

The rule is written down in full (and recorded in ADR-0016) because an unstated aggregation is
indistinguishable from an arbitrary one. It deliberately does *not* produce the wording of a
revised answer: there is no language model, so correction here decides and records what should
change, and rewording is an explicitly separate future step.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from chakaso.claims.model import Claim, ClaimId, ClaimStatus
from chakaso.core.identifiers import ChunkId
from chakaso.correction.decision import (
    ClaimDisposition,
    ClaimReassessment,
    CorrectionDecision,
)
from chakaso.evidence import EvidencePack
from chakaso.grounding import (
    Contradiction,
    GroundingEvaluator,
    StructuralGroundingEvaluator,
)

__all__ = ["Reassessment", "decide_answer", "reassess"]


@dataclass(frozen=True, slots=True)
class Reassessment:
    """The outcome of re-evaluating one answer's claims against evidence supplied now."""

    answer_id: str
    claims: tuple[ClaimReassessment, ...]
    decision: CorrectionDecision
    contradictions: tuple[Contradiction, ...]
    evaluator_name: str

    @property
    def changed(self) -> bool:
        """Whether reassessment decided to do anything at all to the answer."""
        return self.decision is not CorrectionDecision.RETAIN

    def claims_with(self, disposition: ClaimDisposition) -> tuple[ClaimId, ...]:
        """Claim identifiers decided to have that disposition, ordered by their string form."""
        return tuple(
            sorted(
                (r.claim_id for r in self.claims if r.disposition is disposition),
                key=str,
            )
        )

    @property
    def corrections(self) -> tuple[ClaimId, ...]:
        """Claims the supplied evidence contradicts and so must be corrected."""
        return self.claims_with(ClaimDisposition.CORRECT)

    @property
    def qualifications(self) -> tuple[ClaimId, ...]:
        """Claims whose support fell away and should be softened rather than removed."""
        return self.claims_with(ClaimDisposition.QUALIFY)


def decide_answer(
    reassessments: Sequence[ClaimReassessment],
    *,
    contradictions: Collection[Contradiction] = (),
) -> CorrectionDecision:
    """Aggregate per-claim dispositions into one answer-level decision.

    Precedence, highest first, each trigger specific and tested:

    1. ``CORRECT`` — some claim is contradicted by the evidence. This is the only definite case
       where the answer asserted something the evidence opposes.
    2. ``NEEDS_REVIEW`` — a conflict was recorded that was *not* adjudicated into a contradicted
       status, or a claim required to be evidenced was left uncertain. The engine refuses to pick
       a winner (ADR-0015), so it flags a human instead of guessing.
    3. ``QUALIFY`` — some claim lost support it previously had, without anything being contradicted.
    4. ``ABSTAIN`` — there were required claims and, over the reassessment evidence, none is
       supported: the answer can no longer be grounded, so withholding beats asserting.
    5. ``RETAIN`` — otherwise nothing the evidence now supports has changed.

    An empty reassessment (no claims) is ``RETAIN``: with nothing asserted there is nothing to
    revise, and it is not reported as an abstention.
    """
    if any(r.disposition is ClaimDisposition.CORRECT for r in reassessments):
        return CorrectionDecision.CORRECT
    required = [r for r in reassessments if r.required]
    unsettled = any(r.new_status is ClaimStatus.UNCERTAIN for r in required)
    if contradictions or unsettled:
        return CorrectionDecision.NEEDS_REVIEW
    if any(r.disposition is ClaimDisposition.QUALIFY for r in reassessments):
        return CorrectionDecision.QUALIFY
    if required and not any(r.new_status is ClaimStatus.SUPPORTED for r in reassessments):
        return CorrectionDecision.ABSTAIN
    return CorrectionDecision.RETAIN


def reassess(
    answer_id: str,
    prior_claims: Sequence[Claim],
    evidence: EvidencePack,
    *,
    grounding_evaluator: GroundingEvaluator | None = None,
    requires_citation: Collection[ClaimId] | None = None,
    expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
) -> Reassessment:
    """Re-ground ``prior_claims`` against ``evidence`` and decide the answer's disposition.

    Args:
        answer_id: the answer being reassessed; carried onto the result for the record.
        prior_claims: the answer's claims, each carrying the status it held when asserted.
        evidence: the pack to reassess against. The caller composes it — typically the evidence
            still standing plus the new evidence that prompted reassessment — because "supported
            now" only ever means "supported by the pack you supplied" (ADR-0015).
        grounding_evaluator: the boundary that re-establishes statuses; the structural default
            can only produce supported/unsupported/not_evaluated, so a contradiction or
            uncertainty arrives only through a supplied judgement.
        requires_citation: claims that must be evidenced; drives both grounding and the
            answer-level abstention/review conditions.
        expected_evidence: per-claim expected chunks, so a present-but-wrong citation reads as
            no valid support, exactly as in first-pass citation validation.
    """
    evaluator = grounding_evaluator or StructuralGroundingEvaluator()
    grounding = evaluator.evaluate(
        answer_id,
        prior_claims,
        evidence,
        requires_citation=requires_citation,
        expected_evidence=expected_evidence,
    )
    required = requires_citation or frozenset()
    reassessments = tuple(
        ClaimReassessment.assess(
            claim.claim_id,
            claim.status,
            grounding.status_of(claim),
            required=claim.claim_id in required,
        )
        for claim in prior_claims
    )
    return Reassessment(
        answer_id=answer_id,
        claims=reassessments,
        decision=decide_answer(reassessments, contradictions=grounding.contradictions),
        contradictions=grounding.contradictions,
        evaluator_name=grounding.evaluator_name,
    )
