"""Reassessment as an application operation over a stored answer.

The correction package already has primitives — ``reassess`` turns prior claims plus evidence
into a decision, and ``record_reassessment`` writes it down. This module lifts them to the
operation the conversation actually needs: given the *identifier* of a stored answer and new
evidence, re-ground that answer's recorded claims and produce a decision and a correction record.

It is deliberately a deterministic boundary, not an autonomous one (Section 16). The challenged
answer is named by ``AnswerId``; no natural-language resolver decides which answer "the previous
one" meant, because that resolver does not exist and a guess about it would be a fake capability.
The extension point is the identifier: a future front end resolves a follow-up to an ``AnswerId``
and calls this.

Two things it does *not* do, so the honesty is not accidental: it does not fabricate the prose of
a revised answer (there is no model), and it does not decide a contradiction's winner — a
``contradicted`` status still arrives only through a supplied grounding judgement (ADR-0015).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from chakaso.answers import AnswerId, AnswerStore, UnknownAnswerError
from chakaso.claims.model import ClaimId, ClaimStatus
from chakaso.core.identifiers import ChunkId
from chakaso.correction.reassess import Reassessment, reassess
from chakaso.correction.record import CorrectionRecord, record_reassessment
from chakaso.evidence import EvidencePack
from chakaso.grounding import GroundingEvaluator

__all__ = ["reassess_and_record", "reassess_stored_answer"]


def reassess_stored_answer(
    store: AnswerStore,
    prior_answer_id: AnswerId,
    reassessment_evidence: EvidencePack,
    *,
    grounding_evaluator: GroundingEvaluator | None = None,
    requires_citation: Collection[ClaimId] | None = None,
    expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
) -> Reassessment:
    """Re-ground a stored answer's recorded claims against new evidence.

    Args:
        store: where the prior answer is read from; nothing is written here.
        prior_answer_id: the answer being challenged. It must be recorded, or the call
            raises — a reassessment of an answer that does not exist is refused, not a no-op.
        reassessment_evidence: the evidence the claims are re-evaluated against. The caller
            composes it (standing evidence plus the new evidence, or new evidence alone to
            detect withdrawal), because "supported now" only means "supported by what you give"
            (ADR-0015).
        requires_citation: claims that must stay evidenced. When omitted it defaults to the
            claims the prior answer recorded as *supported* — a claim it asserted on evidence is
            still expected to be evidenced on reassessment — so a caller rarely needs to pass it.
        expected_evidence: per-claim expected chunks, forwarded to citation validation.

    Raises:
        UnknownAnswerError: ``prior_answer_id`` is not in the store.
    """
    prior = store.get(prior_answer_id)
    if prior is None:
        message = f"cannot reassess answer {prior_answer_id}: it is not recorded in the store"
        raise UnknownAnswerError(message)

    required = requires_citation
    if required is None:
        required = frozenset(
            claim.claim_id for claim in prior.claims if claim.status is ClaimStatus.SUPPORTED
        )

    return reassess(
        str(prior_answer_id),
        prior.claims,
        reassessment_evidence,
        grounding_evaluator=grounding_evaluator,
        requires_citation=required,
        expected_evidence=expected_evidence,
    )


def reassess_and_record(
    store: AnswerStore,
    prior_answer_id: AnswerId,
    reassessment_evidence: EvidencePack,
    *,
    grounding_evaluator: GroundingEvaluator | None = None,
    requires_citation: Collection[ClaimId] | None = None,
    expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
) -> tuple[Reassessment, CorrectionRecord]:
    """Reassess a stored answer and record the outcome against it.

    Returns the ``Reassessment`` and a ``CorrectionRecord`` whose ``supersedes`` names the prior
    answer, so the decision joins the answer's history. The record is returned, not silently
    persisted — storing answer *records* and storing correction *records* are separate concerns,
    and the caller decides where the correction goes.
    """
    result = reassess_stored_answer(
        store,
        prior_answer_id,
        reassessment_evidence,
        grounding_evaluator=grounding_evaluator,
        requires_citation=requires_citation,
        expected_evidence=expected_evidence,
    )
    record = record_reassessment(result, supersedes=str(prior_answer_id))
    return result, record
