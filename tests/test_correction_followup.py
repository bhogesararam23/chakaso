"""Tests for the follow-up reassessment integration.

The integration is thin on purpose, so the tests check exactly the two things it adds and no
more: that merging evidence packs is de-duplicated and independent of assembly order, and that a
follow-up reassesses against the evidence the caller composes — keeping a claim supported when
standing evidence is included, qualifying it when support is withdrawn by reassessing against new
evidence alone, and correcting only when a supplied judgement contradicts it.
"""

from __future__ import annotations

from chakaso.claims.model import Claim, ClaimStatus
from chakaso.correction import (
    CorrectionDecision,
    merge_evidence,
    reassess_followup,
    record_reassessment,
)
from chakaso.evidence import EvidencePack
from chakaso.grounding import ManualGroundingEvaluator
from chakaso.retrieval import ingest_text

_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
_GAMMA = ingest_text("Gamma is another fact.", label="gamma")
CHUNK_A = _ALPHA.chunks[0].chunk_id
AB_PACK = EvidencePack.of((*_ALPHA.chunks, *_BETA.chunks), [_ALPHA.source, _BETA.source])
BG_PACK = EvidencePack.of((*_BETA.chunks, *_GAMMA.chunks), [_BETA.source, _GAMMA.source])
GAMMA_PACK = EvidencePack.of(_GAMMA.chunks, [_GAMMA.source])


def prior_claim() -> Claim:
    return Claim(
        answer_id="ans",
        position=0,
        text="alpha claim",
        cited_evidence_ids=(CHUNK_A,),
        status=ClaimStatus.SUPPORTED,
    )


def test_merge_unions_packs_without_duplicates_or_order_dependence() -> None:
    forward = merge_evidence(AB_PACK, BG_PACK)
    backward = merge_evidence(BG_PACK, AB_PACK)

    ids = [chunk.chunk_id for chunk in forward.chunks]
    assert len(ids) == len(set(ids)) == 3
    assert forward.chunks == backward.chunks


def test_followup_keeps_a_claim_supported_when_standing_evidence_is_included() -> None:
    claim = prior_claim()

    result = reassess_followup(
        "ans",
        [claim],
        GAMMA_PACK,
        standing_evidence=AB_PACK,
        requires_citation={claim.claim_id},
    )

    assert result.decision is CorrectionDecision.RETAIN


def test_followup_qualifies_when_support_is_withdrawn_by_reassessing_on_new_evidence_only() -> None:
    claim = prior_claim()

    # No standing evidence: the pack contains only new material, so the prior citation is no
    # longer supplied and the required claim reads unsupported — the honest "support withdrawn".
    result = reassess_followup("ans", [claim], GAMMA_PACK, requires_citation={claim.claim_id})

    assert result.decision is CorrectionDecision.QUALIFY
    assert result.qualifications == (claim.claim_id,)


def test_a_supplied_judgement_can_correct_a_followup_reassessment() -> None:
    claim = prior_claim()
    evaluator = ManualGroundingEvaluator({claim.claim_id: ClaimStatus.CONTRADICTED})

    result = reassess_followup(
        "ans", [claim], GAMMA_PACK, standing_evidence=AB_PACK, grounding_evaluator=evaluator
    )

    assert result.decision is CorrectionDecision.CORRECT
    assert result.corrections == (claim.claim_id,)


def test_a_followup_reassessment_records_as_a_revision() -> None:
    claim = prior_claim()

    result = reassess_followup("ans", [claim], GAMMA_PACK, requires_citation={claim.claim_id})
    record = record_reassessment(result, supersedes="cor_prior")

    assert record.decision is CorrectionDecision.QUALIFY
    assert record.is_revision is True
