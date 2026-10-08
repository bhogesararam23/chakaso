"""Tests for the reassessment engine and its answer-level decision.

The decision rule is only trustworthy if each branch is pinned: a contradiction outranks
everything, an unadjudicated conflict or an unsettled required claim goes to review rather than
a guess, lost support qualifies, total loss of grounding abstains, and anything else retains.
End to end, the structural evaluator can qualify or retain but never invents a contradiction;
a contradiction reaches ``CORRECT`` only when a supplied judgement asserts one.
"""

from __future__ import annotations

from chakaso.claims.model import Claim, ClaimId, ClaimStatus, derive_claim_id
from chakaso.core.identifiers import ChunkId
from chakaso.correction import (
    ClaimReassessment,
    CorrectionDecision,
    Reassessment,
    decide_answer,
    reassess,
)
from chakaso.correction.decision import ClaimDisposition
from chakaso.evidence import EvidencePack
from chakaso.grounding import Contradiction, ManualGroundingEvaluator
from chakaso.retrieval import ingest_text

_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
PACK = EvidencePack.of((*_ALPHA.chunks, *_BETA.chunks), [_ALPHA.source, _BETA.source])
BETA_ONLY = EvidencePack.of(_BETA.chunks, [_BETA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id


def cid(n: int) -> ClaimId:
    return derive_claim_id("ans", n, f"claim {n}")


def reassess_one(
    index: int,
    prior: ClaimStatus,
    new: ClaimStatus,
    *,
    required: bool = False,
) -> ClaimReassessment:
    return ClaimReassessment.assess(cid(index), prior, new, required=required)


def test_a_contradiction_outranks_a_weakened_claim() -> None:
    contradicted = reassess_one(0, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED)
    weakened = reassess_one(1, ClaimStatus.SUPPORTED, ClaimStatus.UNSUPPORTED)

    assert decide_answer([contradicted, weakened]) is CorrectionDecision.CORRECT


def test_a_recorded_conflict_without_a_winner_goes_to_review() -> None:
    retained = reassess_one(0, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED)
    conflict = Contradiction(cid(0), (ChunkId("chk_" + "1" * 16), ChunkId("chk_" + "2" * 16)))

    # A disagreement exists but no claim was adjudicated to contradicted; the engine will not
    # pick a winner, so it flags a human (ADR-0015) rather than deciding arbitrarily.
    assert decide_answer([retained], contradictions=[conflict]) is CorrectionDecision.NEEDS_REVIEW


def test_an_unsettled_required_claim_goes_to_review() -> None:
    unsettled = reassess_one(0, ClaimStatus.NOT_EVALUATED, ClaimStatus.UNCERTAIN, required=True)

    assert decide_answer([unsettled]) is CorrectionDecision.NEEDS_REVIEW


def test_lost_support_qualifies_when_nothing_is_contradicted() -> None:
    weakened = reassess_one(0, ClaimStatus.SUPPORTED, ClaimStatus.UNSUPPORTED)

    assert decide_answer([weakened]) is CorrectionDecision.QUALIFY


def test_a_required_answer_with_no_support_at_all_abstains() -> None:
    # Prior was never supported, so the per-claim disposition is retain (nothing changed); but
    # the whole answer is required and nothing is supported by the evidence, so abstention is
    # the honest outcome rather than quietly retaining.
    unsupported = reassess_one(0, ClaimStatus.NOT_EVALUATED, ClaimStatus.UNSUPPORTED, required=True)

    assert decide_answer([unsupported]) is CorrectionDecision.ABSTAIN


def test_no_change_retains_and_an_empty_reassessment_is_not_an_abstention() -> None:
    stable = reassess_one(0, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED)

    assert decide_answer([stable]) is CorrectionDecision.RETAIN
    assert decide_answer([]) is CorrectionDecision.RETAIN


def _prior_claim(cited: tuple[ChunkId, ...]) -> Claim:
    return Claim(
        answer_id="ans",
        position=0,
        text="alpha claim",
        cited_evidence_ids=cited,
        status=ClaimStatus.SUPPORTED,
    )


def test_reassess_qualifies_when_evidence_no_longer_supports_a_required_claim() -> None:
    claim = _prior_claim((CHUNK_A,))

    result = reassess("ans", [claim], BETA_ONLY, requires_citation={claim.claim_id})

    assert isinstance(result, Reassessment)
    assert result.decision is CorrectionDecision.QUALIFY
    assert result.qualifications == (claim.claim_id,)
    assert result.evaluator_name == "structural-0.1"


def test_reassess_retains_when_the_evidence_still_supports_the_claim() -> None:
    claim = _prior_claim((CHUNK_A,))

    result = reassess("ans", [claim], PACK, requires_citation={claim.claim_id})

    assert result.decision is CorrectionDecision.RETAIN
    assert result.changed is False


def test_a_supplied_judgement_is_the_only_route_to_a_correction() -> None:
    claim = _prior_claim((CHUNK_A,))
    evaluator = ManualGroundingEvaluator({claim.claim_id: ClaimStatus.CONTRADICTED})

    result = reassess("ans", [claim], PACK, grounding_evaluator=evaluator)

    assert result.decision is CorrectionDecision.CORRECT
    assert result.corrections == (claim.claim_id,)


def test_structural_reassessment_cannot_invent_a_contradiction() -> None:
    claim = _prior_claim((CHUNK_A,))

    result = reassess("ans", [claim], PACK)

    assert result.contradictions == ()
    assert result.decision is not CorrectionDecision.CORRECT
    # with no required claim and a valid citation the structural pass simply re-confirms support
    assert result.claims[0].disposition is ClaimDisposition.RETAIN
