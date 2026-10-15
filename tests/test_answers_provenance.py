"""Tests for the historical evaluation snapshot (ADR-0028).

The snapshot is a *record of what a completed turn decided*, distinct from the live,
recomputable ``AnswerEvaluation``. These cover the mapping from a real evaluation, the derived
views, and the construction invariants that make a corrupt or impossible snapshot fail loudly
rather than become trustworthy-looking research history.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.answers import EvaluationProvenance, GroundedClaim, InvalidAnswerError
from chakaso.claims.model import Claim, ClaimId, ClaimStatus
from chakaso.core.identifiers import ChunkId
from chakaso.evaluation import evaluate_answer
from chakaso.evidence import EvidencePack, resolve_citations
from chakaso.retrieval import ingest_text

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
_DOC = ingest_text("Alpha is a fact about chakaso.", label="alpha", now=lambda: NOW)
_PACK = EvidencePack.of(_DOC.chunks, [_DOC.source])
_CHUNK = _DOC.chunks[0].chunk_id
_OUTSIDE = ChunkId("chk_" + "0" * 16)  # valid format, deliberately not in the pack


def _claims() -> tuple[Claim, Claim, Claim]:
    supported = Claim("ans_x", 0, "Alpha is a fact.", cited_evidence_ids=(_CHUNK,))
    fabricated = Claim("ans_x", 1, "Beta needs a source.", cited_evidence_ids=(_OUTSIDE,))
    bare = Claim("ans_x", 2, "Gamma is uncited.")
    return supported, fabricated, bare


def test_from_turn_freezes_citation_and_grounding_outcomes() -> None:
    supported, fabricated, bare = _claims()
    claims = (supported, fabricated, bare)
    evaluation = evaluate_answer(
        "ans_x", claims, _PACK, requires_citation={fabricated.claim_id, bare.claim_id}
    )
    resolution = resolve_citations(f"See [{_CHUNK}] and [{_OUTSIDE}] plus a bad [src_0123].", _PACK)

    provenance = EvaluationProvenance.from_turn(evaluation, resolution)

    assert provenance.claims == 3
    # Structural citation counts come straight off the evaluation's metrics.
    assert provenance.citations_valid == 1
    assert provenance.citations_unknown == 1
    assert provenance.citations_made == 2
    assert provenance.required_claims == 2
    # The fabricated citation names a chunk outside the pack, so that claim is unsupported; the
    # supported claim is supported; the bare claim (required, no citation) is unsupported.
    assert provenance.supported_claims == (supported.claim_id,)
    assert set(provenance.unsupported_claims) == {fabricated.claim_id, bare.claim_id}
    # The named evaluator is recorded, so a change of judge is visible in provenance.
    assert provenance.grounding_evaluator == "structural-0.1"
    # Invented / malformed references in the answer text become durable facts (closes K-006's
    # "nothing persists this" condition). They are stored as their string form, exactly as the
    # citation-resolution surface reports them.
    assert str(_OUTSIDE) in provenance.unresolved_reference_ids
    assert "src_0123" in provenance.malformed_reference_ids
    assert provenance.has_unresolved_references is True
    # Structural grounding cannot call this grounded: an unknown citation exists.
    assert provenance.is_grounded is False


def test_derived_views_match_the_recomputed_evaluation() -> None:
    supported, fabricated, _bare = _claims()
    claims = (supported, fabricated)
    evaluation = evaluate_answer("ans_x", claims, _PACK)
    resolution = resolve_citations("only a real cite", _PACK)

    provenance = EvaluationProvenance.from_turn(evaluation, resolution)

    # The snapshot's evidence_coverage mirrors the live evaluation's definition.
    assert provenance.evidence_coverage == evaluation.evidence_coverage
    assert provenance.citation_precision == evaluation.citations.citation_precision
    assert provenance.has_unresolved_references is False


def test_grounding_is_ordered_and_deduplicated_by_claim_id() -> None:
    a = GroundedClaim(_claim_id("a"), ClaimStatus.SUPPORTED)
    b = GroundedClaim(_claim_id("b"), ClaimStatus.UNSUPPORTED)

    # Build provenance with claims supplied out of order and one duplicated.
    provenance = EvaluationProvenance(
        claims=2,
        citations_made=0,
        citations_valid=0,
        citations_unknown=0,
        citations_irrelevant=0,
        required_claims=0,
        uncited_claims=0,
        grounded_claims=(b, a, b),
    )

    assert [item.claim_id for item in provenance.grounded_claims] == sorted(
        [a.claim_id, b.claim_id], key=str
    )


def test_negative_counts_are_refused() -> None:
    with pytest.raises(InvalidAnswerError, match="must not be negative"):
        EvaluationProvenance(
            claims=-1,
            citations_made=0,
            citations_valid=0,
            citations_unknown=0,
            citations_irrelevant=0,
            required_claims=0,
            uncited_claims=0,
        )


def test_inconsistent_citation_counts_are_refused() -> None:
    # valid + unknown + irrelevant must equal citations made — a snapshot that disagrees is corrupt.
    with pytest.raises(InvalidAnswerError, match="inconsistent"):
        EvaluationProvenance(
            claims=1,
            citations_made=2,
            citations_valid=1,
            citations_unknown=0,
            citations_irrelevant=0,
            required_claims=0,
            uncited_claims=0,
        )


def test_uncited_cannot_exceed_required_claims() -> None:
    with pytest.raises(InvalidAnswerError, match="uncited claims"):
        EvaluationProvenance(
            claims=1,
            citations_made=0,
            citations_valid=0,
            citations_unknown=0,
            citations_irrelevant=0,
            required_claims=1,
            uncited_claims=2,
        )


def test_an_empty_evaluator_name_is_refused() -> None:
    with pytest.raises(InvalidAnswerError, match="must name its evaluator"):
        EvaluationProvenance(
            claims=0,
            citations_made=0,
            citations_valid=0,
            citations_unknown=0,
            citations_irrelevant=0,
            required_claims=0,
            uncited_claims=0,
            grounding_evaluator="   ",
        )


def test_a_supported_snapshot_is_grounded_but_means_only_structure() -> None:
    claim = Claim("ans_y", 0, "Alpha is a fact.", cited_evidence_ids=(_CHUNK,))
    evaluation = evaluate_answer("ans_y", (claim,), _PACK)
    resolution = resolve_citations(f"From [{_CHUNK}].", _PACK)

    provenance = EvaluationProvenance.from_turn(evaluation, resolution)

    # is_grounded here is the structural definition: every claim supported, none contradicted,
    # no unknown citation. It is not a claim that the evidence proves the assertion (ADR-0015).
    assert provenance.is_grounded is True
    assert provenance.evidence_coverage == 1.0


def _claim_id(text: str) -> ClaimId:
    # A real claim's derived identifier; the ordering test only needs distinct, valid claim ids.
    return Claim("ans_z", 0, text).claim_id
