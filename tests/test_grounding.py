"""Tests for grounding evaluation and its model.

The structural evaluator must never overclaim: it marks a claim supported or unsupported by
citation presence and leaves the rest unevaluated, and it cannot produce contradiction or
uncertainty. Those come only from a supplied judgement. The precedence hook is checked to
refuse to guess when it has no basis.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.claims.model import Claim, ClaimStatus
from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.evidence import EvidencePack
from chakaso.grounding import (
    Contradiction,
    GroundingEvaluator,
    ManualGroundingEvaluator,
    SourceAuthority,
    StructuralGroundingEvaluator,
    most_recent_wins,
)
from chakaso.retrieval import ingest_text

_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
PACK = EvidencePack.of((*_ALPHA.chunks, *_BETA.chunks), [_ALPHA.source, _BETA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id
CHUNK_B = _BETA.chunks[0].chunk_id


def claim(index: int, cited: tuple[ChunkId, ...] = ()) -> Claim:
    return Claim(answer_id="ans", position=index, text=f"claim {index}", cited_evidence_ids=cited)


def src(n: int) -> SourceId:
    return SourceId(f"src_{n:016x}")


def test_structural_evaluator_maps_presence_to_supported_or_unsupported() -> None:
    supported = claim(0, (CHUNK_A,))
    unsupported = claim(1)
    unevaluated = claim(2)

    result = StructuralGroundingEvaluator().evaluate(
        "ans",
        [supported, unsupported, unevaluated],
        PACK,
        requires_citation={unsupported.claim_id},
    )

    assert result.status_of(supported) is ClaimStatus.SUPPORTED
    assert result.status_of(unsupported) is ClaimStatus.UNSUPPORTED
    assert result.status_of(unevaluated) is ClaimStatus.NOT_EVALUATED


def test_structural_evaluator_never_invents_contradiction() -> None:
    result = StructuralGroundingEvaluator().evaluate("ans", [claim(0, (CHUNK_A,))], PACK)

    assert result.contradicted == ()
    assert result.is_fully_supported is True


def test_expected_evidence_makes_a_present_but_wrong_citation_unsupported_via_no_valid() -> None:
    # A claim citing CHUNK_B when only CHUNK_A was expected has no *valid* citation, so
    # structural grounding marks it unsupported when evidence was required.
    wrong = claim(0, (CHUNK_B,))
    result = StructuralGroundingEvaluator().evaluate(
        "ans",
        [wrong],
        PACK,
        requires_citation={wrong.claim_id},
        expected_evidence={wrong.claim_id: {CHUNK_A}},
    )

    assert result.status_of(wrong) is ClaimStatus.UNSUPPORTED


def test_manual_evaluator_can_assert_contradiction_and_carries_it() -> None:
    conflicted = claim(0)
    contradiction = Contradiction(conflicted.claim_id, (CHUNK_A, CHUNK_B), detail="dates differ")

    result = ManualGroundingEvaluator(
        {conflicted.claim_id: ClaimStatus.CONTRADICTED},
        [contradiction],
    ).evaluate("ans", [conflicted], PACK)

    assert result.status_of(conflicted) is ClaimStatus.CONTRADICTED
    assert result.contradicted == (conflicted.claim_id,)
    assert result.contradictions == (contradiction,)
    assert result.is_fully_supported is False


def test_status_of_defaults_to_not_evaluated_for_unknown_claim() -> None:
    result = ManualGroundingEvaluator({}).evaluate("ans", [], PACK)

    assert result.status_of(claim(9)) is ClaimStatus.NOT_EVALUATED


def test_a_contradiction_needs_at_least_two_chunks() -> None:
    with pytest.raises(ValueError, match="at least two"):
        Contradiction(claim(0).claim_id, (CHUNK_A,))


def test_most_recent_wins_picks_the_newest_or_refuses_to_guess() -> None:
    older = SourceAuthority(src(1), published_at=datetime(2020, 1, 1, tzinfo=UTC))
    newer = SourceAuthority(src(2), published_at=datetime(2024, 1, 1, tzinfo=UTC))

    assert most_recent_wins([older, newer]) is newer
    assert most_recent_wins([SourceAuthority(src(3))]) is None
    assert most_recent_wins([older, older]) is None  # tie: no winner declared


def test_both_evaluators_satisfy_the_protocol() -> None:
    assert isinstance(StructuralGroundingEvaluator(), GroundingEvaluator)
    assert isinstance(ManualGroundingEvaluator({}), GroundingEvaluator)
