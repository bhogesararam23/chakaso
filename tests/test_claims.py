"""Tests for the claim model.

The properties that matter: a claim identifier is stable under evaluation (so a citation or
correction can point at a claim across a run), a status never changes identity, the
default status is *not evaluated* (so an unchecked claim is not read as a supported one),
and malformed claims are refused.
"""

from __future__ import annotations

import pytest

from chakaso.claims import Claim, ClaimId, ClaimStatus, InvalidClaimError, derive_claim_id
from chakaso.core.identifiers import ChunkId


def chk(n: int) -> ChunkId:
    return ChunkId(f"chk_{n:016x}")


def make_claim(**overrides: object) -> Claim:
    fields: dict[str, object] = {
        "answer_id": "ans-1",
        "position": 0,
        "text": "Paris is the capital of France.",
    }
    fields.update(overrides)
    return Claim(**fields)  # type: ignore[arg-type]


def test_text_is_trimmed_and_status_defaults_to_not_evaluated() -> None:
    claim = make_claim(text="  padded claim  ")

    assert claim.text == "padded claim"
    assert claim.status is ClaimStatus.NOT_EVALUATED


def test_claim_id_is_stable_across_evaluation() -> None:
    claim = make_claim()
    evaluated = claim.with_status(ClaimStatus.SUPPORTED)

    assert evaluated.claim_id == claim.claim_id
    assert evaluated.status is ClaimStatus.SUPPORTED
    # The original is untouched — claims are immutable and correction preserves history.
    assert claim.status is ClaimStatus.NOT_EVALUATED


def test_derived_claim_id_is_deterministic_and_input_sensitive() -> None:
    base = derive_claim_id("ans", 0, "text")
    assert base == derive_claim_id("ans", 0, "text")
    assert base != derive_claim_id("ans", 1, "text")
    assert base != derive_claim_id("ans", 0, "other")
    assert base != derive_claim_id("other", 0, "text")


def test_cited_evidence_is_deduplicated_in_order() -> None:
    claim = make_claim(cited_evidence_ids=(chk(2), chk(1), chk(2)))

    assert claim.cited_evidence_ids == (chk(2), chk(1))
    assert claim.is_cited is True


def test_a_claim_without_evidence_is_not_cited() -> None:
    assert make_claim().is_cited is False


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_blank_text_is_refused(text: str) -> None:
    with pytest.raises(InvalidClaimError, match="must have text"):
        make_claim(text=text)


def test_negative_position_is_refused() -> None:
    with pytest.raises(InvalidClaimError, match="negative"):
        make_claim(position=-1)


@pytest.mark.parametrize("answer_id", ["", "  "])
def test_a_claim_must_name_its_answer(answer_id: str) -> None:
    with pytest.raises(InvalidClaimError, match="answer_id"):
        make_claim(answer_id=answer_id)


def test_claim_id_must_be_well_formed() -> None:
    assert str(ClaimId("clm_" + "0" * 16)) == "clm_" + "0" * 16

    with pytest.raises(InvalidClaimError, match="not a valid claim identifier"):
        ClaimId("clm_short")
