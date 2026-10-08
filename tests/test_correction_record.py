"""Tests for the correction record.

A correction has to be recorded, deterministic and append-only to mean anything: the same
reassessment must yield the same identifier regardless of the order claims were supplied in, a
record may only claim to supersede an earlier answer when it actually did, and the summary must
be an honest structural tally rather than fluent text pretending to be a revised answer (there
is no model to produce one).
"""

from __future__ import annotations

from chakaso.claims.model import Claim, ClaimId, ClaimStatus, derive_claim_id
from chakaso.correction import (
    ClaimReassessment,
    CorrectionDecision,
    CorrectionRecord,
    reassess,
    record_reassessment,
)
from chakaso.correction.decision import ClaimDisposition
from chakaso.evidence import EvidencePack
from chakaso.grounding import Contradiction
from chakaso.retrieval import ingest_text

_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
CHUNK_A = _ALPHA.chunks[0].chunk_id
BETA_ONLY = EvidencePack.of(_BETA.chunks, [_BETA.source])


def cid(n: int) -> ClaimId:
    return derive_claim_id("ans", n, f"claim {n}")


def reassessment(index: int, prior: ClaimStatus, new: ClaimStatus) -> ClaimReassessment:
    return ClaimReassessment.assess(cid(index), prior, new)


def make_record(
    claims: tuple[ClaimReassessment, ...],
    *,
    decision: CorrectionDecision = CorrectionDecision.CORRECT,
    supersedes: str | None = None,
) -> CorrectionRecord:
    return CorrectionRecord(
        answer_id="ans",
        decision=decision,
        claims=claims,
        contradictions=(),
        evaluator_name="structural-0.1",
        supersedes=supersedes,
    )


def test_a_reassessment_records_into_an_append_only_correction() -> None:
    prior = Claim(
        answer_id="ans",
        position=0,
        text="alpha claim",
        cited_evidence_ids=(CHUNK_A,),
        status=ClaimStatus.SUPPORTED,
    )

    result = reassess("ans", [prior], BETA_ONLY, requires_citation={prior.claim_id})
    record = record_reassessment(result, supersedes="cor_prior")

    assert record.decision is CorrectionDecision.QUALIFY
    assert record.claims == result.claims
    assert record.evaluator_name == "structural-0.1"
    assert record.is_revision is True


def test_record_id_is_independent_of_claim_order() -> None:
    forward = make_record(
        (
            reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED),
            reassessment(1, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED),
        )
    )
    reversed_ = make_record(forward.claims[::-1])

    assert forward.record_id == reversed_.record_id


def test_record_id_changes_when_the_decision_changes() -> None:
    claims = (reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED),)
    as_correct = make_record(claims, decision=CorrectionDecision.CORRECT)
    as_retain = make_record(claims, decision=CorrectionDecision.RETAIN)

    assert as_correct.record_id != as_retain.record_id


def test_record_id_is_derived_and_shaped() -> None:
    record = make_record((reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED),))
    body = record.record_id.removeprefix("cor_")

    assert record.record_id.startswith("cor_")
    assert len(body) == 16
    assert all(character in "0123456789abcdef" for character in body)


def test_a_record_only_claims_a_revision_when_it_superseded_and_changed() -> None:
    changed = make_record(
        (reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED),),
        supersedes="cor_prior",
    )
    unchanged_with_reference = make_record(
        (reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED),),
        decision=CorrectionDecision.RETAIN,
        supersedes="cor_prior",
    )
    changed_without_reference = make_record(
        (reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED),),
    )

    assert changed.is_revision is True
    # A retain that names a prior did not supersede anything; a change that names no prior is
    # not a revision of a recorded answer. Neither is reported as a revision.
    assert unchanged_with_reference.is_revision is False
    assert changed_without_reference.is_revision is False


def test_summary_is_a_structural_tally_not_rewritten_prose() -> None:
    record = make_record(
        (
            reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED),
            reassessment(1, ClaimStatus.SUPPORTED, ClaimStatus.UNSUPPORTED),
            reassessment(2, ClaimStatus.SUPPORTED, ClaimStatus.SUPPORTED),
        )
    )

    assert record.summary == "correct (retain=1, qualify=1, correct=1; 0 recorded conflict(s))"


def test_claims_with_returns_the_decided_identifiers_in_order() -> None:
    record = make_record(
        (
            reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED),
            reassessment(1, ClaimStatus.SUPPORTED, ClaimStatus.UNSUPPORTED),
        )
    )

    assert record.claims_with(ClaimDisposition.CORRECT) == (cid(0),)
    assert record.claims_with(ClaimDisposition.QUALIFY) == (cid(1),)
    assert record.claims_with(ClaimDisposition.RETAIN) == ()


def test_recorded_conflicts_are_carried_through_and_counted() -> None:
    conflict = Contradiction(cid(0), (CHUNK_A, _BETA.chunks[0].chunk_id))
    record = CorrectionRecord(
        answer_id="ans",
        decision=CorrectionDecision.CORRECT,
        claims=(reassessment(0, ClaimStatus.SUPPORTED, ClaimStatus.CONTRADICTED),),
        contradictions=(conflict,),
        evaluator_name="manual-0.1",
    )

    assert record.contradictions == (conflict,)
    assert "1 recorded conflict(s)" in record.summary
