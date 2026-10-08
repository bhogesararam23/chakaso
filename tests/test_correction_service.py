"""Tests for the application-level reassessment operation.

The operation reads a *stored* answer by identifier and re-grounds its recorded claims, so the
tests check the things that make it honest rather than convenient: a supported claim stays
supported when the evidence still backs it, qualifies when the evidence it relied on is gone, and
that the "required" set defaults to the claims the answer actually asserted as evidenced. A
reassessment of an unknown answer is refused, not silently a no-op, and a contradiction still
comes only from a supplied judgement — never inferred here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.answers import AnswerRecord, InMemoryAnswerStore, UnknownAnswerError, new_answer_id
from chakaso.claims.model import Claim, ClaimStatus
from chakaso.correction import CorrectionDecision, reassess_and_record, reassess_stored_answer
from chakaso.correction.record import CorrectionRecord
from chakaso.evidence import EvidencePack
from chakaso.grounding import ManualGroundingEvaluator
from chakaso.retrieval import ingest_text

NOW = datetime(2026, 10, 8, tzinfo=UTC)
_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
PACK = EvidencePack.of((*_ALPHA.chunks, *_BETA.chunks), [_ALPHA.source, _BETA.source])
BETA_ONLY = EvidencePack.of(_BETA.chunks, [_BETA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id
SOURCE_A = _ALPHA.source.source_id


def _prior_in(store: InMemoryAnswerStore) -> tuple[AnswerRecord, Claim]:
    answer_id = new_answer_id()
    claim = Claim(
        answer_id=str(answer_id),
        position=0,
        text="alpha claim",
        cited_evidence_ids=(CHUNK_A,),
        status=ClaimStatus.SUPPORTED,
    )
    record = AnswerRecord(
        answer_id=answer_id,
        conversation_id="conv_s",
        text="alpha claim",
        created_at=NOW,
        model_id="deterministic-double",
        model_is_development_double=True,
        claims=(claim,),
        evidence_source_ids=(SOURCE_A,),
        cited_source_ids=(SOURCE_A,),
    )
    store.save(record)
    return record, claim


def test_reassessment_keeps_a_supported_claim_supported_when_evidence_remains() -> None:
    store = InMemoryAnswerStore()
    record, _ = _prior_in(store)

    result = reassess_stored_answer(store, record.answer_id, PACK)

    assert result.decision is CorrectionDecision.RETAIN


def test_reassessment_qualifies_when_the_backing_evidence_is_gone() -> None:
    store = InMemoryAnswerStore()
    record, _ = _prior_in(store)

    # New evidence alone drops the chunk the claim relied on; the default required set (the
    # previously-supported claim) makes the lost support a qualification rather than silence.
    result = reassess_stored_answer(store, record.answer_id, BETA_ONLY)

    assert result.decision is CorrectionDecision.QUALIFY
    assert result.qualifications == (record.claims[0].claim_id,)


def test_required_set_defaults_to_previously_supported_claims() -> None:
    store = InMemoryAnswerStore()
    record, claim = _prior_in(store)

    # Explicitly requiring nothing means the lost citation reads as unevaluated, not unsupported,
    # so the answer is retained — proving the default (not this override) is what drove QUALIFY.
    result = reassess_stored_answer(
        store, record.answer_id, BETA_ONLY, requires_citation=frozenset()
    )

    assert result.claims[0].prior_status is ClaimStatus.SUPPORTED
    assert result.decision is CorrectionDecision.RETAIN
    assert claim.status is ClaimStatus.SUPPORTED


def test_a_supplied_judgement_is_the_only_route_to_a_stored_correction() -> None:
    store = InMemoryAnswerStore()
    record, claim = _prior_in(store)
    evaluator = ManualGroundingEvaluator({claim.claim_id: ClaimStatus.CONTRADICTED})

    result = reassess_stored_answer(store, record.answer_id, PACK, grounding_evaluator=evaluator)

    assert result.decision is CorrectionDecision.CORRECT


def test_reassessing_an_unknown_answer_is_refused() -> None:
    store = InMemoryAnswerStore()

    with pytest.raises(UnknownAnswerError, match="not recorded"):
        reassess_stored_answer(store, new_answer_id(), PACK)


def test_reassess_and_record_links_the_correction_to_the_prior_answer() -> None:
    store = InMemoryAnswerStore()
    record, _ = _prior_in(store)

    result, correction = reassess_and_record(store, record.answer_id, BETA_ONLY)

    assert isinstance(correction, CorrectionRecord)
    assert correction.supersedes == str(record.answer_id)
    assert correction.decision is result.decision
    assert correction.is_revision is True
