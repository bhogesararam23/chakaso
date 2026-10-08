"""Tests for the answer record.

An ``AnswerRecord`` ties the layers together, so the tests check both that it carries the
connective fields faithfully and that it refuses the inconsistent states that would later make
a correction history untrustworthy: citing evidence it was not given (ADR-0003), carrying a claim
that belongs to another answer, correcting itself, or timestamping a naive time.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from chakaso.answers import AnswerId, AnswerRecord, InvalidAnswerError, new_answer_id
from chakaso.claims.model import Claim
from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.evaluation import evaluate_answer
from chakaso.evidence import EvidencePack
from chakaso.retrieval import ingest_text

NOW = datetime(2026, 10, 8, tzinfo=UTC)
_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
PACK = EvidencePack.of(_ALPHA.chunks, [_ALPHA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id
SOURCE_A = _ALPHA.source.source_id


def src(n: int) -> SourceId:
    return SourceId(f"src_{n:016x}")


def chk(n: int) -> ChunkId:
    return ChunkId(f"chk_{n:016x}")


def make_record(**overrides: object) -> AnswerRecord:
    base: dict[str, object] = {
        "answer_id": new_answer_id(),
        "conversation_id": "conv_1",
        "text": "An answer.",
        "created_at": NOW,
        "model_id": "deterministic-double",
        "model_is_development_double": True,
    }
    base.update(overrides)
    return AnswerRecord(**base)  # type: ignore[arg-type]


def test_a_valid_record_retains_its_fields_and_deduplicates_identifiers() -> None:
    record = make_record(
        evidence_source_ids=(src(1), src(1), src(2)),
        cited_source_ids=(src(1), src(1)),
        cited_chunk_ids=(chk(3), chk(3)),
    )

    assert record.evidence_source_ids == (src(1), src(2))
    assert record.cited_source_ids == (src(1),)
    assert record.cited_chunk_ids == (chk(3),)
    assert record.is_correction is False
    assert record.claim_count == 0


def test_a_record_can_supersede_a_different_answer() -> None:
    prior = new_answer_id()
    record = make_record(correction_of=prior)

    assert record.is_correction is True
    assert record.correction_of == prior


def test_an_answer_cannot_cite_evidence_it_was_not_given() -> None:
    with pytest.raises(InvalidAnswerError, match="not given as evidence"):
        make_record(evidence_source_ids=(src(1),), cited_source_ids=(src(2),))


def test_a_self_correcting_answer_is_refused() -> None:
    answer_id = new_answer_id()
    with pytest.raises(InvalidAnswerError, match="cannot correct itself"):
        make_record(answer_id=answer_id, correction_of=answer_id)


def test_a_claim_must_belong_to_the_answer_that_records_it() -> None:
    record_id = new_answer_id()
    foreign = Claim(answer_id=str(new_answer_id()), position=0, text="belongs elsewhere")
    with pytest.raises(InvalidAnswerError, match="is recorded under"):
        make_record(answer_id=record_id, claims=(foreign,))


@pytest.mark.parametrize(
    "overrides",
    [
        {"conversation_id": "  "},
        {"text": ""},
        {"model_id": "   "},
    ],
)
def test_missing_required_provenance_is_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidAnswerError):
        make_record(**overrides)


def test_a_naive_timestamp_is_refused() -> None:
    with pytest.raises(InvalidAnswerError, match="timezone-aware"):
        make_record(created_at=datetime(2026, 10, 8))


def test_an_evaluation_result_can_be_attached_to_the_record() -> None:
    answer_id = new_answer_id()
    claim = Claim(
        answer_id=str(answer_id),
        position=0,
        text="alpha claim",
        cited_evidence_ids=(CHUNK_A,),
    )
    evaluation = evaluate_answer(str(answer_id), [claim], PACK)

    record = make_record(
        answer_id=answer_id,
        evidence_source_ids=(SOURCE_A,),
        cited_source_ids=(SOURCE_A,),
        claims=(claim,),
        evaluation=evaluation,
    )

    assert record.evaluation is evaluation
    assert record.claim_count == 1


def test_records_are_immutable() -> None:
    record = make_record()
    with pytest.raises(FrozenInstanceError):
        record.text = "changed"  # type: ignore[misc]


def test_answer_id_is_a_typed_identifier() -> None:
    assert isinstance(make_record().answer_id, AnswerId)
