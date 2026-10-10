"""Tests for the answer-record serialization codec.

Durable storage is only honest if a reload equals what was saved, so the central test is the
round-trip: identifier, text, model, claims, every reference id, metadata and the correction link
must survive unchanged. Two behaviours are the design, not accidents, and are tested explicitly:
the in-memory ``evaluation`` is *not* persisted (it is recomputed provenance, not history), and a
corrupted stored document raises rather than producing a plausible-but-wrong record.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from chakaso.answers import AnswerId, AnswerRecord, AnswerSerializationError, new_answer_id
from chakaso.answers.serialize import decode_document, decode_json, encode_document, encode_json
from chakaso.claims.model import Claim, ClaimStatus
from chakaso.evaluation import evaluate_answer
from chakaso.evidence import EvidencePack
from chakaso.retrieval import ingest_text

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
PACK = EvidencePack.of(_ALPHA.chunks, [_ALPHA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id
SOURCE_A = _ALPHA.source.source_id


def _record(*, correction_of: AnswerId | None = None) -> AnswerRecord:
    answer_id = new_answer_id()
    claim = Claim(
        answer_id=str(answer_id),
        position=0,
        text="Alpha is a fact.",
        cited_evidence_ids=(CHUNK_A,),
        status=ClaimStatus.SUPPORTED,
    )
    return AnswerRecord(
        answer_id=answer_id,
        conversation_id="conv_1",
        text="Alpha is a fact.",
        created_at=NOW,
        model_id="deterministic",
        model_is_development_double=True,
        claims=(claim,),
        evidence_source_ids=(SOURCE_A,),
        cited_source_ids=(SOURCE_A,),
        cited_chunk_ids=(CHUNK_A,),
        retrieval_metadata={"retriever": "lexical"},
        turn_index=1,
        correction_of=correction_of,
        provenance={"note": "value"},
    )


def test_round_trip_reconstructs_every_persisted_field() -> None:
    record = _record()

    loaded = decode_json(encode_json(record))

    assert loaded == record


def test_evaluation_is_not_persisted_but_the_answer_is() -> None:
    record = _record()
    with_evaluation = replace(
        record, evaluation=evaluate_answer(str(record.answer_id), record.claims, PACK)
    )
    assert with_evaluation.evaluation is not None

    loaded = decode_json(encode_json(with_evaluation))

    # The answer's substance survives; its evaluation is a recomputed view, deliberately not
    # frozen into durable history.
    assert loaded.evaluation is None
    assert loaded.answer_id == with_evaluation.answer_id
    assert loaded.claims == with_evaluation.claims
    assert loaded.cited_chunk_ids == with_evaluation.cited_chunk_ids


def test_correction_link_round_trips_and_preserves_chain() -> None:
    prior = _record()
    revised = _record(correction_of=prior.answer_id)

    loaded = decode_document(encode_document(revised))

    assert loaded.correction_of == prior.answer_id
    assert loaded.is_correction is True


def test_malformed_json_is_rejected_not_guessed() -> None:
    with pytest.raises(AnswerSerializationError, match="not valid JSON"):
        decode_json("{ this is not json ")


def test_missing_required_field_is_rejected() -> None:
    document = encode_document(_record())
    del document["text"]

    with pytest.raises(AnswerSerializationError, match="missing 'text'"):
        decode_document(document)


def test_malformed_identifier_is_rejected_by_its_owner() -> None:
    document = encode_document(_record())
    document["answer_id"] = "ans_not-hex"

    with pytest.raises(AnswerSerializationError):
        decode_document(document)


def test_a_boolean_is_not_mistaken_for_a_turn_index() -> None:
    document = encode_document(_record())
    document["turn_index"] = True  # bool is a subclass of int; must not be accepted as an index

    loaded = decode_document(document)

    assert loaded.turn_index is None
