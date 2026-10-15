"""Tests for the composed, explicit reassessment wired into the application (P21).

These exercise the *composition*, not the reassessment boundary itself (that is covered by
``test_correction_service``). What is under test is the front edge a conversation now reaches: a
follow-up that names a stored answer is recognised deterministically and drives ``reassess_and_record``
against the evidence the follow-up makes available, while an ordinary turn, an unknown target, and an
ambiguous reference are each handled by the existing typed contract — never by a guess. The original
answer and its provenance are never rewritten, and no revised prose is fabricated (there is no model).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from chakaso.answers import (
    AnswerRecord,
    InMemoryAnswerStore,
    SQLiteAnswerStore,
    UnknownAnswerError,
    new_answer_id,
)
from chakaso.app import ReassessmentOutcome, build_application
from chakaso.claims.model import Claim, ClaimStatus
from chakaso.config import load_config
from chakaso.correction import CorrectionDecision, CorrectionRecord, Reassessment
from chakaso.correction.errors import AmbiguousReassessmentTargetError
from chakaso.retrieval import Corpus, ingest_text

NOW = datetime(2026, 10, 15, tzinfo=UTC)
_DOC = ingest_text("Chakaso grounds answers in retrieved evidence.", label="spec")
_CHUNK = _DOC.chunks[0].chunk_id
_SOURCE = _DOC.source.source_id


def _populated_corpus() -> Corpus:
    corpus = Corpus()
    corpus.add_document(_DOC)
    return corpus


def _empty_corpus() -> Corpus:
    return Corpus()


def _prior_record(*, supported: bool = True) -> AnswerRecord:
    answer_id = new_answer_id()
    claim = Claim(
        answer_id=str(answer_id),
        position=0,
        text="Chakaso grounds answers in retrieved evidence.",
        cited_evidence_ids=(_CHUNK,),
        status=ClaimStatus.SUPPORTED if supported else ClaimStatus.UNSUPPORTED,
    )
    return AnswerRecord(
        answer_id=answer_id,
        conversation_id="conv_p21",
        text="Chakaso grounds answers in retrieved evidence.",
        created_at=NOW,
        model_id="deterministic-double",
        model_is_development_double=True,
        claims=(claim,),
        evidence_source_ids=(_SOURCE,),
        cited_source_ids=(_SOURCE,),
    )


def _store_with(record: AnswerRecord) -> InMemoryAnswerStore:
    store = InMemoryAnswerStore()
    store.save(record)
    return store


# -- a valid, explicitly named target triggers reassessment -------------------


def test_naming_a_stored_answer_triggers_a_reassessment_that_stands() -> None:
    record = _prior_record()
    app = build_application(load_config(), corpus=_populated_corpus(), store=_store_with(record))

    outcome = app.reassess_followup(f"please reconsider {record.answer_id}")

    assert isinstance(outcome, ReassessmentOutcome)
    assert outcome is not None
    assert outcome.target == record.answer_id
    # The cited chunk is still in the corpus, so the previously-supported claim stays supported.
    assert outcome.reassessment.decision is CorrectionDecision.RETAIN
    assert isinstance(outcome.reassessment, Reassessment)
    assert isinstance(outcome.record, CorrectionRecord)
    assert outcome.record.supersedes == str(record.answer_id)
    assert outcome.standing_chunk_count == 1


# -- an unknown but valid target is refused deterministically -----------------


def test_naming_an_unrecorded_answer_is_refused_not_silently_ignored() -> None:
    app = build_application(load_config(), corpus=_populated_corpus(), store=InMemoryAnswerStore())
    missing = new_answer_id()

    with pytest.raises(UnknownAnswerError, match="not recorded"):
        app.reassess_followup(f"reconsider {missing}")


# -- an ordinary follow-up does not trigger, and does not mutate --------------


def test_an_ordinary_follow_up_is_not_reassessment_and_changes_nothing() -> None:
    record = _prior_record()
    store = _store_with(record)
    app = build_application(load_config(), corpus=_populated_corpus(), store=store)

    outcome = app.reassess_followup("what else should I know about it?")

    assert outcome is None
    assert len(store) == 1  # nothing recorded, nothing rewritten


# -- an ambiguous reference never guesses -------------------------------------


def test_two_named_answers_are_refused_as_ambiguous_without_a_guess() -> None:
    first = _prior_record()
    second = _prior_record()
    store = _store_with(first)
    store.save(second)
    app = build_application(load_config(), corpus=_populated_corpus(), store=store)

    with pytest.raises(AmbiguousReassessmentTargetError, match="exactly one"):
        app.reassess_followup(f"compare {first.answer_id} with {second.answer_id}")


# -- the original answer is never mutated; the event is distinct and append-only


def test_reassessment_leaves_the_original_answer_and_history_untouched() -> None:
    record = _prior_record()
    store = _store_with(record)
    app = build_application(load_config(), corpus=_populated_corpus(), store=store)

    outcome = app.reassess_followup(f"reassess {record.answer_id}")

    assert outcome is not None
    # A distinct event: a content-identified correction record naming the target as superseded.
    assert outcome.record.record_id != str(record.answer_id)
    assert outcome.record.answer_id == str(record.answer_id)
    # The store is unchanged: the same single answer, byte-identical, still under the same id.
    assert len(store) == 1
    assert store.get(record.answer_id) == record
    assert store.get(record.answer_id).claims == record.claims
    assert store.get(record.answer_id).evaluation_provenance == record.evaluation_provenance


def test_reassessment_produces_no_revised_answer_text() -> None:
    record = _prior_record()
    store = _store_with(record)
    app = build_application(load_config(), corpus=_populated_corpus(), store=store)

    outcome = app.reassess_followup(f"reassess {record.answer_id}")

    assert outcome is not None
    # The correction summary is a structural tally, and no new answer record was created with
    # rewritten prose — the original text is the only answer text the store holds.
    assert outcome.record.summary.startswith(outcome.record.decision.value)
    assert [r.text for r in store.list()] == [record.text]


# -- withdrawn support is reflected honestly ----------------------------------


def test_reassessment_against_a_corpus_without_the_evidence_loses_support() -> None:
    record = _prior_record()
    # The durable store still names the answer, but the corpus no longer holds its evidence, so
    # the previously-supported claim can no longer be re-grounded — honestly a qualification.
    app = build_application(load_config(), corpus=_empty_corpus(), store=_store_with(record))

    outcome = app.reassess_followup(f"reconsider {record.answer_id}")

    assert outcome is not None
    assert outcome.decision.triggers_reassessment is True
    assert outcome.reassessment.decision is CorrectionDecision.QUALIFY


# -- failure leaves no partial state ------------------------------------------


def test_unknown_target_failure_leaves_the_store_unchanged() -> None:
    record = _prior_record()
    store = _store_with(record)
    app = build_application(load_config(), corpus=_populated_corpus(), store=store)

    with pytest.raises(UnknownAnswerError):
        app.reassess_followup(f"reconsider {new_answer_id()}")

    assert len(store) == 1  # a refused reassessment wrote nothing


# -- durable backend: same semantics, survives restart ------------------------


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_reassessment_over_sqlite_is_consistent_and_persists_the_untouched_answer(
    tmp_path: Path,
) -> None:
    db = tmp_path / "answers.db"
    record = _prior_record()
    with SQLiteAnswerStore(db, journal_mode="wal", initialize=True) as store:
        store.save(record)

    # The application opens the configured durable store itself, so the reassessment reads the same
    # persisted answer a real restart would present and closes cleanly afterwards.
    loaded = load_config(
        _write(tmp_path, f'[storage]\nbackend = "sqlite"\npath = "{db.as_posix()}"\n')
    )
    app = build_application(loaded, corpus=_populated_corpus())

    outcome = app.reassess_followup(f"reconsider {record.answer_id}")
    app.close()

    assert outcome is not None
    # In-memory and durable backends give the same decision for the same evidence (conformance).
    assert outcome.reassessment.decision is CorrectionDecision.RETAIN

    # A reopen sees the original answer exactly as written — reassessment never mutated history.
    with SQLiteAnswerStore(db, initialize=False) as reopened:
        assert len(reopened.list()) == 1
        restored = reopened.get(record.answer_id)
        assert restored is not None
        assert restored.text == record.text
        assert restored.claims == record.claims


def test_applications_turn_still_conducts_an_ordinary_ground_answer() -> None:
    # Backward compatibility: routing reassessment through the application does not change an
    # ordinary conversational turn, which still records its answer.
    app = build_application(load_config(), corpus=_populated_corpus())

    turn = app.turn("What is Chakaso?")

    assert turn.plan is not None
    assert app.store.list()
