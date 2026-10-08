"""Tests for wiring answer persistence into a conversation turn.

The record must be a faithful account of what the turn produced, and persistence must not
weaken the turn's atomicity: if saving fails, the conversation is left exactly as it was, with
no half-recorded answer and no adopted state. The development double is deliberately the model
here, to prove the record layer works unchanged on real (if fake) model output — no claim,
anywhere, that it understands an answer.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.answers import AnswerId, AnswerRecord, AnswerStoreError, InMemoryAnswerStore
from chakaso.claims.extract import SentenceClaimExtractor
from chakaso.conversation import Conversation, ConversationManager
from chakaso.evaluation import AnswerEvaluation
from chakaso.evidence import EvidencePack
from chakaso.models.deterministic import DeterministicModel
from chakaso.retrieval import ingest_text

SESSION = datetime(2026, 10, 8, tzinfo=UTC)
_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
PACK = EvidencePack.of(_ALPHA.chunks, [_ALPHA.source])


def build_manager(
    store: InMemoryAnswerStore | None = None,
    extractor: SentenceClaimExtractor | None = None,
) -> ConversationManager:
    return ConversationManager(
        DeterministicModel(),
        Conversation(conversation_id="conv_ar"),
        now=lambda: SESSION,
        answer_store=store,
        claim_extractor=extractor,
    )


class _FailingStore:
    """A store whose save always raises, to exercise the atomicity guarantee."""

    def save(self, record: AnswerRecord) -> None:
        message = "storage is unavailable"
        raise AnswerStoreError(message)

    def get(self, answer_id: AnswerId) -> AnswerRecord | None:
        return None

    def list(self, *, conversation_id: str | None = None) -> tuple[AnswerRecord, ...]:
        return ()

    def history(self, answer_id: AnswerId) -> tuple[AnswerRecord, ...]:
        return ()


def test_a_turn_with_a_store_records_the_answer_faithfully() -> None:
    store = InMemoryAnswerStore()
    manager = build_manager(store, SentenceClaimExtractor())

    reply = manager.send("hello", evidence=PACK)
    record = reply.answer_record

    assert isinstance(record, AnswerRecord)
    assert store.get(record.answer_id) is record
    assert record.text == reply.text
    assert record.conversation_id == "conv_ar"
    assert record.model_id == reply.generation.model_id
    assert record.model_is_development_double is True
    assert record.turn_index == len(reply.conversation.turns) - 1
    assert record.created_at == reply.turn.created_at
    assert record.evidence_source_ids == tuple(PACK.sources)
    assert record.provenance["model_display_name"]


def test_the_record_evaluates_claims_under_the_answer_it_records() -> None:
    store = InMemoryAnswerStore()
    manager = build_manager(store, SentenceClaimExtractor())

    record = manager.send("hello", evidence=PACK).answer_record
    assert record is not None

    evaluation = record.evaluation
    assert isinstance(evaluation, AnswerEvaluation)
    assert evaluation.answer_id == str(record.answer_id)
    # every extracted claim is identified with the answer it is recorded under
    assert all(claim.answer_id == str(record.answer_id) for claim in record.claims)


def test_without_a_store_the_turn_is_unchanged_and_unrecorded() -> None:
    reply = build_manager().send("hello")

    assert reply.answer_record is None
    assert len(reply.conversation.turns) == 2


def test_a_persistence_failure_aborts_the_turn_with_the_conversation_intact() -> None:
    manager = build_manager(_FailingStore(), SentenceClaimExtractor())  # type: ignore[arg-type]
    before = manager.conversation

    with pytest.raises(AnswerStoreError):
        manager.send("hello", evidence=PACK)

    # The live conversation was never swapped, and nothing was recorded: no half-completed turn.
    assert manager.conversation is before
    assert before.turns == ()


def test_store_is_used_only_when_supplied_but_extractor_alone_is_harmless() -> None:
    manager = build_manager(None, SentenceClaimExtractor())

    assert manager.send("hello").answer_record is None
