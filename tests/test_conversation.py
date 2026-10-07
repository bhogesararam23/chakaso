"""Tests for conversation state.

The properties that matter: history is append-only, an answer cannot cite evidence
it was not given, and the mapping to model messages is the only thing that reaches a
model.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from chakaso.conversation import Conversation, ConversationError, Turn
from chakaso.core.identifiers import SourceId
from chakaso.models import Message, Role

SESSION = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
FIRST = SourceId("src_1111111111111111")
SECOND = SourceId("src_2222222222222222")


def conversation() -> Conversation:
    return Conversation(conversation_id="conv_1")


def test_empty_conversation_has_no_turns() -> None:
    state = conversation()

    assert state.turns == ()
    assert state.last_turn is None
    assert state.last_user_turn is None
    assert state.previous_source_ids == ()
    assert state.to_model_messages() == ()


def test_a_conversation_requires_an_identifier() -> None:
    with pytest.raises(ConversationError, match="conversation_id"):
        Conversation(conversation_id="")


def test_user_turn_is_appended_without_mutating_the_original() -> None:
    state = conversation()

    updated = state.with_user_turn("What does the specification say?", at=SESSION)

    assert state.turns == ()
    assert len(updated.turns) == 1
    assert updated.last_turn is not None
    assert updated.last_turn.role is Role.USER
    assert updated.last_turn.is_from_user is True


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ConversationError, match="timezone-aware"):
        conversation().with_user_turn("hello", at=datetime(2026, 10, 7, 12, 0))


def test_a_turn_must_have_text() -> None:
    with pytest.raises(ConversationError, match="must have text"):
        conversation().with_user_turn("", at=SESSION)


def test_history_is_append_only() -> None:
    # An earlier answer has to remain in the record for a revision to be
    # explainable, so a turn that predates its predecessor is rejected.
    state = conversation().with_user_turn("first", at=SESSION)

    with pytest.raises(ConversationError, match="append-only"):
        state.with_user_turn("second", at=SESSION - timedelta(minutes=1))


def test_turns_at_the_same_instant_are_allowed() -> None:
    state = conversation().with_user_turn("first", at=SESSION)
    state = state.with_assistant_turn("second", at=SESSION)

    assert len(state.turns) == 2


def test_last_user_turn_skips_a_trailing_assistant_turn() -> None:
    state = (
        conversation()
        .with_user_turn("question", at=SESSION)
        .with_assistant_turn("answer", at=SESSION)
    )

    assert state.last_turn is not None
    assert state.last_turn.role is Role.ASSISTANT
    assert state.last_user_turn is not None
    assert state.last_user_turn.text == "question"


def test_assistant_turn_cannot_cite_evidence_it_was_not_given() -> None:
    # The same rule ADR-0003 enforces on generated text, enforced on the state that
    # records the answer.
    with pytest.raises(ConversationError, match="not in evidence_source_ids"):
        conversation().with_assistant_turn(
            "The window closes in April.", at=SESSION, cited_source_ids=(FIRST,)
        )


def test_assistant_turn_records_evidence_and_citations_separately() -> None:
    # "Which of those did you use?" is unanswerable if only the union is kept.
    state = conversation().with_assistant_turn(
        "The window closes in April.",
        at=SESSION,
        evidence_source_ids=(FIRST, SECOND),
        cited_source_ids=(FIRST,),
    )

    turn = state.turns[0]
    assert turn.evidence_source_ids == (FIRST, SECOND)
    assert turn.cited_source_ids == (FIRST,)


def test_previous_source_ids_are_most_recent_first_and_distinct() -> None:
    state = (
        conversation()
        .with_user_turn("first question", at=SESSION)
        .with_assistant_turn(
            "first answer", at=SESSION, evidence_source_ids=(FIRST,), cited_source_ids=(FIRST,)
        )
        .with_user_turn("follow-up", at=SESSION + timedelta(minutes=1))
        .with_assistant_turn(
            "second answer",
            at=SESSION + timedelta(minutes=1),
            evidence_source_ids=(SECOND,),
            cited_source_ids=(SECOND,),
        )
    )

    assert state.previous_source_ids == (SECOND, FIRST)


def test_to_model_messages_maps_role_and_text() -> None:
    state = (
        conversation()
        .with_user_turn("question", at=SESSION)
        .with_assistant_turn("answer", at=SESSION)
    )

    assert state.to_model_messages() == (
        Message(role=Role.USER, content="question"),
        Message(role=Role.ASSISTANT, content="answer"),
    )


def test_to_model_messages_keeps_only_the_most_recent_turns() -> None:
    state = conversation()
    for index in range(5):
        state = state.with_user_turn(f"turn {index}", at=SESSION + timedelta(minutes=index))

    assert [message.content for message in state.to_model_messages(max_turns=2)] == [
        "turn 3",
        "turn 4",
    ]


def test_to_model_messages_rejects_an_impossible_turn_count() -> None:
    with pytest.raises(ConversationError, match="max_turns"):
        conversation().to_model_messages(max_turns=0)


def test_model_messages_carry_no_conversation_provenance() -> None:
    # The mapping is the boundary: a model sees roles and text and nothing else.
    state = conversation().with_assistant_turn(
        "answer", at=SESSION, evidence_source_ids=(FIRST,), cited_source_ids=(FIRST,)
    )

    message = state.to_model_messages()[0]
    assert set(dataclasses.asdict(message)) == {"role", "content"}


def test_active_topic_can_be_set_and_cleared() -> None:
    state = conversation().with_active_topic("submission deadlines")

    assert state.active_topic == "submission deadlines"
    assert state.with_active_topic(None).active_topic is None


def test_blank_active_topic_is_rejected() -> None:
    with pytest.raises(ConversationError, match="active topic"):
        conversation().with_active_topic("   ")


def test_entities_are_merged_in_first_seen_order() -> None:
    state = conversation().with_entities("Chakaso", "BPE").with_entities("BPE", "FAISS")

    assert state.entities == ("Chakaso", "BPE", "FAISS")


def test_blank_entity_is_rejected() -> None:
    with pytest.raises(ConversationError, match="entity"):
        conversation().with_entities("Chakaso", "  ")


def test_open_questions_are_deduplicated() -> None:
    state = (
        conversation()
        .with_open_questions("Which deadline applies?")
        .with_open_questions("Which deadline applies?", "Is the source current?")
    )

    assert state.open_questions == ("Which deadline applies?", "Is the source current?")


def test_blank_open_question_is_rejected() -> None:
    with pytest.raises(ConversationError, match="open question"):
        conversation().with_open_questions("   ")


def test_resolving_an_unrecorded_question_is_not_an_error() -> None:
    # Nothing to do is not a failure; treating it as one would make callers guard
    # against a condition that has already happened.
    state = conversation().with_open_questions("Which deadline applies?")

    assert state.without_open_question("Never asked").open_questions == ("Which deadline applies?",)
    assert state.without_open_question("Which deadline applies?").open_questions == ()


def test_conversation_and_turn_are_frozen() -> None:
    state = conversation().with_user_turn("hello", at=SESSION)

    with pytest.raises(dataclasses.FrozenInstanceError):
        state.active_topic = "something"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.turns[0].text = "something else"  # type: ignore[misc]


def test_turn_defaults_to_no_evidence() -> None:
    turn = Turn(role=Role.USER, text="hello", created_at=SESSION)

    assert turn.evidence_source_ids == ()
    assert turn.cited_source_ids == ()
    assert turn.is_from_user is True
