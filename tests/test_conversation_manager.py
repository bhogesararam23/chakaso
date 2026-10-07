"""Tests for the conversation manager.

The properties under test are the ones a later component will depend on: a turn
either completes or nothing changes, follow-ups carry the recent context, the model
is reached only through the boundary, and a reference outside the supplied evidence
is never resolved to a source.

The manager is exercised against small purpose-built models rather than the
deterministic double, because a test that asserts on the double's fixed output is
testing the double. The double appears once, at the end, as an integration check
that the real wiring works.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from chakaso.conversation import (
    CONVERSATION_ID_PREFIX,
    Conversation,
    ConversationManager,
    ConversationManagerError,
    InvalidGenerationError,
    InvalidUserInputError,
    Reply,
    new_conversation_id,
)
from chakaso.evidence import EvidenceChunk, EvidencePack, SourceRecord
from chakaso.models import (
    DETERMINISTIC_MODEL_ID,
    DeterministicModel,
    FinishReason,
    GenerationParams,
    GenerationResult,
    LanguageModel,
    Message,
    ModelMetadata,
    Role,
)
from chakaso.models.errors import ModelError, UnsupportedCapabilityError

SESSION = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


class RecordingModel:
    """A model that records what it was asked and returns a fixed answer.

    Returns the same text every time unless ``answers`` is supplied, so that tests
    can assert on the request the manager built rather than on the reply.
    """

    def __init__(
        self,
        *,
        answer: str = "A recorded answer.",
        model_id: str = "recording",
        finish_reason: FinishReason = FinishReason.STOP,
        answers: Sequence[str] | None = None,
    ) -> None:
        self._metadata = ModelMetadata(
            model_id=model_id, display_name="Recording test model", context_window=1024
        )
        self._answer = answer
        self._finish_reason = finish_reason
        self._answers = list(answers) if answers is not None else None
        self.requests: list[tuple[tuple[Message, ...], GenerationParams]] = []

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        self.requests.append((tuple(messages), params))
        text = self._answers.pop(0) if self._answers else self._answer
        return GenerationResult(
            text=text, model_id=self._metadata.model_id, finish_reason=self._finish_reason
        )

    @property
    def last_messages(self) -> tuple[Message, ...]:
        return self.requests[-1][0]


class FailingModel(RecordingModel):
    """A model whose generation fails."""

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        self.requests.append((tuple(messages), params))
        message = "the model could not be reached"
        raise ModelError(message)


class MisattributingModel(RecordingModel):
    """A model that attributes its output to something else."""

    def __init__(self) -> None:
        super().__init__(model_id="declared")
        self._metadata = ModelMetadata(
            model_id="declared", display_name="Misattributing", context_window=1024
        )

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        result = super().generate(messages, params)
        return GenerationResult(
            text=result.text, model_id="someone-else", finish_reason=result.finish_reason
        )


def clock(*moments: datetime):
    """Return a clock that yields ``moments`` in order, repeating the last."""
    remaining = list(moments)

    def now() -> datetime:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return now


def manager(
    model: LanguageModel | None = None,
    *,
    conversation: Conversation | None = None,
    params: GenerationParams | None = None,
    max_context_turns: int | None = None,
    now=None,
) -> ConversationManager:
    return ConversationManager(
        model if model is not None else RecordingModel(),
        conversation if conversation is not None else Conversation(conversation_id="conv_test"),
        params=params,
        max_context_turns=max_context_turns,
        now=now if now is not None else clock(SESSION),
    )


def source_record(text: str = "The window closes on 14 April.") -> SourceRecord:
    return SourceRecord.create(
        url="https://example.org/specification",
        title="Specification",
        content=text,
        retrieved_at=SESSION,
    )


def pack_with(source: SourceRecord) -> tuple[EvidencePack, EvidenceChunk]:
    chunk = EvidenceChunk.create(
        source_id=source.source_id, text="The window closes on 14 April.", position=0
    )
    return EvidencePack.of([chunk], [source]), chunk


# ---------------------------------------------------------------------------
# Construction and dependency injection
# ---------------------------------------------------------------------------


def test_construction_starts_from_the_given_conversation() -> None:
    conversation = Conversation(conversation_id="conv_abc")
    instance = manager(conversation=conversation)

    assert instance.conversation is conversation
    assert instance.conversation.conversation_id == "conv_abc"


def test_construction_rejects_an_impossible_context_bound() -> None:
    with pytest.raises(ConversationManagerError, match="max_context_turns"):
        manager(max_context_turns=0)


def test_context_bound_is_exposed() -> None:
    assert manager(max_context_turns=5).max_context_turns == 5
    assert manager().max_context_turns is None


def test_model_metadata_is_exposed_for_reporting() -> None:
    # A front end has to be able to say what produced an answer, including that it
    # was a development double.
    instance = manager()
    assert instance.model_metadata.model_id == "recording"
    assert instance.model_metadata.development_double is False


def test_any_language_model_implementation_is_accepted() -> None:
    # Dependency injection, structurally: the manager is satisfied by anything with
    # the two required members and has no knowledge of a concrete adapter.
    class Minimal:
        @property
        def metadata(self) -> ModelMetadata:
            return ModelMetadata(model_id="minimal", display_name="Minimal", context_window=8)

        def generate(
            self, messages: Sequence[Message], params: GenerationParams
        ) -> GenerationResult:
            return GenerationResult(text="minimal answer", model_id="minimal")

    instance = ConversationManager(Minimal(), Conversation(conversation_id="conv_min"))

    assert instance.send("hello").text == "minimal answer"


# ---------------------------------------------------------------------------
# Turn semantics
# ---------------------------------------------------------------------------


def test_first_turn_records_a_user_and_an_assistant_turn() -> None:
    instance = manager()

    reply = instance.send("What does the specification say?")

    assert isinstance(reply, Reply)
    assert [turn.role for turn in reply.conversation.turns] == [Role.USER, Role.ASSISTANT]
    assert reply.conversation.turns[0].text == "What does the specification say?"
    assert reply.text == reply.turn.text
    assert reply.turn.role is Role.ASSISTANT


def test_reply_carries_the_generation_result() -> None:
    reply = manager().send("hello")

    assert reply.generation.model_id == "recording"
    assert reply.is_truncated is False


def test_truncated_generation_is_reported() -> None:
    # A caller that cannot tell a cut-off answer from a complete one will present
    # one as the other.
    instance = manager(RecordingModel(finish_reason=FinishReason.LENGTH))

    assert instance.send("hello").is_truncated is True


def test_follow_up_turn_receives_the_previous_turns() -> None:
    model = RecordingModel(answers=["The window closes on 14 April.", "It closes at 23:59 UTC."])
    instance = manager(model)

    instance.send("When does the window close?")
    second = instance.send("And at what time?")

    assert [message.content for message in model.last_messages] == [
        "When does the window close?",
        "The window closes on 14 April.",
        "And at what time?",
    ]
    assert [turn.role for turn in second.conversation.turns] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
    ]


def test_multiple_turns_accumulate_in_order() -> None:
    instance = manager(RecordingModel(answers=["a", "b", "c"]))

    for question in ("one", "two", "three"):
        instance.send(question)

    assert [turn.text for turn in instance.conversation.turns if turn.is_from_user] == [
        "one",
        "two",
        "three",
    ]


def test_user_input_is_stripped_before_becoming_a_turn() -> None:
    reply = manager().send("   hello   ")

    assert reply.conversation.turns[0].text == "hello"


def test_generated_text_is_not_rewritten() -> None:
    # Provenance: what is recorded is what the model produced. A caller that wants
    # it trimmed for display can trim it.
    instance = manager(RecordingModel(answer="  spaced answer  "))

    assert instance.send("hello").text == "  spaced answer  "


def test_the_request_carries_the_configured_generation_parameters() -> None:
    model = RecordingModel()
    params = GenerationParams(max_new_tokens=17, temperature=0.5)
    instance = manager(model, params=params)

    instance.send("hello")

    assert model.requests[-1][1] == params


# ---------------------------------------------------------------------------
# Context projection
# ---------------------------------------------------------------------------


def test_context_projection_limits_what_the_model_sees() -> None:
    model = RecordingModel(answers=["a1", "a2", "a3"])
    instance = manager(model, max_context_turns=4)

    for question in ("q1", "q2", "q3"):
        instance.send(question)

    # Three turns completed before the third request, so six turns exist and only
    # the most recent four are projected.
    assert [message.content for message in model.last_messages] == ["a1", "q2", "a2", "q3"]


def test_context_projection_is_unbounded_by_default() -> None:
    model = RecordingModel(answers=["a1", "a2", "a3", "a4"])
    instance = manager(model)

    for question in ("q1", "q2", "q3", "q4"):
        instance.send(question)

    assert len(model.last_messages) == 7


def test_a_single_turn_bound_still_projects_the_current_message() -> None:
    # The message being answered must never be projected away; a bound that dropped
    # it would ask the model to answer nothing.
    model = RecordingModel()
    instance = manager(model, max_context_turns=1)

    instance.send("only message")

    assert [message.content for message in model.last_messages] == ["only message"]


# ---------------------------------------------------------------------------
# Immutability and failure isolation
# ---------------------------------------------------------------------------


def test_previous_state_is_preserved() -> None:
    instance = manager()
    before = instance.conversation

    instance.send("hello")

    assert before.turns == ()
    assert len(instance.conversation.turns) == 2
    assert instance.conversation is not before


def test_a_failed_model_call_leaves_the_conversation_unchanged() -> None:
    # A turn either completes or nothing changes. A user turn recorded without a
    # reply would be indistinguishable from one that is still being answered.
    instance = manager(FailingModel())
    before = instance.conversation

    with pytest.raises(ModelError):
        instance.send("hello")

    assert instance.conversation is before
    assert instance.conversation.turns == ()


def test_an_empty_generation_leaves_the_conversation_unchanged() -> None:
    instance = manager(RecordingModel(answer="   "))

    with pytest.raises(InvalidGenerationError, match="empty response"):
        instance.send("hello")

    assert instance.conversation.turns == ()


def test_a_misattributed_generation_leaves_the_conversation_unchanged() -> None:
    instance = manager(MisattributingModel())

    with pytest.raises(InvalidGenerationError, match="attributed to"):
        instance.send("hello")

    assert instance.conversation.turns == ()


def test_invalid_user_input_leaves_the_conversation_unchanged() -> None:
    instance = manager()

    with pytest.raises(InvalidUserInputError):
        instance.send("   ")

    assert instance.conversation.turns == ()


def test_a_failure_does_not_advance_the_conversation_for_the_next_turn() -> None:
    # After a failure the conversation must be exactly as it was, so a retry is a
    # first turn rather than a follow-up to a turn that never happened.
    model = RecordingModel(answers=["first answer", "second answer"])
    instance = manager(model)

    instance.send("first question")
    state_after_first = instance.conversation

    with pytest.raises(ModelError):
        ConversationManager(FailingModel(), state_after_first, now=clock(SESSION)).send("doomed")

    reply = instance.send("second question")

    assert [message.content for message in model.last_messages] == [
        "first question",
        "first answer",
        "second question",
    ]
    assert reply.conversation.turns[2].text == "second question"


def test_model_failures_are_not_translated() -> None:
    # Failures must stay distinguishable. Wrapping a capability failure in a
    # generic conversation error would make a defect look like a caller mistake.
    class Incapable(RecordingModel):
        def generate(
            self, messages: Sequence[Message], params: GenerationParams
        ) -> GenerationResult:
            message = "this model cannot do that"
            raise UnsupportedCapabilityError(message)

    with pytest.raises(UnsupportedCapabilityError):
        manager(Incapable()).send("hello")


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def test_turns_are_timestamped_in_order() -> None:
    instance = manager(now=clock(SESSION, SESSION + timedelta(seconds=5)))

    reply = instance.send("hello")

    assert reply.conversation.turns[0].created_at == SESSION
    assert reply.conversation.turns[1].created_at == SESSION + timedelta(seconds=5)


def test_a_clock_that_moves_backwards_does_not_break_append_only_history() -> None:
    # An NTP correction must not turn into a confusing state error, and must not
    # produce a turn that predates its predecessor.
    instance = manager(now=clock(SESSION, SESSION - timedelta(hours=1)))

    reply = instance.send("hello")

    assert reply.conversation.turns[0].created_at == SESSION
    assert reply.conversation.turns[1].created_at >= SESSION


# ---------------------------------------------------------------------------
# Evidence and citation provenance
# ---------------------------------------------------------------------------


def test_without_evidence_a_cited_identifier_is_unresolved() -> None:
    # The rule ADR-0003 exists to enforce: an identifier the model was not given is
    # never resolved to a source.
    invented = "src_0123456789abcdef"
    instance = manager(RecordingModel(answer=f"The deadline is in March [{invented}]."))

    reply = instance.send("When is the deadline?")

    assert reply.citation_resolution.citations == ()
    assert reply.citation_resolution.unknown_identifiers == (invented,)
    assert reply.has_unresolved_references is True
    assert reply.turn.cited_source_ids == ()
    assert reply.turn.evidence_source_ids == ()


def test_a_reference_to_supplied_evidence_resolves() -> None:
    source = source_record()
    pack, chunk = pack_with(source)
    instance = manager(RecordingModel(answer=f"The window closes on 14 April [{chunk.chunk_id}]."))

    reply = instance.send("When does the window close?", evidence=pack)

    assert reply.has_unresolved_references is False
    assert len(reply.citation_resolution.citations) == 1
    assert reply.citation_resolution.citations[0].source == source
    assert reply.turn.cited_source_ids == (source.source_id,)


def test_supplied_evidence_is_recorded_on_the_turn_even_when_uncited() -> None:
    # "Which of those did you use?" is unanswerable if only the union is kept.
    source = source_record()
    pack, _chunk = pack_with(source)
    instance = manager(RecordingModel(answer="The window closes on 14 April."))

    reply = instance.send("When does the window close?", evidence=pack)

    assert reply.turn.evidence_source_ids == (source.source_id,)
    assert reply.turn.cited_source_ids == ()


def test_evidence_from_an_earlier_turn_does_not_leak_into_a_later_one() -> None:
    # A citation must be resolvable only against the evidence supplied for the turn
    # that produced it, not against anything the conversation has ever seen.
    source = source_record()
    pack, chunk = pack_with(source)
    model = RecordingModel(answers=[f"Answer [{chunk.chunk_id}].", f"Answer [{chunk.chunk_id}]."])
    instance = manager(model)

    instance.send("first", evidence=pack)
    second = instance.send("second")

    assert second.turn.cited_source_ids == ()
    assert second.citation_resolution.unknown_identifiers == (str(chunk.chunk_id),)


def test_a_malformed_reference_is_reported_separately_from_a_fabricated_one() -> None:
    instance = manager(RecordingModel(answer="See [src_0123] and [src_ffffffffffffffff]."))

    reply = instance.send("hello")

    assert reply.citation_resolution.malformed_identifiers == ("src_0123",)
    assert reply.citation_resolution.unknown_identifiers == ("src_ffffffffffffffff",)


def test_evidence_pack_sources_are_recorded_in_pack_order() -> None:
    first = source_record("first body")
    second = source_record("second body")
    chunks = [
        EvidenceChunk.create(source_id=first.source_id, text="first body", position=0),
        EvidenceChunk.create(source_id=second.source_id, text="second body", position=0),
    ]
    pack = EvidencePack.of(chunks, [first, second])
    instance = manager()

    reply = instance.send("hello", evidence=pack)

    assert reply.turn.evidence_source_ids == (first.source_id, second.source_id)


# ---------------------------------------------------------------------------
# Conversation identity helpers
# ---------------------------------------------------------------------------


def test_new_conversation_id_is_prefixed_and_unique() -> None:
    identifiers = {new_conversation_id() for _ in range(50)}

    assert len(identifiers) == 50
    assert all(identifier.startswith(CONVERSATION_ID_PREFIX) for identifier in identifiers)


def test_new_conversation_id_is_accepted_by_the_state_type() -> None:
    assert Conversation(conversation_id=new_conversation_id()).conversation_id


# ---------------------------------------------------------------------------
# Integration with the deterministic development double
# ---------------------------------------------------------------------------


def test_the_development_double_can_hold_a_conversation() -> None:
    # The end-to-end wiring check. The answer is the double's fixed text, and the
    # test asserts on the plumbing, not on the text being an answer to anything.
    double = DeterministicModel()
    instance = ConversationManager(
        double, Conversation(conversation_id=new_conversation_id()), now=clock(SESSION)
    )

    first = instance.send("What does the specification say?")
    second = instance.send("And what about the appendix?")

    assert double.metadata.development_double is True
    assert instance.model_metadata.model_id == DETERMINISTIC_MODEL_ID
    assert len(second.conversation.turns) == 4
    assert second.turn.text
    assert first.conversation.turns[1].text != ""
    assert second.has_unresolved_references is False


def test_the_development_double_sees_the_projected_context() -> None:
    double = DeterministicModel()
    instance = ConversationManager(
        double, Conversation(conversation_id="conv_ctx"), now=clock(SESSION)
    )

    reply = instance.send("0123456789")

    # The double reports the last message's length, which is how a test can confirm
    # the projection reached the model without asserting on a language model.
    assert "10 characters" in reply.text
