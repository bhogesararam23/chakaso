"""Conversation state.

This is the application's conversation type, not the model's input type. A turn
carries provenance a model has no use for — when it happened, which evidence was
retrieved for it and which sources it referenced — and
:meth:`Conversation.to_model_messages` is the single place where the two are
mapped. Keeping them apart means a change to conversation state cannot change what
a model sees by accident.

State is immutable and every operation returns a new conversation. History is
therefore appended to rather than edited, which is the same rule the correction
path depends on: an earlier answer has to remain in the record for a revision to be
explainable.

What is deliberately absent: claims, and an answer record. Both belong to the
reassessment path, which does not exist yet. Adding them here would put the shape of
correction into the shape of conversation before anything has decided what a claim
is.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from chakaso.core.errors import ChakasoError
from chakaso.core.identifiers import SourceId
from chakaso.models import Message, Role

__all__ = ["Conversation", "ConversationError", "Turn"]


class ConversationError(ChakasoError, ValueError):
    """Conversation state would be internally inconsistent."""


def _require_aware(moment: datetime, field_name: str) -> datetime:
    """Return ``moment`` if it carries a timezone, else raise.

    Naive timestamps silently mean "whatever the machine's timezone was", which is
    the ambiguity a conversation record exists to remove.
    """
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        message = (
            f"{field_name} must be timezone-aware, got a naive datetime. "
            "Use datetime.now(datetime.UTC) or attach an explicit offset."
        )
        raise ConversationError(message)
    return moment


@dataclass(frozen=True, slots=True)
class Turn:
    """One user or assistant turn.

    ``evidence_source_ids`` is every source supplied to the model for this turn, and
    ``cited_source_ids`` is the subset the answer actually referenced. They are
    different facts: a follow-up such as "which of those did you use?" is
    unanswerable if only the union is kept.
    """

    role: Role
    text: str
    created_at: datetime
    evidence_source_ids: tuple[SourceId, ...] = ()
    cited_source_ids: tuple[SourceId, ...] = ()

    def __post_init__(self) -> None:
        if not self.text:
            message = "a turn must have text; an empty turn is not a turn"
            raise ConversationError(message)
        _require_aware(self.created_at, "created_at")

        cited = set(self.cited_source_ids)
        evidence = set(self.evidence_source_ids)
        if not cited <= evidence:
            unexpected = sorted(str(source_id) for source_id in cited - evidence)
            message = (
                "a turn cannot cite evidence it was not given: "
                f"{unexpected} not in evidence_source_ids"
            )
            raise ConversationError(message)

    @property
    def is_from_user(self) -> bool:
        """Whether the user produced this turn."""
        return self.role is Role.USER


@dataclass(frozen=True, slots=True)
class Conversation:
    """One conversation: its turns, its topic, and what is still open."""

    conversation_id: str
    turns: tuple[Turn, ...] = ()
    active_topic: str | None = None
    entities: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.conversation_id:
            message = "conversation_id must not be empty"
            raise ConversationError(message)

    # -- reading ------------------------------------------------------------

    @property
    def last_turn(self) -> Turn | None:
        """The most recent turn, or ``None`` in an empty conversation."""
        return self.turns[-1] if self.turns else None

    @property
    def last_user_turn(self) -> Turn | None:
        """The most recent user turn, or ``None`` if the user has not spoken."""
        for turn in reversed(self.turns):
            if turn.is_from_user:
                return turn
        return None

    @property
    def previous_source_ids(self) -> tuple[SourceId, ...]:
        """Every source this conversation has touched, most recent first.

        Newest first because the question this answers is "which sources were we
        just talking about".
        """
        seen: dict[SourceId, None] = {}
        for turn in reversed(self.turns):
            for source_id in (*turn.cited_source_ids, *turn.evidence_source_ids):
                seen.setdefault(source_id, None)
        return tuple(seen)

    def to_model_messages(self, *, max_turns: int | None = None) -> tuple[Message, ...]:
        """Map the conversation onto messages a model can be given.

        Args:
            max_turns: keep only this many of the most recent turns. ``None`` keeps
                all of them. This is the only context control that exists yet;
                budgeting against a model's ``context_window`` is the caller's job,
                because only the caller knows what else is going into the prompt.
        """
        if max_turns is not None and max_turns < 1:
            message = f"max_turns must be at least 1 when given, got {max_turns}"
            raise ConversationError(message)

        selected = self.turns if max_turns is None else self.turns[-max_turns:]
        return tuple(Message(role=turn.role, content=turn.text) for turn in selected)

    # -- appending ----------------------------------------------------------

    def with_user_turn(self, text: str, *, at: datetime) -> Conversation:
        """Return a conversation with a user turn appended."""
        turn = Turn(role=Role.USER, text=text, created_at=at)
        return self._append(turn)

    def with_assistant_turn(
        self,
        text: str,
        *,
        at: datetime,
        evidence_source_ids: tuple[SourceId, ...] = (),
        cited_source_ids: tuple[SourceId, ...] = (),
    ) -> Conversation:
        """Return a conversation with an assistant turn appended.

        Raises:
            ConversationError: the turn cites a source that was not supplied as
                evidence. An answer cannot cite what it was not given, and this is
                the same rule ADR-0003 enforces on the text.
        """
        turn = Turn(
            role=Role.ASSISTANT,
            text=text,
            created_at=at,
            evidence_source_ids=tuple(evidence_source_ids),
            cited_source_ids=tuple(cited_source_ids),
        )
        return self._append(turn)

    def _append(self, turn: Turn) -> Conversation:
        if self.turns:
            previous = self.turns[-1]
            if turn.created_at < previous.created_at:
                message = (
                    f"turn timestamp {turn.created_at.isoformat()} predates the previous "
                    f"turn at {previous.created_at.isoformat()}; history is append-only"
                )
                raise ConversationError(message)
        return replace(self, turns=(*self.turns, turn))

    # -- annotating ---------------------------------------------------------

    def with_active_topic(self, topic: str | None) -> Conversation:
        """Return a conversation whose active topic is ``topic``."""
        if topic is not None and not topic.strip():
            message = "active topic must not be blank; use None to clear it"
            raise ConversationError(message)
        return replace(self, active_topic=topic)

    def with_entities(self, *names: str) -> Conversation:
        """Return a conversation with ``names`` added to its entities.

        Merged rather than replaced, preserving first-seen order and ignoring names
        already present, because an entity that has been mentioned does not stop
        being relevant when a new one appears.
        """
        merged: dict[str, None] = dict.fromkeys(self.entities)
        for name in names:
            stripped = name.strip()
            if not stripped:
                message = "an entity name must not be blank"
                raise ConversationError(message)
            merged.setdefault(stripped, None)
        return replace(self, entities=tuple(merged))

    def with_open_questions(self, *questions: str) -> Conversation:
        """Return a conversation with ``questions`` recorded as open.

        Deduplicated for the same reason as entities: a question asked twice is one
        open question, and counting it twice would misrepresent what is unresolved.
        """
        merged: dict[str, None] = dict.fromkeys(self.open_questions)
        for question in questions:
            stripped = question.strip()
            if not stripped:
                message = "an open question must not be blank"
                raise ConversationError(message)
            merged.setdefault(stripped, None)
        return replace(self, open_questions=tuple(merged))

    def without_open_question(self, question: str) -> Conversation:
        """Return a conversation with ``question`` no longer open.

        Absent questions are ignored rather than raising: resolving a question that
        was never recorded is not an error, it just has nothing to do.
        """
        remaining = tuple(item for item in self.open_questions if item != question.strip())
        return replace(self, open_questions=remaining)
