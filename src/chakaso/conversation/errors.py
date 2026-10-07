"""Conversation-layer errors.

Each failure the conversation layer can produce has its own type, because they
require different responses. "The user sent nothing" is a caller mistake, "the
model returned nothing" is an implementation problem, and "the model could not be
called at all" is an operational one. Collapsing them into one exception makes an
operational failure look like a caller mistake, and a caller mistake look like
something worth retrying.

Failures originating in the model boundary propagate unchanged. The manager does
not translate them, because `UnsupportedCapabilityError` and `ModelError` already
say precisely what went wrong, and re-wrapping them would lose the ability to tell
a model that cannot do something from a model that failed while doing it.
"""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = [
    "ConversationError",
    "ConversationManagerError",
    "InvalidGenerationError",
    "InvalidUserInputError",
]


class ConversationError(ChakasoError, ValueError):
    """Conversation state would be internally inconsistent.

    Deriving from :class:`ValueError` as well means untrusted input — a turn, an
    entity name, an open question — can be validated with the built-in type.
    """


class ConversationManagerError(ConversationError):
    """Base class for failures raised while conducting a turn.

    A subclass of :class:`ConversationError` so that a caller which only wants to
    know "the conversation layer rejected this" can catch one type, while a caller
    that needs to distinguish the causes can catch the subclasses.
    """


class InvalidUserInputError(ConversationManagerError):
    """A user turn was empty or otherwise unusable.

    The caller's problem, and not worth retrying with the same input.
    """


class InvalidGenerationError(ConversationManagerError):
    """The model returned something that cannot be recorded as an answer.

    An empty response, or a response attributed to a different model than the one
    the manager called. The second case matters because generation provenance that
    cannot be trusted makes an answer unusable for evaluation, which is the point
    of recording it at all.
    """
