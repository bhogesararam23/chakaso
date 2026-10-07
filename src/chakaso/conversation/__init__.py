"""Conversation: immutable state, and the manager that drives it.

`state` holds the records — turns with their provenance, the active topic, entities
and open questions. `manager` conducts a turn against the language-model boundary
and advances the state.

The response engine available today is the deterministic development double. There
is no language model in this repository, no retrieval, and no correction.
"""

from __future__ import annotations

from chakaso.conversation.errors import (
    ConversationError,
    ConversationManagerError,
    InvalidGenerationError,
    InvalidUserInputError,
)
from chakaso.conversation.manager import ConversationManager, Reply
from chakaso.conversation.state import (
    CONVERSATION_ID_PREFIX,
    Conversation,
    Turn,
    new_conversation_id,
)

__all__ = [
    "CONVERSATION_ID_PREFIX",
    "Conversation",
    "ConversationError",
    "ConversationManager",
    "ConversationManagerError",
    "InvalidGenerationError",
    "InvalidUserInputError",
    "Reply",
    "Turn",
    "new_conversation_id",
]
