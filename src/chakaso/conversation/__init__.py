"""Conversation state.

Immutable, append-only conversation state: turns with their provenance, the active
topic, entities, open questions and the sources the conversation has touched. The
conversation manager, which will use this, does not exist yet.
"""

from __future__ import annotations

from chakaso.conversation.state import Conversation, ConversationError, Turn

__all__ = ["Conversation", "ConversationError", "Turn"]
