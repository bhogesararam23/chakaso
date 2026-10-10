"""The planner boundary: a typed decision, not a model.

ADR-0022 makes query planning a layer the rest of the runtime depends on through this
protocol, exactly as ADR-0002 makes language modelling depend on a boundary. The only
implementation today is `chakaso.planning.deterministic.DeterministicQueryPlanner`; a
future intelligent planner replaces it behind the same signature without touching callers.
The planner is given the turn and the conversation state (which it may only read) and
returns a `QueryPlan`. It retrieves nothing itself and understands nothing — it decides.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from chakaso.conversation.state import Conversation
from chakaso.planning.plan import QueryPlan

__all__ = ["QueryPlanner"]


@runtime_checkable
class QueryPlanner(Protocol):
    """Decide what retrieval, if any, a turn warrants."""

    def plan(self, message: str, *, conversation: Conversation | None = None) -> QueryPlan:
        """Return the plan for ``message`` given ``conversation`` context.

        Args:
            message: the user's turn, in its original form.
            conversation: prior conversation state the planner may read to tell a
                follow-up from a new topic, or ``None`` for a standalone turn.

        Returns:
            A `QueryPlan` naming the chosen mode and the real reasons it fired.
        """
        ...
