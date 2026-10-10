"""The retrieval-aware runtime's errors.

A runtime that composes several layers can fail where the layers themselves would not: a
plan asking for a retrieval mode nothing was wired to serve, or a turn that cannot proceed.
These are the runtime's own failures, kept distinct from the planner's and retrieval's so a
caller can tell "this layer could not coordinate" from "a component failed."
"""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["ConversationRuntimeError"]


class ConversationRuntimeError(ChakasoError, ValueError):
    """The runtime cannot conduct a turn as planned — a mode is unwired, or a stage cannot run.

    It is raised before any state changes, so a turn the runtime cannot honour leaves the
    conversation exactly as it was: an unexecutable plan is refused, not half-run.
    """
