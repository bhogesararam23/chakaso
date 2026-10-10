"""Errors raised by the query-planning layer.

Planning is a decision layer, so its failures are about decisions that cannot be
made honestly: a plan that says it needs no retrieval while naming a retrieval mode,
an answer-reuse claim with nothing to reuse, or a plan with no explanation of itself.
These are different from a runtime error — they mean a plan was constructed that would
misrepresent what the planner actually decided, and ADR-0022's whole point is that a
plan reports only real decisions. Both derive from ``ValueError`` so a caller validating
an untrusted plan can catch the built-in it already expects.
"""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = ["InvalidQueryPlanError", "PlanningError"]


class PlanningError(ChakasoError, ValueError):
    """Base class for every failure the planning layer raises deliberately."""


class InvalidQueryPlanError(PlanningError):
    """A ``QueryPlan`` cannot be constructed — it would misreport a decision.

    A plan whose mode and ``retrieval_required`` disagree, one that claims to reuse
    evidence while naming no source to reuse, or one that carries no explanation is
    refused at construction: an uninspectable plan is exactly the "fabricated reasoning"
    the planner boundary exists to prevent (ADR-0022).
    """
