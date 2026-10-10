"""The query plan: an inspectable record of what the planner decided.

A ``QueryPlan`` is the planner's whole output and nothing more. It names the retrieval mode
and whether retrieval is required at all, keeps both the user's original turn and its
normalized form, records which conversation context informed the decision and — as plain
strings — the actual reasons the rules fired. It carries a planner version so a plan is a
reproducible artifact and a later planner can be compared against an earlier one (ADR-0022).

The shape holds references and decisions, not data: ``source_constraints`` are evidence
identifiers owned by retrieval (the planner never mints one), and no retrieved content
lives here — a plan says what to fetch, it does not fetch it. The invariants on
construction make the honesty rule structural rather than a convention: a mode and its
``retrieval_required`` flag cannot disagree, an evidence-reuse claim must name the sources
it reuses, a plan that retrieves must have something to retrieve for, and every plan must
explain itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chakaso.core.identifiers import SourceId
from chakaso.planning.errors import InvalidQueryPlanError
from chakaso.planning.mode import QueryMode

__all__ = ["QueryPlan"]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """What the planner decided to do for one turn, and why."""

    mode: QueryMode
    original_query: str
    normalized_query: str
    retrieval_required: bool
    planner_version: str
    source_constraints: tuple[SourceId, ...] = ()
    top_k: int | None = None
    reused_evidence: bool = False
    context_used: tuple[str, ...] = ()
    explanation: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.planner_version.strip():
            message = "a query plan must carry the version of the planner that produced it"
            raise InvalidQueryPlanError(message)
        if not self.explanation:
            message = (
                f"a query plan must explain its decision; mode {self.mode.value!r} with no "
                "reasons is uninspectable and is refused (ADR-0022)"
            )
            raise InvalidQueryPlanError(message)

        if self.retrieval_required is not self.mode.requires_retrieval:
            message = (
                f"mode {self.mode.value!r} and retrieval_required={self.retrieval_required} "
                "disagree: NONE never retrieves and every other mode always does"
            )
            raise InvalidQueryPlanError(message)

        if self.top_k is not None and self.top_k < 1:
            message = f"top_k must be at least 1 when given, got {self.top_k}"
            raise InvalidQueryPlanError(message)

        if self.retrieval_required and not self.normalized_query.strip():
            message = "a plan that retrieves must have a non-empty query to retrieve for"
            raise InvalidQueryPlanError(message)

        if self.reused_evidence and not self.source_constraints:
            message = (
                "a plan cannot claim to reuse evidence while naming no source to reuse; "
                "reuse means retrieval constrained to prior conversation sources"
            )
            raise InvalidQueryPlanError(message)

        for item in self.context_used:
            if not item.strip():
                message = "context_used entries must be non-blank labels"
                raise InvalidQueryPlanError(message)

        object.__setattr__(self, "source_constraints", _dedup(self.source_constraints))

    @property
    def is_follow_up(self) -> bool:
        """Whether this plan leans on evidence already in the conversation."""
        return self.reused_evidence


def _dedup(identifiers: Iterable[SourceId]) -> tuple[SourceId, ...]:
    """Preserve order while removing duplicate source constraints."""
    return tuple(dict.fromkeys(identifiers))
