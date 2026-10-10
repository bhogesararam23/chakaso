"""Query planning: decide, for one turn, what retrieval should happen — if any.

This is a deterministic *decision* layer, not a language model and not a discourse
resolver (ADR-0022). Given a user turn and the conversation so far it answers one question
— "what retrieval operation should this turn perform?" — and produces an inspectable
`QueryPlan`: the retrieval mode, whether retrieval is required, the original and normalized
query, which prior sources to constrain retrieval to, the context that informed the
decision, the reasons it fired, and the planner's version. It performs no retrieval, owns
no evidence identifiers, and claims no semantic understanding.

The boundary (`QueryPlanner`, `QueryPlan`, `QueryMode`, `normalize_query`) lives here; the
only implementation, `DeterministicQueryPlanner`, is in `chakaso.planning.deterministic`.
A plan can retrieve lexically, densely, or by hybrid fusion (all backed by retrievers,
ADR-0023/ADR-0024), or not at all. The dense retriever's embedding is a non-semantic
development double, so dense and hybrid plans promise similarity and fusion computation, not
understanding.
"""

from __future__ import annotations

from chakaso.planning.deterministic import PLANNER_VERSION, DeterministicQueryPlanner
from chakaso.planning.errors import InvalidQueryPlanError, PlanningError
from chakaso.planning.mode import QueryMode
from chakaso.planning.normalize import normalize_query
from chakaso.planning.plan import QueryPlan
from chakaso.planning.planner import QueryPlanner

__all__ = [
    "PLANNER_VERSION",
    "DeterministicQueryPlanner",
    "InvalidQueryPlanError",
    "PlanningError",
    "QueryMode",
    "QueryPlan",
    "QueryPlanner",
    "normalize_query",
]
