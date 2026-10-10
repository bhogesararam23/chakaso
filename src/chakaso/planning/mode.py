"""The retrieval mode a query plan chooses.

ADR-0022 keeps planning a decision layer: a plan names *how* to retrieve without
performing retrieval. These modes are the strategies the retrieval stack can produce.

``NONE`` is not a strategy but the decision not to retrieve — a greeting needs no
evidence. ``LEXICAL`` is the only mode a retriever backs today (BM25,
`chakaso.retrieval`). ``DENSE`` and ``HYBRID`` name the retrieval strategies added in
later phases behind an embedding boundary; they are declared here so the planner's
vocabulary is stable, but a planner must not select them until a retriever exists, and a
fixture dense provider must never be described as semantic. Naming a mode makes no claim
that meaning was understood.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["QueryMode"]


class QueryMode(StrEnum):
    """How a planned turn should retrieve evidence, or not."""

    NONE = "none"
    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"

    @property
    def requires_retrieval(self) -> bool:
        """Whether choosing this mode means evidence must be retrieved."""
        return self is not QueryMode.NONE
