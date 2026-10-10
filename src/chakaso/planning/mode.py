"""The retrieval mode a query plan chooses.

ADR-0022 keeps planning a decision layer: a plan names *how* to retrieve without
performing retrieval. These modes are the strategies the retrieval stack can produce.

``NONE`` is not a strategy but the decision not to retrieve — a greeting needs no
evidence. The other three are all backed by a retriever today: ``LEXICAL`` (BM25), ``DENSE``
(an exact similarity index over an `EmbeddingModel` boundary) and ``HYBRID`` (reciprocal-rank
fusion of the two), all in `chakaso.retrieval` (ADR-0023/ADR-0024). The dense retriever's only
embedding is a non-semantic development double, so a dense or hybrid result is similarity and
fusion over a fixture metric, never understanding. Naming a mode makes no claim that meaning was
understood.
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
