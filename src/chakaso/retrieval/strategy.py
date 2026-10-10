"""Which retrieval strategy produced a result.

Evaluation (ADR-0013's lesson) has to be able to tell a lexical hit from a dense one from
a fused one, so every ``RetrievalResult`` carries the strategy that computed it rather than
leaving a caller to infer it from which retriever object ran. The three values correspond
to the modes a query plan can name; ``HYBRID`` marks a result produced by fusing the other
two, and it does not exist until the hybrid retriever is built.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["RetrievalStrategy"]


class RetrievalStrategy(StrEnum):
    """The mechanism that produced a retrieval result."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"
