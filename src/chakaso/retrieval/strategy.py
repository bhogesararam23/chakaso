"""How a retrieval result was produced, and — for a fused result — what fed it.

Two things evaluation needs live here. A ``RetrievalStrategy`` names the single mechanism that
made a result (lexical match, dense similarity, or a hybrid fusion). ``HybridProvenance`` is
what a *fused* result must additionally carry so it stays auditable: where each component
ranked the chunk and what score it gave, alongside the fused score that decided the final
order. Without it, a hybrid result would be an unattributable number — the exact thing the
fusion is supposed to make inspectable rather than hide (ADR-0024).

Both are dependency-free vocabulary: the retrieval modules import them, never the reverse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from chakaso.retrieval.errors import RetrievalError

__all__ = ["HybridProvenance", "RetrievalStrategy"]


class RetrievalStrategy(StrEnum):
    """The mechanism that produced a retrieval result."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class HybridProvenance:
    """The component contributions behind one fused hybrid result.

    Attributes:
        lexical_rank: The chunk's 1-based rank in the lexical result, or ``None`` if lexical
            retrieval did not return it.
        lexical_score: The lexical score for the chunk, when it was returned lexically.
        dense_rank: The chunk's 1-based rank in the dense result, or ``None`` if dense
            retrieval did not return it.
        dense_score: The dense similarity for the chunk, when it was returned densely.
        fused_score: The value the fusion computed that placed this result in the final order.

    A fused result must trace to at least one component; one with neither rank recorded would
    be a result nothing produced, and construction refuses it.
    """

    lexical_rank: int | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    fused_score: float = 0.0

    def __post_init__(self) -> None:
        if self.lexical_rank is None and self.dense_rank is None:
            message = "hybrid provenance must record at least one component that returned the chunk"
            raise RetrievalError(message)
        if not math.isfinite(self.fused_score):
            message = f"hybrid fused_score must be finite, got {self.fused_score!r}"
            raise RetrievalError(message)
