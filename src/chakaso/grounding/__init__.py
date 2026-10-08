"""Grounding: deciding, per claim, how the supplied evidence bears on it.

The automatic evaluator is structural and honest about the limit of "structural": it can
tell supported from unsupported by citation presence, not whether a source proves a claim.
Contradiction and uncertainty enter only through a judgement a caller supplies, and the
source-authority hooks let a case state a precedence without the engine imposing one.
"""

from __future__ import annotations

from chakaso.grounding.evaluate import (
    GroundingEvaluator,
    ManualGroundingEvaluator,
    StructuralGroundingEvaluator,
)
from chakaso.grounding.model import (
    Contradiction,
    GroundingResult,
    SourceAuthority,
    most_recent_wins,
)
from chakaso.grounding.semantic import (
    FixtureSemanticJudge,
    SemanticEvaluation,
    SemanticGroundingEvaluator,
    SemanticJudge,
    SemanticJudgement,
)

__all__ = [
    "Contradiction",
    "FixtureSemanticJudge",
    "GroundingEvaluator",
    "GroundingResult",
    "ManualGroundingEvaluator",
    "SemanticEvaluation",
    "SemanticGroundingEvaluator",
    "SemanticJudge",
    "SemanticJudgement",
    "SourceAuthority",
    "StructuralGroundingEvaluator",
    "most_recent_wins",
]
