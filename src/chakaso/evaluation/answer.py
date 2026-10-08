"""Answer evaluation: combine retrieval, citation and grounding into separate dimensions.

An answer's trustworthiness is not one number. This assembles the pieces — how well the
citations hold (citation metrics), how the supplied evidence bears on the claims (a
grounding result) — into a single view that keeps them *separate*, because a high citation
precision over a few well-cited claims is not the same fact as every claim being supported,
and merging them into one "quality score" would destroy exactly the distinctions the
project exists to make.

There is no merged score here, and `evidence_coverage` is a defined quantity, not vibes:
the fraction of claims the supplied evidence supports — the grounded-claim support rate
`docs/evaluation.md` names. A model is not in the loop; the answer and its claims are
supplied (by a fixture or, later, a real model), and the grounding evaluator may be
structural or a supplied judgement.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

from chakaso.citation import CitationMetrics, citation_metrics, validate_citations
from chakaso.claims.model import Claim, ClaimId
from chakaso.core.identifiers import ChunkId
from chakaso.evidence import EvidencePack
from chakaso.grounding import GroundingEvaluator, GroundingResult, StructuralGroundingEvaluator

__all__ = ["AnswerEvaluation", "evaluate_answer"]


@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    """An answer assessed across separate dimensions, never one merged score."""

    answer_id: str
    claims: int
    citations: CitationMetrics
    grounding: GroundingResult

    @property
    def evidence_coverage(self) -> float:
        """Grounded-claim support rate: claims the supplied evidence supports / all claims.

        ``0.0`` when there are no claims — an answer with nothing to check is not
        reported as fully covered.
        """
        if self.claims == 0:
            return 0.0
        return len(self.grounding.supported) / self.claims

    @property
    def unsupported_claims(self) -> tuple[ClaimId, ...]:
        """Claims that needed evidence and had none that was valid."""
        return self.grounding.unsupported

    @property
    def contradicted_claims(self) -> tuple[ClaimId, ...]:
        """Claims the supplied evidence contradicts, if any evaluator asserted so."""
        return self.grounding.contradicted

    @property
    def is_grounded(self) -> bool:
        """Every claim supported, no contradiction, and no citation to unsupplied evidence."""
        return (
            self.claims > 0
            and self.grounding.is_fully_supported
            and not self.grounding.contradicted
            and self.citations.unknown == 0
        )


def evaluate_answer(
    answer_id: str,
    claims: Iterable[Claim],
    pack: EvidencePack,
    *,
    grounding_evaluator: GroundingEvaluator | None = None,
    requires_citation: Collection[ClaimId] | None = None,
    expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
) -> AnswerEvaluation:
    """Validate citations, ground the claims, and assemble the separate dimensions.

    The grounding evaluator is injectable so a structural pass today can be replaced by a
    semantic one later without changing callers; the default is the honest structural
    evaluator.
    """
    collected = tuple(claims)
    citation_results = validate_citations(
        collected,
        pack,
        expected_evidence=expected_evidence,
        require_citation_for=requires_citation,
    )
    evaluator = grounding_evaluator or StructuralGroundingEvaluator()
    grounding = evaluator.evaluate(
        answer_id,
        collected,
        pack,
        requires_citation=requires_citation,
        expected_evidence=expected_evidence,
    )
    return AnswerEvaluation(
        answer_id=answer_id,
        claims=len(collected),
        citations=citation_metrics(citation_results),
        grounding=grounding,
    )
