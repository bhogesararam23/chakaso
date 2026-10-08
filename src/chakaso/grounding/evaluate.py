"""Grounding evaluators behind one replaceable boundary.

`StructuralGroundingEvaluator` decides only what a string comparison can honestly decide:
a claim with a valid citation is `supported`, a claim that needed evidence and had none is
`unsupported`, and everything else is left `not_evaluated`. It never emits
`contradicted` or `uncertain`, because noticing that two sources disagree requires
understanding them, which this does not do.

`ManualGroundingEvaluator` is the place those stronger statuses come from: a judgement —
a human label on a small set, or a future model — supplies per-claim statuses and any
contradictions. Both satisfy the same `GroundingEvaluator` protocol, so the answer
evaluator and correction logic are written against the boundary and can move from
structural to semantic evaluation without a rewrite. A semantic evaluator is explicitly a
future extension point, not a present capability.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from chakaso.citation import validate_citations
from chakaso.claims.model import Claim, ClaimId, ClaimStatus
from chakaso.core.identifiers import ChunkId
from chakaso.evidence import EvidencePack
from chakaso.grounding.model import Contradiction, GroundingResult

__all__ = [
    "GroundingEvaluator",
    "ManualGroundingEvaluator",
    "StructuralGroundingEvaluator",
]


@runtime_checkable
class GroundingEvaluator(Protocol):
    """The boundary: decide, per claim, how the supplied evidence bears on it."""

    def evaluate(
        self,
        answer_id: str,
        claims: Iterable[Claim],
        pack: EvidencePack,
        *,
        requires_citation: Collection[ClaimId] | None = None,
        expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
    ) -> GroundingResult:
        """Return a grounding result for ``claims`` against ``pack``."""
        ...


class StructuralGroundingEvaluator:
    """Support/unsupported by citation presence only; never contradiction or uncertainty."""

    name: str = "structural-0.1"

    def evaluate(
        self,
        answer_id: str,
        claims: Iterable[Claim],
        pack: EvidencePack,
        *,
        requires_citation: Collection[ClaimId] | None = None,
        expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
    ) -> GroundingResult:
        statuses: dict[ClaimId, ClaimStatus] = {}
        for result in validate_citations(
            claims,
            pack,
            expected_evidence=expected_evidence,
            require_citation_for=requires_citation,
        ):
            if result.has_valid_citation:
                statuses[result.claim_id] = ClaimStatus.SUPPORTED
            elif result.requires_citation:
                statuses[result.claim_id] = ClaimStatus.UNSUPPORTED
            else:
                statuses[result.claim_id] = ClaimStatus.NOT_EVALUATED
        return GroundingResult(
            answer_id=answer_id,
            statuses=MappingProxyType(statuses),
            evaluator_name=self.name,
        )


class ManualGroundingEvaluator:
    """Applies statuses and contradictions a caller (human label or future model) decided."""

    name: str = "manual-0.1"

    def __init__(
        self,
        statuses: Mapping[ClaimId, ClaimStatus],
        contradictions: Iterable[Contradiction] = (),
    ) -> None:
        self._statuses = MappingProxyType(dict(statuses))
        self._contradictions = tuple(contradictions)

    def evaluate(
        self,
        answer_id: str,
        claims: Iterable[Claim],  # noqa: ARG002 - a judgement is supplied, not derived
        pack: EvidencePack,  # noqa: ARG002
        *,
        requires_citation: Collection[ClaimId] | None = None,  # noqa: ARG002
        expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,  # noqa: ARG002
    ) -> GroundingResult:
        return GroundingResult(
            answer_id=answer_id,
            statuses=self._statuses,
            contradictions=self._contradictions,
            evaluator_name=self.name,
        )
