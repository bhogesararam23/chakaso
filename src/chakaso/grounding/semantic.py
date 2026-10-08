"""The semantic grounding boundary: does evidence actually support a claim?

Structural grounding (``grounding.evaluate``) can only ever say a claim cited supplied evidence;
it cannot know whether that evidence *proves* the claim. That judgement is semantic, needs an
evaluator that does not exist here, and is exactly the line ADR-0015 refuses to blur. This module
does not cross the line — it builds the *boundary* a future semantic judge will implement, so the
answer-evaluation and correction code can be written against it today and validated against a
fixture, without any claim that the capability is real.

Three pieces:

* :class:`SemanticJudge` — the per-claim question, ``judge(claim, evidence) -> SemanticEvaluation``.
  A trained Chakaso model or a careful human process would implement this later.
* :class:`SemanticGroundingEvaluator` — an adapter from a per-claim judge onto the existing
  :class:`~chakaso.grounding.evaluate.GroundingEvaluator` boundary, so ``evaluate_answer`` and
  ``reassess`` take a semantic evaluator *without any change to their own signatures*.
* :class:`FixtureSemanticJudge` — the only judge that exists: it returns relations a caller
  explicitly supplied. It is a fixture, not intelligence, and is named so it can never be mistaken
  for the real thing in a report.

A judgement is still "supported / contradicted / uncertain *by the evidence as judged*", never
"true" (ADR-0014). There is no numeric confidence: no calibrated judge exists, and a number here
would be the fabrication the project forbids. External judge APIs are deliberately not runtime
dependencies (ADR-0001).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from chakaso.claims.model import Claim, ClaimId, ClaimStatus
from chakaso.core.identifiers import ChunkId
from chakaso.evidence import EvidencePack
from chakaso.grounding.model import Contradiction, GroundingResult

__all__ = [
    "FixtureSemanticJudge",
    "SemanticEvaluation",
    "SemanticGroundingEvaluator",
    "SemanticJudge",
    "SemanticJudgement",
]


class SemanticJudgement(StrEnum):
    """A per-claim semantic verdict about how the evidence bears on it.

    There is no ``unsupported`` verdict: semantic grounding either establishes support, opposes
    it, leaves it open, or does not decide. "the evidence simply was not there" is a structural
    fact the structural evaluator already handles, not a semantic judgement.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    NOT_EVALUATED = "not_evaluated"


#: The single source of the judgement -> claim-status translation, defined after the enum.
_JUDGEMENT_TO_STATUS: Mapping[SemanticJudgement, ClaimStatus] = MappingProxyType(
    {
        SemanticJudgement.SUPPORTED: ClaimStatus.SUPPORTED,
        SemanticJudgement.CONTRADICTED: ClaimStatus.CONTRADICTED,
        SemanticJudgement.UNCERTAIN: ClaimStatus.UNCERTAIN,
        SemanticJudgement.NOT_EVALUATED: ClaimStatus.NOT_EVALUATED,
    }
)


@dataclass(frozen=True, slots=True)
class SemanticEvaluation:
    """One claim's semantic verdict, the judge that made it, and any conflicting evidence."""

    claim_id: ClaimId
    judgement: SemanticJudgement
    evaluator_name: str
    rationale: str = ""
    conflicting_chunk_ids: tuple[ChunkId, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "conflicting_chunk_ids", tuple(dict.fromkeys(self.conflicting_chunk_ids))
        )

    @property
    def status(self) -> ClaimStatus:
        """The claim status this judgement maps to."""
        return _JUDGEMENT_TO_STATUS[self.judgement]


@runtime_checkable
class SemanticJudge(Protocol):
    """The per-claim boundary: decide how ``evidence`` bears on a single ``claim``."""

    def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:
        """Return the semantic evaluation for ``claim`` against ``evidence``."""
        ...


class FixtureSemanticJudge:
    """A judge that returns relations a caller supplied — a fixture, not semantic understanding.

    The caller (a test, or a benchmark case whose "expected relation" was authored by a human)
    provides the per-claim judgement; anything not listed is ``not_evaluated``. It is named so it
    reads unmistakably as a fixture in any report, because presenting it as a semantic evaluator
    would be the single most dangerous confusion this boundary invites.
    """

    name: str = "fixture-semantic-0.1"

    def __init__(self, judgements: Mapping[ClaimId, SemanticJudgement]) -> None:
        self._judgements = MappingProxyType(dict(judgements))

    def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:  # noqa: ARG002
        # `evidence` is part of the shared judge signature; a fixture does not consult it, because
        # it is replaying a supplied relation, not reading the text.
        judgement = self._judgements.get(claim.claim_id, SemanticJudgement.NOT_EVALUATED)
        return SemanticEvaluation(
            claim_id=claim.claim_id,
            judgement=judgement,
            evaluator_name=self.name,
            rationale="fixture: caller-supplied relation",
        )


class SemanticGroundingEvaluator:
    """Adapts a per-claim :class:`SemanticJudge` to the answer-level grounding boundary.

    It satisfies the same :class:`~chakaso.grounding.evaluate.GroundingEvaluator` protocol as the
    structural evaluator, so injecting one into ``evaluate_answer`` or ``reassess`` is a
    configuration choice, not a code change (ADR-0015/0019). It ignores the citation-structure
    parameters: a semantic judge decides on evidence meaning, not on whether an identifier was
    present. Where a judge names two or more conflicting chunks the result records the
    disagreement — still without naming a winner, which is correction's job, not the evaluator's.
    """

    def __init__(self, judge: SemanticJudge, *, name: str | None = None) -> None:
        self._judge = judge
        self.name: str = name if name is not None else str(getattr(judge, "name", "semantic-0.1"))

    def evaluate(
        self,
        answer_id: str,
        claims: Iterable[Claim],
        pack: EvidencePack,
        *,
        requires_citation: Collection[ClaimId] | None = None,  # noqa: ARG002 - semantics, not presence
        expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,  # noqa: ARG002
    ) -> GroundingResult:
        statuses: dict[ClaimId, ClaimStatus] = {}
        contradictions: list[Contradiction] = []
        for claim in claims:
            evaluation = self._judge.judge(claim, pack)
            statuses[claim.claim_id] = evaluation.status
            named_conflict = (
                evaluation.judgement is SemanticJudgement.CONTRADICTED
                and len(evaluation.conflicting_chunk_ids) >= 2
            )
            if named_conflict:
                contradictions.append(
                    Contradiction(
                        claim.claim_id,
                        evaluation.conflicting_chunk_ids,
                        detail=evaluation.rationale,
                    )
                )
        return GroundingResult(
            answer_id=answer_id,
            statuses=MappingProxyType(statuses),
            contradictions=tuple(contradictions),
            evaluator_name=self.name,
        )
