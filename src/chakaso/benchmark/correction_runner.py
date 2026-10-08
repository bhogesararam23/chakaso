"""Runs the correction benchmark: cases → reassessment → outcome → metrics.

The runner is honest about what a "correction benchmark" can be with no model. Each case names a
prior answer's claims, the evidence to reassess against, and the decision the case *expects*. The
runner builds a real ``EvidencePack`` from the shared fixture corpus (so citations resolve to real
chunk identifiers), constructs the claims, and calls the same ``reassess`` the application uses —
choosing the structural evaluator, or a fixture semantic judge, purely from what the case declares
(a case that needs a contradiction or an uncertain verdict supplies the relation; the runner never
infers one). A case passes when the decision rule reaches the expected outcome.

So what is measured is the *decision rule*, over synthetic cases, not real-world correction skill.
The metrics keep both failure directions visible (an answer kept when it should change, and one
changed when it should not) and abstention/review appropriateness separate. A rate with no matching
cases is ``None``, never a fabricated ``0.0``/``1.0``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from chakaso.benchmark.correction import CorrectionCase, CorrectionDataset
from chakaso.benchmark.fixtures import build_corpus, resolved_chunk_ids
from chakaso.benchmark.identity import CaseId
from chakaso.claims.model import Claim, ClaimId, ClaimStatus
from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.correction.decision import CorrectionDecision
from chakaso.correction.reassess import Reassessment, reassess
from chakaso.evidence import EvidenceChunk, EvidencePack, SourceRecord
from chakaso.grounding import (
    FixtureSemanticJudge,
    GroundingEvaluator,
    SemanticGroundingEvaluator,
    StructuralGroundingEvaluator,
)
from chakaso.retrieval import Corpus

__all__ = [
    "CorrectionBenchmarkMetrics",
    "CorrectionBenchmarkRun",
    "CorrectionBenchmarkRunner",
    "CorrectionCaseOutcome",
]


@dataclass(frozen=True, slots=True)
class CorrectionCaseOutcome:
    """One case's expected decision, the decision reached, and how to inspect the difference."""

    case_id: CaseId
    case_version: str
    description: str
    expected_decision: CorrectionDecision
    actual_decision: CorrectionDecision
    evaluator_name: str
    reassessment_labels: tuple[str, ...]
    affected_claim_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether the decision rule reached the case's expected decision."""
        return self.actual_decision is self.expected_decision


@dataclass(frozen=True, slots=True)
class CorrectionBenchmarkMetrics:
    """Aggregate accuracy and both correction failure directions over a run, with counts."""

    total_cases: int
    passed_cases: int
    expected_by_decision: Mapping[CorrectionDecision, int]
    correct_by_decision: Mapping[CorrectionDecision, int]
    unjustified_persistence_cases: int
    unnecessary_revision_cases: int
    required_change_cases: int
    expected_retain_cases: int

    @property
    def decision_accuracy(self) -> float | None:
        """Cases whose reached decision matched the expected one; ``None`` if there were none."""
        if not self.total_cases:
            return None
        return self.passed_cases / self.total_cases

    def rate_for(self, decision: CorrectionDecision) -> float | None:
        """Accuracy over cases whose *expected* decision is ``decision``; ``None`` if none."""
        expected = self.expected_by_decision.get(decision, 0)
        if not expected:
            return None
        return self.correct_by_decision.get(decision, 0) / expected

    @property
    def appropriate_abstention_rate(self) -> float | None:
        """Of the cases that should abstain, how many did."""
        return self.rate_for(CorrectionDecision.ABSTAIN)

    @property
    def appropriate_review_rate(self) -> float | None:
        """Of the cases that should go to review, how many did."""
        return self.rate_for(CorrectionDecision.NEEDS_REVIEW)

    @property
    def unjustified_persistence_rate(self) -> float | None:
        """Of the cases that required a change, how many the system left unchanged."""
        if not self.required_change_cases:
            return None
        return self.unjustified_persistence_cases / self.required_change_cases

    @property
    def unnecessary_revision_rate(self) -> float | None:
        """Of the cases that should have stayed, how many the system changed anyway."""
        if not self.expected_retain_cases:
            return None
        return self.unnecessary_revision_cases / self.expected_retain_cases


@dataclass(frozen=True, slots=True)
class CorrectionBenchmarkRun:
    """A completed correction benchmark run and its provenance."""

    dataset_id: str
    dataset_version: str
    dataset_content_fingerprint: str
    outcomes: tuple[CorrectionCaseOutcome, ...]
    metrics: CorrectionBenchmarkMetrics
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def passed(self) -> bool:
        """Whether every case reached its expected decision."""
        return all(outcome.passed for outcome in self.outcomes)


class CorrectionBenchmarkRunner:
    """Runs a correction dataset through ``reassess`` and scores the decisions."""

    def __init__(self, corpus: Corpus | None = None) -> None:
        self._corpus = corpus if corpus is not None else build_corpus()
        self._resolved = resolved_chunk_ids(self._corpus)

    def run(self, dataset: CorrectionDataset) -> CorrectionBenchmarkRun:
        outcomes = tuple(self._run_case(case) for case in dataset.cases)
        return CorrectionBenchmarkRun(
            dataset_id=str(dataset.dataset_id),
            dataset_version=dataset.version,
            dataset_content_fingerprint=dataset.content_fingerprint,
            outcomes=outcomes,
            metrics=_aggregate(outcomes),
        )

    def _run_case(self, case: CorrectionCase) -> CorrectionCaseOutcome:
        claims = self._build_claims(case)
        pack = self._build_pack(case.reassessment_labels)
        required = self._required_claims(case, claims)
        evaluator = self._evaluator(case, claims)
        result = reassess(
            str(case.case_id),
            claims,
            pack,
            grounding_evaluator=evaluator,
            requires_citation=required,
        )
        return CorrectionCaseOutcome(
            case_id=case.case_id,
            case_version=case.version,
            description=case.description,
            expected_decision=case.expected_decision,
            actual_decision=result.decision,
            evaluator_name=result.evaluator_name,
            reassessment_labels=case.reassessment_labels,
            affected_claim_ids=tuple(str(cid) for cid in _changed_claims(result)),
        )

    def _build_claims(self, case: CorrectionCase) -> tuple[Claim, ...]:
        answer_id = str(case.case_id)
        claims: list[Claim] = []
        for position, claim in enumerate(case.claims):
            cited: tuple[ChunkId, ...] = ()
            if claim.cite_label is not None:
                _source_id, chunk_ids = self._resolve(claim.cite_label)
                cited = chunk_ids
            claims.append(
                Claim(
                    answer_id=answer_id,
                    position=position,
                    text=claim.text,
                    cited_evidence_ids=cited,
                    status=claim.prior_status,
                )
            )
        return tuple(claims)

    def _build_pack(self, labels: Sequence[str]) -> EvidencePack:
        chunks: list[EvidenceChunk] = []
        sources: list[SourceRecord] = []
        for label in labels:
            source_id, chunk_ids = self._resolve(label)
            source = self._corpus.source_for(source_id)
            if source is None:  # pragma: no cover - resolved ids come from this corpus
                message = f"corpus has no source for fixture label {label!r}"
                raise LookupError(message)
            sources.append(source)
            for chunk_id in chunk_ids:
                chunk = self._corpus.chunk_for(chunk_id)
                if chunk is not None:
                    chunks.append(chunk)
        return EvidencePack.of(chunks, sources)

    def _resolve(self, label: str) -> tuple[SourceId, tuple[ChunkId, ...]]:
        if label not in self._resolved:
            message = (
                f"correction benchmark references unknown fixture label {label!r}; "
                "cases may name only labels in the shared corpus"
            )
            raise LookupError(message)
        return self._resolved[label]

    @staticmethod
    def _required_claims(case: CorrectionCase, claims: Sequence[Claim]) -> frozenset[ClaimId]:
        if case.required_positions is not None:
            return frozenset(claims[position].claim_id for position in case.required_positions)
        return frozenset(
            claim.claim_id for claim in claims if claim.status is ClaimStatus.SUPPORTED
        )

    @staticmethod
    def _evaluator(case: CorrectionCase, claims: Sequence[Claim]) -> GroundingEvaluator:
        relations = {
            claim.claim_id: definition.semantic_relation
            for claim, definition in zip(claims, case.claims, strict=True)
            if definition.semantic_relation is not None
        }
        if relations:
            # A fixture judge asserts the caller-supplied relation; the runner never infers a
            # contradiction or an uncertainty (ADR-0015/0019).
            return SemanticGroundingEvaluator(FixtureSemanticJudge(relations))
        return StructuralGroundingEvaluator()


def _changed_claims(result: Reassessment) -> tuple[ClaimId, ...]:
    return tuple(sorted((r.claim_id for r in result.claims if r.changed), key=str))


def _aggregate(outcomes: Sequence[CorrectionCaseOutcome]) -> CorrectionBenchmarkMetrics:
    expected_by: dict[CorrectionDecision, int] = {}
    correct_by: dict[CorrectionDecision, int] = {}
    for outcome in outcomes:
        expected_by[outcome.expected_decision] = expected_by.get(outcome.expected_decision, 0) + 1
        if outcome.passed:
            correct_by[outcome.expected_decision] = correct_by.get(outcome.expected_decision, 0) + 1

    required_change = sum(
        1 for o in outcomes if o.expected_decision is not CorrectionDecision.RETAIN
    )
    unjustified_persistence = sum(
        1
        for o in outcomes
        if o.expected_decision is not CorrectionDecision.RETAIN
        and o.actual_decision is CorrectionDecision.RETAIN
    )
    expected_retain = sum(1 for o in outcomes if o.expected_decision is CorrectionDecision.RETAIN)
    unnecessary_revision = sum(
        1
        for o in outcomes
        if o.expected_decision is CorrectionDecision.RETAIN
        and o.actual_decision is not CorrectionDecision.RETAIN
    )

    return CorrectionBenchmarkMetrics(
        total_cases=len(outcomes),
        passed_cases=sum(1 for o in outcomes if o.passed),
        expected_by_decision=MappingProxyType(expected_by),
        correct_by_decision=MappingProxyType(correct_by),
        unjustified_persistence_cases=unjustified_persistence,
        unnecessary_revision_cases=unnecessary_revision,
        required_change_cases=required_change,
        expected_retain_cases=expected_retain,
    )
