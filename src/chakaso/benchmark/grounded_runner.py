"""The grounded-answer evaluation runner: a workload and a pipeline in, a measured run out.

This composes boundaries that already exist and are tested on their own. It never reimplements a
retriever, a citation validator or a grounding evaluator: it retrieves through a `RetrievalService`,
takes an answer from an injected provider, decomposes it through `chakaso.claims`, and scores it with
`evaluate_answer` over the `chakaso.citation` / `chakaso.grounding` primitives. Retrieval metrics and
answer metrics are kept separate; abstention, required-fact coverage, prohibited-claim occurrence and
citation integrity are each their own dimension; and nothing here merges into one quality number.

The runner distinguishes the three ways a case can fail so an aggregate never hides one of them:

* a **retrieval** failure — the search raised, so nothing downstream was scored;
* a **generation** failure — the provider raised or produced nothing parseable;
* an **evaluation** failure — scoring the answer against the evidence raised.

A failed case is recorded as failed and counted in the failure totals, never silently scored as a
success or as a zero that drags another metric down. Timeouts are not handled here: the providers are
in-process and no wall-clock guard is pretended; that gap is documented rather than papered over.

The answer providers define what the run measures. `ReferenceAnswerProvider` is an **oracle, not a
model**: it replays each case's own required facts and cites only gold chunks the retrieval actually
surfaced, so it exists to validate that the harness scores a correct, well-cited answer as correct —
a fact about the harness, never a capability claim, recorded as ``is_oracle=True``.
`NonAnsweringProvider` abstains everywhere, the honest shape of today's runtime: there is no model
that answers, so a run with no oracle is expected to abstain on every case. A future model provider
would see only the question and the pack and ignore the gold evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.grounded_cases import CaseSplit, GroundedCase
from chakaso.benchmark.grounded_dataset import GroundedDataset, resolved_chunk_ids
from chakaso.benchmark.grounded_metrics import (
    citation_validity_rate,
    mean,
    prohibited_claim_rate,
    rate,
    required_fact_coverage,
    unsupported_claim_rate,
)
from chakaso.benchmark.grounded_workload import case_evidence
from chakaso.claims.extract import StructuredClaimExtractor
from chakaso.core.identifiers import ChunkId
from chakaso.evaluation import (
    AnswerEvaluation,
    evaluate_answer,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from chakaso.evidence import EvidencePack
from chakaso.retrieval import DEFAULT_TOP_K, Corpus, RetrievalService

__all__ = [
    "FailureCounts",
    "GeneratedAnswer",
    "GroundedAnswerMetrics",
    "GroundedCaseOutcome",
    "GroundedEvaluationRunner",
    "GroundedRetrievalMetrics",
    "GroundedRun",
    "NonAnsweringProvider",
    "ReferenceAnswerProvider",
    "StageFailure",
]


class StageFailure(StrEnum):
    """Which stage of a case failed, or ``NONE`` when it was scored end to end."""

    NONE = "none"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """What an answer provider produced for one case.

    ``claims`` pairs each claim's text with the chunk identifiers it cites; ``abstained`` says the
    provider declined to answer, which is itself a scored disposition, not a missing answer.
    """

    text: str
    claims: tuple[tuple[str, tuple[ChunkId, ...]], ...]
    abstained: bool
    label: str


class AnswerProvider(Protocol):
    """The seam a run answers through. Only the reference oracle uses ``gold`` and ``forbidden``."""

    label: str
    is_oracle: bool

    def answer(
        self,
        case: GroundedCase,
        pack: EvidencePack,
        *,
        gold: tuple[ChunkId, ...],
        forbidden: tuple[ChunkId, ...],
    ) -> GeneratedAnswer: ...


class ReferenceAnswerProvider:
    """An oracle that replays each case's own required facts, citing surfaced gold evidence.

    This is not a model and must never be reported as one. It exists to prove the harness scores a
    correct, correctly-cited answer as correct and an abstention as an abstention. It cites a gold
    chunk only when retrieval surfaced it (a perfect answerer still cannot cite evidence that was
    never retrieved), so a retrieval miss legitimately cascades into lower grounding.
    """

    label = "reference-oracle (not a model)"
    is_oracle = True

    def answer(
        self,
        case: GroundedCase,
        pack: EvidencePack,
        *,
        gold: tuple[ChunkId, ...],
        forbidden: tuple[ChunkId, ...],
    ) -> GeneratedAnswer:
        del forbidden
        if not case.answerable:
            return GeneratedAnswer(
                text="The supplied evidence does not support an answer.",
                claims=(),
                abstained=True,
                label=self.label,
            )
        surfaced = tuple(chunk for chunk in gold if pack.chunk_for(chunk) is not None)
        claims = tuple((fact, surfaced) for fact in case.required_facts)
        return GeneratedAnswer(
            text=" ".join(case.required_facts),
            claims=claims,
            abstained=False,
            label=self.label,
        )


class NonAnsweringProvider:
    """Abstains on every case — the honest shape of a runtime with no answering model."""

    label = "non-answering (no model present)"
    is_oracle = False

    def answer(
        self,
        case: GroundedCase,
        pack: EvidencePack,
        *,
        gold: tuple[ChunkId, ...],
        forbidden: tuple[ChunkId, ...],
    ) -> GeneratedAnswer:
        del case, pack, gold, forbidden
        return GeneratedAnswer(text="", claims=(), abstained=True, label=self.label)


@dataclass(frozen=True, slots=True)
class GroundedCaseOutcome:
    """Everything one case measured: what came back, what the answer said, how it scored."""

    case_id: str
    category: BenchmarkCategory
    split: CaseSplit
    case_version: str
    failure: StageFailure
    retrieved_ids: tuple[str, ...] = ()
    gold_ids: tuple[str, ...] = ()
    forbidden_ids: tuple[str, ...] = ()
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    hit: bool = False
    forbidden_hit: bool = False
    answer_text: str | None = None
    answer_label: str | None = None
    claims: int = 0
    citation_valid: int = 0
    citation_unknown: int = 0
    citation_irrelevant: int = 0
    citation_uncited: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0
    evidence_coverage: float | None = None
    is_grounded: bool | None = None
    required_facts_present: int = 0
    required_facts_total: int = 0
    prohibited_present: int = 0
    prohibited_total: int = 0
    abstention_expected: bool = False
    abstention_observed: bool | None = None
    abstention_correct: bool | None = None

    @property
    def has_gold(self) -> bool:
        """Whether this case names gold evidence (so retrieval has a target)."""
        return bool(self.gold_ids)

    @property
    def retrieval_abstained(self) -> bool:
        """A no-gold case that correctly retrieved nothing."""
        return not self.has_gold and not self.retrieved_ids


@dataclass(frozen=True, slots=True)
class GroundedRetrievalMetrics:
    """Retrieval aggregates, scored and abstention cases kept separate."""

    scored_cases: int
    abstention_cases: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mrr: float
    hit_rate: float
    forbidden_hits: int
    false_retrievals: int


@dataclass(frozen=True, slots=True)
class GroundedAnswerMetrics:
    """Answer aggregates across separate dimensions, never a merged score."""

    answered_cases: int
    mean_evidence_coverage: float
    citation_validity: float
    unsupported_claim_rate: float
    required_fact_coverage: float
    prohibited_claim_rate: float
    appropriate_abstention_rate: float
    grounded_answer_rate: float


@dataclass(frozen=True, slots=True)
class FailureCounts:
    """How many cases failed at each stage."""

    retrieval: int
    generation: int
    evaluation: int

    @property
    def total(self) -> int:
        """Cases that failed anywhere."""
        return self.retrieval + self.generation + self.evaluation


@dataclass(frozen=True, slots=True)
class GroundedRun:
    """A completed grounded-answer evaluation: provenance, per-case outcomes, aggregates."""

    dataset_id: str
    dataset_version: str
    dataset_content_fingerprint: str
    provider_label: str
    provider_is_oracle: bool
    top_k: int
    retriever_name: str
    split_filter: str
    outcomes: tuple[GroundedCaseOutcome, ...]
    retrieval_metrics: GroundedRetrievalMetrics
    answer_metrics: GroundedAnswerMetrics
    failures: FailureCounts
    per_category: Mapping[str, Mapping[str, float]]
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


class GroundedEvaluationRunner:
    """Runs a grounded workload through retrieval and an answer provider, scoring each dimension."""

    def __init__(
        self,
        corpus: Corpus,
        *,
        service: RetrievalService | None = None,
        provider: AnswerProvider | None = None,
        top_k: int = DEFAULT_TOP_K,
        retriever_name: str = "lexical",
        split_filter: CaseSplit | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._service = service if service is not None else RetrievalService(corpus)
        self._provider = provider
        self._top_k = top_k
        self._retriever_name = retriever_name
        self._split_filter = split_filter
        self._metadata = MappingProxyType(dict(metadata or {}))

    def run(self, dataset: GroundedDataset) -> GroundedRun:
        """Retrieve and answer every (selected) case and compute the run's metrics."""
        resolved = resolved_chunk_ids(dataset, self._service.corpus)
        evidence = case_evidence(dataset, resolved)
        outcomes = tuple(
            self._run_case(case, evidence[str(case.case_id)])
            for case in dataset.cases
            if self._split_filter is None or case.split is self._split_filter
        )
        return GroundedRun(
            dataset_id=str(dataset.dataset_id),
            dataset_version=dataset.version,
            dataset_content_fingerprint=dataset.content_fingerprint,
            provider_label=self._provider.label if self._provider is not None else "retrieval-only",
            provider_is_oracle=self._provider.is_oracle if self._provider is not None else False,
            top_k=self._top_k,
            retriever_name=self._retriever_name,
            split_filter=self._split_filter.value if self._split_filter is not None else "all",
            outcomes=outcomes,
            retrieval_metrics=_aggregate_retrieval(outcomes),
            answer_metrics=_aggregate_answer(outcomes),
            failures=_aggregate_failures(outcomes),
            per_category=_aggregate_by_category(outcomes),
            metadata=self._metadata,
        )

    def _run_case(
        self,
        case: GroundedCase,
        evidence: tuple[tuple[ChunkId, ...], tuple[ChunkId, ...], tuple[ChunkId, ...]],
    ) -> GroundedCaseOutcome:
        gold, forbidden, _required = evidence
        gold_ids = tuple(str(chunk) for chunk in gold)
        forbidden_ids = tuple(str(chunk) for chunk in forbidden)
        ids_kwargs = {"gold_ids": gold_ids, "forbidden_ids": forbidden_ids}

        try:
            outcome = self._service.search(case.question, top_k=self._top_k)
        except Exception as exc:
            del exc
            return self._outcome(case, failure=StageFailure.RETRIEVAL, **ids_kwargs)

        retrieved = tuple(str(result.chunk.chunk_id) for result in outcome.results)
        relevant = frozenset(gold_ids)
        forbidden_set = frozenset(forbidden_ids)
        retrieval_kwargs: dict[str, object] = {
            **ids_kwargs,
            "retrieved_ids": retrieved,
            "recall_at_k": recall_at_k(retrieved, relevant, k=self._top_k),
            "precision_at_k": precision_at_k(retrieved, relevant, k=self._top_k),
            "reciprocal_rank": reciprocal_rank(retrieved, relevant),
            "hit": any(chunk_id in relevant for chunk_id in retrieved),
            "forbidden_hit": any(chunk_id in forbidden_set for chunk_id in retrieved),
        }

        if self._provider is None:
            return self._outcome(case, failure=StageFailure.NONE, **retrieval_kwargs)

        try:
            generated = self._provider.answer(case, outcome.pack, gold=gold, forbidden=forbidden)
        except Exception as exc:
            del exc
            return self._outcome(case, failure=StageFailure.GENERATION, **retrieval_kwargs)

        try:
            evaluation = self._evaluate(case, outcome.pack, generated, gold)
        except Exception as exc:
            del exc
            return self._outcome(case, failure=StageFailure.EVALUATION, **retrieval_kwargs)

        return self._outcome_with_answer(case, evaluation, generated, **retrieval_kwargs)

    def _evaluate(
        self,
        case: GroundedCase,
        pack: EvidencePack,
        generated: GeneratedAnswer,
        gold: tuple[ChunkId, ...],
    ) -> AnswerEvaluation:
        answer_id = f"{case.case_id}:{generated.label}"
        claims = StructuredClaimExtractor(generated.claims).extract(answer_id, generated.text)
        requires = frozenset(claim.claim_id for claim in claims) if case.answerable else frozenset()
        expected = {claim.claim_id: gold for claim in claims}
        return evaluate_answer(
            answer_id,
            claims,
            pack,
            requires_citation=requires,
            expected_evidence=expected,
        )

    def _outcome(
        self,
        case: GroundedCase,
        *,
        failure: StageFailure,
        **retrieval_kwargs: object,
    ) -> GroundedCaseOutcome:
        return GroundedCaseOutcome(
            case_id=str(case.case_id),
            category=case.category,
            split=case.split,
            case_version=case.version,
            failure=failure,
            abstention_expected=case.expected_behavior is ExpectedBehavior.ABSTAIN,
            **retrieval_kwargs,  # type: ignore[arg-type]
        )

    def _outcome_with_answer(
        self,
        case: GroundedCase,
        evaluation: AnswerEvaluation,
        generated: GeneratedAnswer,
        **retrieval_kwargs: object,
    ) -> GroundedCaseOutcome:
        citations = evaluation.citations
        text = generated.text
        expected_abstain = case.expected_behavior is ExpectedBehavior.ABSTAIN
        observed_abstain = generated.abstained
        return GroundedCaseOutcome(
            case_id=str(case.case_id),
            category=case.category,
            split=case.split,
            case_version=case.version,
            failure=StageFailure.NONE,
            **retrieval_kwargs,  # type: ignore[arg-type]
            answer_text=text,
            answer_label=generated.label,
            claims=citations.claims,
            citation_valid=citations.valid,
            citation_unknown=citations.unknown,
            citation_irrelevant=citations.irrelevant,
            citation_uncited=citations.uncited_claims,
            supported_claims=len(evaluation.grounding.supported),
            unsupported_claims=len(evaluation.grounding.unsupported),
            evidence_coverage=evaluation.evidence_coverage,
            is_grounded=evaluation.is_grounded,
            required_facts_present=sum(
                1 for fact in case.required_facts if fact.lower() in text.lower()
            ),
            required_facts_total=len(case.required_facts),
            prohibited_present=sum(
                1 for claim in case.prohibited_claims if claim.lower() in text.lower()
            ),
            prohibited_total=len(case.prohibited_claims),
            abstention_expected=expected_abstain,
            abstention_observed=observed_abstain,
            abstention_correct=expected_abstain == observed_abstain,
        )


def _aggregate_retrieval(outcomes: tuple[GroundedCaseOutcome, ...]) -> GroundedRetrievalMetrics:
    scored = tuple(o for o in outcomes if o.has_gold)
    abstentions = tuple(o for o in outcomes if not o.has_gold)
    return GroundedRetrievalMetrics(
        scored_cases=len(scored),
        abstention_cases=len(abstentions),
        mean_recall_at_k=mean(tuple(o.recall_at_k for o in scored)),
        mean_precision_at_k=mean(tuple(o.precision_at_k for o in scored)),
        mrr=mean(tuple(o.reciprocal_rank for o in scored)),
        hit_rate=mean(tuple(1.0 if o.hit else 0.0 for o in scored)),
        forbidden_hits=sum(1 for o in outcomes if o.forbidden_hit),
        false_retrievals=sum(1 for o in abstentions if o.retrieved_ids),
    )


def _aggregate_answer(outcomes: tuple[GroundedCaseOutcome, ...]) -> GroundedAnswerMetrics:
    scored = tuple(
        o for o in outcomes if o.failure is StageFailure.NONE and o.answer_text is not None
    )
    answered = tuple(o for o in scored if o.abstention_observed is False)
    valid = sum(o.citation_valid for o in scored)
    unknown = sum(o.citation_unknown for o in scored)
    irrelevant = sum(o.citation_irrelevant for o in scored)
    unsupported = sum(o.unsupported_claims for o in scored)
    total_claims = sum(o.claims for o in scored)
    facts_present = sum(o.required_facts_present for o in scored)
    facts_total = sum(o.required_facts_total for o in scored)
    prohibited_present = sum(o.prohibited_present for o in scored)
    prohibited_total = sum(o.prohibited_total for o in scored)
    decisions = tuple(o for o in scored if o.abstention_correct is not None)
    appropriate = sum(1 for o in decisions if o.abstention_correct)
    grounded = sum(1 for o in answered if o.is_grounded)
    # Evidence coverage is a per-case mean over *answered* cases only: an abstention has no claims,
    # so folding its definitionally-zero coverage into the mean would conflate "declined to answer"
    # with "answered with poorly-grounded claims" — the conflation the separate dimensions exist to
    # avoid. Abstention quality is scored by the appropriate-abstention rate, not here.
    coverage_values = tuple(o.evidence_coverage or 0.0 for o in answered)
    return GroundedAnswerMetrics(
        answered_cases=len(answered),
        mean_evidence_coverage=mean(coverage_values),
        citation_validity=citation_validity_rate(
            valid=valid, unknown=unknown, irrelevant=irrelevant
        ),
        unsupported_claim_rate=unsupported_claim_rate(
            unsupported=unsupported, total_claims=total_claims
        ),
        required_fact_coverage=required_fact_coverage(
            facts_present=facts_present, facts_required=facts_total
        ),
        prohibited_claim_rate=prohibited_claim_rate(
            prohibited_present=prohibited_present, prohibited_total=prohibited_total
        ),
        appropriate_abstention_rate=rate(appropriate, len(decisions)),
        grounded_answer_rate=rate(grounded, len(answered)),
    )


def _aggregate_failures(outcomes: tuple[GroundedCaseOutcome, ...]) -> FailureCounts:
    return FailureCounts(
        retrieval=sum(1 for o in outcomes if o.failure is StageFailure.RETRIEVAL),
        generation=sum(1 for o in outcomes if o.failure is StageFailure.GENERATION),
        evaluation=sum(1 for o in outcomes if o.failure is StageFailure.EVALUATION),
    )


def _aggregate_by_category(
    outcomes: tuple[GroundedCaseOutcome, ...],
) -> Mapping[str, Mapping[str, float]]:
    buckets: dict[str, list[GroundedCaseOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.category.value, []).append(outcome)
    per_category: dict[str, Mapping[str, float]] = {}
    for category, group in sorted(buckets.items()):
        scored = tuple(o for o in group if o.has_gold)
        per_category[category] = MappingProxyType(
            {
                "cases": float(len(group)),
                "scored": float(len(scored)),
                "mean_recall_at_k": mean(tuple(o.recall_at_k for o in scored)),
                "forbidden_hits": float(sum(1 for o in group if o.forbidden_hit)),
                "failures": float(sum(1 for o in group if o.failure is not StageFailure.NONE)),
            }
        )
    return MappingProxyType(per_category)
