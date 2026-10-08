"""The retrieval benchmark runner: cases in, a measured run out.

It wires the pieces that already exist — a `BenchmarkDataset`, a `RetrievalService`, the
metric functions in `chakaso.evaluation` — and records, per case, what retrieval returned
against what was expected. It computes nothing it has not been given and asserts nothing
about answer quality: it scores retrieval only. A model produces no answer here (the only
model is a development double), so nothing downstream of retrieval is claimed.

Metric semantics are fixed explicitly, because a metric whose definition shifts is worse
than no metric:

* recall@k, precision@k and MRR are averaged over *scored* cases — those that name at least
  one gold chunk. A case that expects no evidence (a missing-evidence or false-premise case)
  has no defined recall, and folding it into the mean at zero would report retrieval as worse
  than it was.
* those no-gold cases are scored separately as abstentions: did retrieval correctly return
  nothing. A no-gold case that returns chunks is a false positive, counted as such, not a
  zero that drags down recall.
* a forbidden hit — retrieving evidence the case marks unacceptable — is reported on its own,
  because precision does not catch a confidently-wrong citation on a case that also has a
  correct one.

The run carries the dataset identity, version and content fingerprint plus its
configuration, so a number can be traced to the exact judgements and settings behind it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from chakaso.benchmark.cases import BenchmarkCase, BenchmarkCategory
from chakaso.benchmark.dataset import BenchmarkDataset
from chakaso.benchmark.identity import CaseId
from chakaso.evaluation import (
    duplicate_count,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from chakaso.retrieval import DEFAULT_TOP_K, RetrievalService

__all__ = ["BenchmarkRun", "CaseOutcome", "RetrievalBenchmarkRunner", "RetrievalMetrics"]


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    """What retrieval returned for one case, and how it scored."""

    case_id: CaseId
    category: BenchmarkCategory
    case_version: str
    retrieved_ids: tuple[str, ...]
    gold_ids: tuple[str, ...]
    forbidden_ids: tuple[str, ...]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    hit: bool
    forbidden_hit: bool

    @property
    def has_gold(self) -> bool:
        """Whether this case names gold evidence (and so is scored for retrieval)."""
        return bool(self.gold_ids)

    @property
    def abstained(self) -> bool:
        """Whether a no-gold case correctly retrieved nothing."""
        return not self.has_gold and not self.retrieved_ids


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Aggregates over a run, with scored and abstention cases kept separate."""

    scored_cases: int
    abstention_cases: int
    mean_recall_at_k: float
    mean_precision_at_k: float
    mrr: float
    hit_rate: float
    forbidden_hits: int
    false_retrievals: int
    duplicates: int

    @property
    def total_cases(self) -> int:
        """Scored plus abstention cases — the whole run."""
        return self.scored_cases + self.abstention_cases


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    """A completed run: its provenance, per-case outcomes, and aggregates."""

    dataset_id: str
    dataset_version: str
    dataset_content_fingerprint: str
    top_k: int
    retriever_name: str
    outcomes: tuple[CaseOutcome, ...]
    metrics: RetrievalMetrics
    metadata: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))


class RetrievalBenchmarkRunner:
    """Runs a dataset against a retrieval service and scores what came back."""

    def __init__(
        self,
        service: RetrievalService,
        *,
        top_k: int = DEFAULT_TOP_K,
        retriever_name: str = "lexical",
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self._service = service
        self._top_k = top_k
        self._retriever_name = retriever_name
        self._metadata = MappingProxyType(dict(metadata or {}))

    def run(self, dataset: BenchmarkDataset) -> BenchmarkRun:
        """Retrieve for every case and compute the run's metrics."""
        outcomes = tuple(self._run_case(case) for case in dataset.cases)
        return BenchmarkRun(
            dataset_id=str(dataset.dataset_id),
            dataset_version=dataset.version,
            dataset_content_fingerprint=dataset.content_fingerprint,
            top_k=self._top_k,
            retriever_name=self._retriever_name,
            outcomes=outcomes,
            metrics=_aggregate(outcomes),
            metadata=self._metadata,
        )

    def _run_case(self, case: BenchmarkCase) -> CaseOutcome:
        outcome = self._service.search(case.query, top_k=self._top_k)
        retrieved = tuple(str(result.chunk.chunk_id) for result in outcome.results)
        gold = tuple(str(identifier) for identifier in case.gold_chunk_ids)
        forbidden = tuple(str(identifier) for identifier in case.forbidden_chunk_ids)
        relevant = frozenset(gold)
        forbidden_set = frozenset(forbidden)

        return CaseOutcome(
            case_id=case.case_id,
            category=case.category,
            case_version=case.version,
            retrieved_ids=retrieved,
            gold_ids=gold,
            forbidden_ids=forbidden,
            recall_at_k=recall_at_k(retrieved, relevant, k=self._top_k),
            precision_at_k=precision_at_k(retrieved, relevant, k=self._top_k),
            reciprocal_rank=reciprocal_rank(retrieved, relevant),
            hit=any(identifier in relevant for identifier in retrieved),
            forbidden_hit=any(identifier in forbidden_set for identifier in retrieved),
        )


def _aggregate(outcomes: tuple[CaseOutcome, ...]) -> RetrievalMetrics:
    """Combine per-case outcomes, keeping scored and abstention cases distinct."""
    scored = tuple(outcome for outcome in outcomes if outcome.has_gold)
    abstentions = tuple(outcome for outcome in outcomes if not outcome.has_gold)

    mean_recall = _mean(tuple(outcome.recall_at_k for outcome in scored))
    mean_precision = _mean(tuple(outcome.precision_at_k for outcome in scored))
    hits = tuple(1.0 if outcome.hit else 0.0 for outcome in scored)

    return RetrievalMetrics(
        scored_cases=len(scored),
        abstention_cases=len(abstentions),
        mean_recall_at_k=mean_recall,
        mean_precision_at_k=mean_precision,
        mrr=_mean(tuple(outcome.reciprocal_rank for outcome in scored)) if scored else 0.0,
        hit_rate=_mean(hits),
        forbidden_hits=sum(1 for outcome in outcomes if outcome.forbidden_hit),
        false_retrievals=sum(1 for outcome in abstentions if outcome.retrieved_ids),
        duplicates=sum(1 for outcome in outcomes if duplicate_count(outcome.retrieved_ids) > 0),
    )


def _mean(values: tuple[float, ...]) -> float:
    """Arithmetic mean of ``values``, or ``0.0`` for an empty collection."""
    return sum(values) / len(values) if values else 0.0
