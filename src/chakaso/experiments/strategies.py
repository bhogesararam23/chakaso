"""A measured comparison of retrieval strategies over the shared development benchmark.

``compare_results`` (in :mod:`chakaso.experiments.compare`) is a *regression* tool: it refuses to
compare results whose configuration differs, because a config difference usually means two different
questions. Comparing **strategies** is the opposite — the strategy *is* the controlled variable: the
same dataset, the same top-k, only the retriever changing. This module runs each strategy over the
identical synthetic dataset and reports the numbers side by side, plus the delta against a chosen
baseline strategy, so the honest question — "did dense or hybrid change anything here?" — has a
measured answer rather than an assumed one.

It reports no overall winner and hard-codes no preferred result: it lists what each strategy scored
and how each differs from the baseline, per metric, in the direction that metric moves when it
improves. The dense and hybrid strategies run over the non-semantic fixture embedding
(ADR-0023/ADR-0024), so any difference here reflects the fusion and ranking mechanics over synthetic
fixtures — never a claim that dense or hybrid understand meaning, and never a result invented rather
than run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chakaso.benchmark import RetrievalBenchmarkRunner
from chakaso.benchmark.dataset import BenchmarkDataset
from chakaso.planning import QueryMode
from chakaso.retrieval import DEFAULT_TOP_K, Corpus, RetrievalService
from chakaso.retrieval.embeddings import EmbeddingModel
from chakaso.runtime import build_retrieval_services

__all__ = [
    "RetrievalStrategyReport",
    "StrategyDelta",
    "StrategyMetrics",
    "compare_retrieval_strategies",
]

#: Which way each reported metric moving counts as better. Anything unlisted would be
#: informational, but every metric here is deliberately classified.
_HIGHER_IS_BETTER = frozenset({"recall_at_k", "precision_at_k", "mrr", "hit_rate"})
_LOWER_IS_BETTER = frozenset({"forbidden_hits", "false_retrievals", "duplicates"})

#: The numeric metrics compared across strategies, in a fixed order for determinism.
_COMPARED = (
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "hit_rate",
    "forbidden_hits",
    "false_retrievals",
    "duplicates",
)


@dataclass(frozen=True, slots=True)
class StrategyMetrics:
    """One strategy's aggregate scores on the shared dataset."""

    strategy: str
    scored_cases: int
    abstention_cases: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    hit_rate: float
    forbidden_hits: int
    false_retrievals: int
    duplicates: int


@dataclass(frozen=True, slots=True)
class StrategyDelta:
    """How a strategy's metric differs from the baseline strategy's."""

    strategy: str
    metric: str
    baseline: float
    candidate: float
    delta: float
    direction: str  # improved | regressed | unchanged


@dataclass(frozen=True, slots=True)
class RetrievalStrategyReport:
    """The measured comparison: one row per strategy, deltas against the baseline."""

    dataset_id: str
    dataset_version: str
    dataset_content_fingerprint: str
    top_k: int
    baseline: str
    results: tuple[StrategyMetrics, ...]
    deltas: tuple[StrategyDelta, ...]

    def metrics_for(self, strategy: str) -> StrategyMetrics | None:
        """The row for ``strategy``, or ``None`` if it was not run."""
        return next((row for row in self.results if row.strategy == strategy), None)

    def to_dict(self) -> dict[str, object]:
        """A deterministic, JSON-serializable view for recording or diffing a run."""
        return {
            "dataset": {
                "dataset_id": self.dataset_id,
                "dataset_version": self.dataset_version,
                "dataset_content_fingerprint": self.dataset_content_fingerprint,
            },
            "top_k": self.top_k,
            "baseline": self.baseline,
            "results": [_metrics_to_dict(row) for row in self.results],
            "deltas": [_delta_to_dict(row) for row in self.deltas],
        }


def compare_retrieval_strategies(
    corpus: Corpus,
    dataset: BenchmarkDataset,
    *,
    strategies: Sequence[str] = ("lexical", "dense", "hybrid"),
    top_k: int = DEFAULT_TOP_K,
    baseline: str = "lexical",
    embedder: EmbeddingModel | None = None,
) -> RetrievalStrategyReport:
    """Score each strategy over ``dataset`` and report the measured deltas against ``baseline``.

    Raises:
        ValueError: an unknown strategy name, or a baseline that is not among ``strategies``.
    """
    services = build_retrieval_services(corpus, embedder=embedder)
    by_strategy = {name: services[QueryMode(name)] for name in _validated(strategies, baseline)}

    results = tuple(_score(name, service, top_k, dataset) for name, service in by_strategy.items())
    base_row = next(row for row in results if row.strategy == baseline)
    delta_rows = tuple(
        _delta(base_row, row, metric)
        for row in results
        if row.strategy != baseline
        for metric in _COMPARED
    )
    return RetrievalStrategyReport(
        dataset_id=str(dataset.dataset_id),
        dataset_version=dataset.version,
        dataset_content_fingerprint=dataset.content_fingerprint,
        top_k=top_k,
        baseline=baseline,
        results=results,
        deltas=delta_rows,
    )


def _validated(strategies: Sequence[str], baseline: str) -> tuple[str, ...]:
    names = tuple(strategies)
    for name in names:
        if name not in {
            mode.value for mode in (QueryMode.LEXICAL, QueryMode.DENSE, QueryMode.HYBRID)
        }:
            message = f"unknown retrieval strategy {name!r}"
            raise ValueError(message)
    if baseline not in names:
        message = f"baseline strategy {baseline!r} must be among the compared strategies"
        raise ValueError(message)
    return names


def _score(
    name: str, service: RetrievalService, top_k: int, dataset: BenchmarkDataset
) -> StrategyMetrics:
    metrics = (
        RetrievalBenchmarkRunner(service, top_k=top_k, retriever_name=name).run(dataset).metrics
    )
    return StrategyMetrics(
        strategy=name,
        scored_cases=metrics.scored_cases,
        abstention_cases=metrics.abstention_cases,
        recall_at_k=metrics.mean_recall_at_k,
        precision_at_k=metrics.mean_precision_at_k,
        mrr=metrics.mrr,
        hit_rate=metrics.hit_rate,
        forbidden_hits=metrics.forbidden_hits,
        false_retrievals=metrics.false_retrievals,
        duplicates=metrics.duplicates,
    )


def _delta(base: StrategyMetrics, row: StrategyMetrics, metric: str) -> StrategyDelta:
    baseline_value = float(getattr(base, metric))
    candidate_value = float(getattr(row, metric))
    diff = candidate_value - baseline_value
    return StrategyDelta(
        strategy=row.strategy,
        metric=metric,
        baseline=baseline_value,
        candidate=candidate_value,
        delta=diff,
        direction=_direction(metric, diff),
    )


def _direction(metric: str, delta: float) -> str:
    if delta == 0:
        return "unchanged"
    if metric in _HIGHER_IS_BETTER:
        return "improved" if delta > 0 else "regressed"
    if metric in _LOWER_IS_BETTER:
        return "improved" if delta < 0 else "regressed"
    return "informational"


def _metrics_to_dict(row: StrategyMetrics) -> dict[str, object]:
    return {
        "strategy": row.strategy,
        "scored_cases": row.scored_cases,
        "abstention_cases": row.abstention_cases,
        "recall_at_k": row.recall_at_k,
        "precision_at_k": row.precision_at_k,
        "mrr": row.mrr,
        "hit_rate": row.hit_rate,
        "forbidden_hits": row.forbidden_hits,
        "false_retrievals": row.false_retrievals,
        "duplicates": row.duplicates,
    }


def _delta_to_dict(row: StrategyDelta) -> dict[str, object]:
    return {
        "strategy": row.strategy,
        "metric": row.metric,
        "baseline": row.baseline,
        "candidate": row.candidate,
        "delta": row.delta,
        "direction": row.direction,
    }
