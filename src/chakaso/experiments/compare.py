"""A deterministic regression baseline: did a measured component get better or worse?

The project's standing rule is evaluation before scaling, which needs a way to compare a run
against a previous one. ``compare_results`` is deliberately a *simple, deterministic* comparison,
not a statistical test: these are small synthetic development benchmarks, and dressing a delta over
ten fixtures in a confidence interval would be theatre (``docs/evaluation.md`` forbids it).

Compatibility is the whole point of putting this behind a function rather than subtracting two
dicts. A metric only measures the thing that defined it, so a comparison is meaningful only when
the two results address the same experiment, the same benchmark identity and version, the same
configuration and the same evaluator set. A difference in any of those means the two numbers answer
different questions, and this refuses rather than interpolating a misleading delta. The
implementation version is allowed to differ — that is precisely the change a regression check is
looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from chakaso.experiments.model import ExperimentError, ExperimentResult

__all__ = [
    "ComparisonDirection",
    "ExperimentComparison",
    "IncompatibleExperimentError",
    "MetricComparison",
    "compare_results",
]

#: Which way a metric moving is good. Anything not listed is compared but reported as
#: ``informational`` — a case count is not "better" when it goes up, it just is.
_HIGHER_IS_BETTER = frozenset(
    {
        "recall_at_k",
        "precision_at_k",
        "mrr",
        "hit_rate",
        "decision_accuracy",
        "appropriate_abstention_rate",
        "appropriate_review_rate",
    }
)
_LOWER_IS_BETTER = frozenset(
    {
        "unjustified_persistence_rate",
        "unnecessary_revision_rate",
        "forbidden_hits",
        "false_retrievals",
        "duplicates",
    }
)


class IncompatibleExperimentError(ExperimentError):
    """Two results do not describe the same experiment/benchmark and cannot be compared."""


class ComparisonDirection(StrEnum):
    """How a metric moved, in terms of whether that is an improvement."""

    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    INFORMATIONAL = "informational"
    NOT_COMPARABLE = "not_comparable"


@dataclass(frozen=True, slots=True)
class MetricComparison:
    """One metric's baseline, candidate, delta and direction."""

    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    direction: ComparisonDirection


@dataclass(frozen=True, slots=True)
class ExperimentComparison:
    """The outcome of comparing a candidate result against a baseline result."""

    experiment_id: str
    benchmark_kind: str
    comparisons: tuple[MetricComparison, ...]

    @property
    def regressions(self) -> tuple[str, ...]:
        """Metrics that got worse; empty means nothing regressed."""
        return tuple(
            c.metric for c in self.comparisons if c.direction is ComparisonDirection.REGRESSED
        )

    @property
    def improvements(self) -> tuple[str, ...]:
        """Metrics that got better."""
        return tuple(
            c.metric for c in self.comparisons if c.direction is ComparisonDirection.IMPROVED
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "benchmark_kind": self.benchmark_kind,
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "comparisons": [
                {
                    "metric": c.metric,
                    "baseline": c.baseline,
                    "candidate": c.candidate,
                    "delta": c.delta,
                    "direction": c.direction.value,
                }
                for c in self.comparisons
            ],
        }


def compare_results(
    baseline: ExperimentResult, candidate: ExperimentResult
) -> ExperimentComparison:
    """Compare ``candidate`` against ``baseline``, refusing incompatible pairs.

    Raises:
        IncompatibleExperimentError: the results differ in experiment, benchmark identity or
            version, configuration, or evaluator set — comparing them would be comparing different
            questions, so it is refused rather than silently done.
    """
    _require_compatible(baseline, candidate)
    comparisons = tuple(
        _compare_metric(metric, baseline.metrics[metric], candidate.metrics[metric])
        for metric in sorted(baseline.metrics.keys() & candidate.metrics.keys())
    )
    return ExperimentComparison(
        experiment_id=str(baseline.experiment_id),
        benchmark_kind=baseline.benchmark_kind.value,
        comparisons=comparisons,
    )


def _require_compatible(baseline: ExperimentResult, candidate: ExperimentResult) -> None:
    if baseline.experiment_id != candidate.experiment_id:
        message = f"cannot compare different experiments {baseline.experiment_id} and {candidate.experiment_id}"
        raise IncompatibleExperimentError(message)
    if baseline.benchmark_kind is not candidate.benchmark_kind:
        message = "cannot compare results from different benchmarks"
        raise IncompatibleExperimentError(message)
    for field_name in ("dataset_id", "dataset_version", "dataset_content_fingerprint"):
        if getattr(baseline, field_name) != getattr(candidate, field_name):
            message = f"cannot compare across differing {field_name}; the benchmarks differ"
            raise IncompatibleExperimentError(message)
    if baseline.configuration != candidate.configuration:
        message = "cannot compare across differing configuration"
        raise IncompatibleExperimentError(message)
    if baseline.evaluators != candidate.evaluators:
        message = "cannot compare across differing evaluators"
        raise IncompatibleExperimentError(message)


def _compare_metric(metric: str, base: float | None, cand: float | None) -> MetricComparison:
    if base is None or cand is None:
        return MetricComparison(metric, base, cand, None, ComparisonDirection.NOT_COMPARABLE)
    delta = cand - base
    return MetricComparison(metric, base, cand, delta, _direction(metric, delta))


def _direction(metric: str, delta: float) -> ComparisonDirection:
    if delta == 0:
        return ComparisonDirection.UNCHANGED
    if metric in _HIGHER_IS_BETTER:
        return ComparisonDirection.IMPROVED if delta > 0 else ComparisonDirection.REGRESSED
    if metric in _LOWER_IS_BETTER:
        return ComparisonDirection.IMPROVED if delta < 0 else ComparisonDirection.REGRESSED
    return ComparisonDirection.INFORMATIONAL
