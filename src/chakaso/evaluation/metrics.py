"""Metric functions for retrieval and citation evaluation.

These are the measurement hooks: pure, deterministic functions over ranked identifiers and
a relevance judgement. They compute what a well-defined quantity is, and nothing more.

This is deliberately *not* the evaluation framework. There is no benchmark, no dataset, no
task version and no measured number anywhere in this repository — inventing one would be
exactly the fabrication the project forbids. What these functions allow is the step that
comes *after* a benchmark exists: given a ranked result and a set of relevant identifiers,
compute a metric. Which inputs are relevant is a judgement a benchmark supplies, not
something this module asserts.

Metrics here have unambiguous definitions. Pipeline-level measures listed in the plan that
need a ground-truth definition first — evidence coverage, for instance — are not invented
into a formula; they wait for the benchmark that says what "relevant" means.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from time import perf_counter
from typing import Final

from chakaso.evidence import CitationResolution

__all__ = [
    "duplicate_count",
    "mean_reciprocal_rank",
    "measure_latency",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "unresolved_reference_count",
]

#: An empty relevance set makes recall undefined; it is reported as this value rather than
#: as a division by zero or a fabricated 1.0.
_NO_RELEVANT: Final[float] = 0.0


def _require_k(k: int) -> None:
    if k < 1:
        message = f"a cut-off k must be at least 1, got {k}"
        raise ValueError(message)


def recall_at_k(retrieved: Sequence[str], relevant: frozenset[str], *, k: int) -> float:
    """The fraction of relevant identifiers that appear in the top ``k`` results.

    A relevance set with no members has no defined recall; it returns ``0.0`` rather than
    inventing a value.
    """
    _require_k(k)
    if not relevant:
        return _NO_RELEVANT
    hits = sum(1 for identifier in retrieved[:k] if identifier in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: frozenset[str], *, k: int) -> float:
    """The fraction of the top ``k`` results that are relevant."""
    _require_k(k)
    window = retrieved[:k]
    if not window:
        return 0.0
    hits = sum(1 for identifier in window if identifier in relevant)
    return hits / len(window)


def reciprocal_rank(retrieved: Sequence[str], relevant: frozenset[str]) -> float:
    """One over the rank of the first relevant result, or ``0.0`` if none is relevant."""
    for position, identifier in enumerate(retrieved, start=1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(runs: Iterable[tuple[Sequence[str], frozenset[str]]]) -> float:
    """The mean reciprocal rank over ``(retrieved, relevant)`` runs.

    An empty collection of runs has no mean; it returns ``0.0``.
    """
    ranks = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in runs]
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def duplicate_count(retrieved: Sequence[str]) -> int:
    """How many repeated identifiers appear in a result list.

    A lexical baseline that returns the same chunk twice, or one that returns several
    near-identical copies, should be measurable as such — with non-overlapping,
    content-derived chunks, a duplicate identifier is a real signal, not an artefact.
    """
    return len(retrieved) - len(set(retrieved))


def unresolved_reference_count(resolution: CitationResolution) -> int:
    """How many references were not resolved: fabricated plus malformed.

    This is the count ADR-0003 and ADR-0009 make observable — a model referencing evidence
    it was not given, plus a model fumbling the identifier format.
    """
    return len(resolution.unknown_identifiers) + len(resolution.malformed_identifiers)


def measure_latency(function: Callable[[], object]) -> float:
    """Return the wall-clock seconds ``function`` took, and nothing else.

    This is an observation of one call, not a benchmark result. No latency figure is
    recorded anywhere as a property of the system; a number here means "this call took this
    long on this machine", which is the only honest claim available before a defined
    benchmark on fixed hardware.
    """
    start = perf_counter()
    function()
    return perf_counter() - start
