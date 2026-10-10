"""Canonical development experiments.

These are the concrete ``Experiment`` records the experiment layer runs by default — data, not
behaviour, so the CLI and tests can ask for "the retrieval baseline" without each inventing its own
hypothesis string. The hypotheses are written honestly: each states what is measurable over the
synthetic development benchmark, and none claims real-world performance or a model capability that
does not exist.
"""

from __future__ import annotations

from chakaso.experiments.model import (
    BenchmarkKind,
    Experiment,
    ExperimentId,
    InvalidExperimentError,
)

__all__ = ["baseline_experiment", "retrieval_strategy_experiment"]

_RETRIEVAL_BASELINE = Experiment(
    experiment_id=ExperimentId("dev-retrieval-baseline"),
    name="Retrieval development baseline",
    hypothesis=(
        "the lexical BM25 retriever reaches a measurable recall@k, precision@k and MRR "
        "over the synthetic retrieval development benchmark"
    ),
    benchmark_kind=BenchmarkKind.RETRIEVAL,
    configuration={"retriever": "lexical", "top_k": "5"},
    notes="Development instrument over hand-built fixtures; not a claim about real-corpus retrieval.",
)

_CORRECTION_BASELINE = Experiment(
    experiment_id=ExperimentId("dev-correction-baseline"),
    name="Correction decision-rule baseline",
    hypothesis=(
        "the written correction decision rule reaches the expected retain/qualify/correct/"
        "abstain/needs-review outcome on every curated synthetic case"
    ),
    benchmark_kind=BenchmarkKind.CORRECTION,
    configuration={"policy": "adr-0016", "grounding": "structural+fixture-semantic"},
    notes="Scores the rule over fixtures; no model runs and no contradiction is inferred.",
)


def baseline_experiment(kind: BenchmarkKind) -> Experiment:
    """The canonical development experiment for a benchmark kind."""
    if kind is BenchmarkKind.RETRIEVAL:
        return _RETRIEVAL_BASELINE
    return _CORRECTION_BASELINE


#: Honest, per-strategy hypotheses. Each states what is measurable over the synthetic
#: development benchmark for that retriever; none claims real-corpus or semantic quality, and
#: the dense/hybrid entries say in the hypothesis itself that they run over a non-semantic
#: fixture embedding.
_STRATEGY_HYPOTHESES = {
    "lexical": (
        "the lexical BM25 retriever reaches a measurable recall@k, precision@k and MRR "
        "over the synthetic retrieval development benchmark"
    ),
    "dense": (
        "the exact dense retriever, over a non-semantic fixture embedding, can be scored on "
        "recall@k, precision@k and MRR over the synthetic retrieval development benchmark"
    ),
    "hybrid": (
        "reciprocal-rank fusion of lexical and dense (fixture embedding) can be scored on the "
        "synthetic retrieval development benchmark"
    ),
}


def retrieval_strategy_experiment(strategy: str, *, top_k: int = 5) -> Experiment:
    """The development experiment that measures one retrieval strategy at a given top-k.

    Raises:
        InvalidExperimentError: ``strategy`` is not one of the backed retrieval strategies.
    """
    if strategy not in _STRATEGY_HYPOTHESES:
        known = ", ".join(sorted(_STRATEGY_HYPOTHESES))
        message = f"unknown retrieval strategy {strategy!r}; expected one of: {known}"
        raise InvalidExperimentError(message)
    return Experiment(
        experiment_id=ExperimentId(f"dev-retrieval-{strategy}"),
        name=f"Retrieval {strategy} strategy measurement",
        hypothesis=_STRATEGY_HYPOTHESES[strategy],
        benchmark_kind=BenchmarkKind.RETRIEVAL,
        configuration={"retriever": strategy, "top_k": str(top_k)},
        notes=(
            "Development measurement over hand-built fixtures; dense/hybrid use the non-semantic "
            "fixture embedding, so results describe mechanism, not real-corpus or semantic quality."
        ),
    )
