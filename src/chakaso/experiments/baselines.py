"""Canonical development experiments.

These are the concrete ``Experiment`` records the experiment layer runs by default — data, not
behaviour, so the CLI and tests can ask for "the retrieval baseline" without each inventing its own
hypothesis string. The hypotheses are written honestly: each states what is measurable over the
synthetic development benchmark, and none claims real-world performance or a model capability that
does not exist.
"""

from __future__ import annotations

from chakaso.experiments.model import BenchmarkKind, Experiment, ExperimentId

__all__ = ["baseline_experiment"]

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
