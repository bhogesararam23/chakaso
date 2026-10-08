"""Experiments: reproducible research artifacts built on the benchmarks.

An experiment pairs a hypothesis with a development benchmark and a configuration; running it
produces a content-fingerprinted, serializable result with environment metadata kept separate from
the measured numbers (ADR-0020). A baseline comparison can then answer, deterministically and only
between compatible benchmark/evaluator versions, whether a component got better or worse
(``compare_results``). No model runs, so every number is a development measurement over synthetic
fixtures — not a claim about real-world performance, and nothing here fabricates a result.
"""

from __future__ import annotations

from chakaso.experiments.model import (
    BenchmarkKind,
    Experiment,
    ExperimentError,
    ExperimentId,
    ExperimentResult,
    InvalidExperimentError,
)
from chakaso.experiments.runner import run_experiment

__all__ = [
    "BenchmarkKind",
    "Experiment",
    "ExperimentError",
    "ExperimentId",
    "ExperimentResult",
    "InvalidExperimentError",
    "run_experiment",
]
