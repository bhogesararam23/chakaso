"""Evaluation: metric functions and the answer-evaluation combiner.

This package turns measured results into numbers and combines the answer's separate
quality dimensions. It reports no headline score and no model-quality result: there is no
calibrated evaluator or general benchmark here, and inventing a figure would be the
fabrication the project's rules forbid. The versioned retrieval *benchmark* the metric
functions run against lives in `chakaso.benchmark`; `answer.py` composes citation and
grounding results into an `AnswerEvaluation` that keeps dimensions distinct.

The full evaluation layer — a model-quality benchmark, harness, regression gates — remains
later work (``docs/evaluation.md``). This is the honest, measurable subset of it.
``semantic_benchmark`` holds a versioned, hand-labelled synthetic set and a deterministic scorer for
a semantic judge — a development instrument that measures the harness, never semantic capability, and
whose only available judge today is a fixture.
"""

from __future__ import annotations

from chakaso.evaluation.answer import AnswerEvaluation, evaluate_answer
from chakaso.evaluation.metrics import (
    duplicate_count,
    mean_reciprocal_rank,
    measure_latency,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unresolved_reference_count,
)
from chakaso.evaluation.semantic_benchmark import (
    SEMANTIC_DATASET_VERSION,
    SemanticBenchmarkRun,
    SemanticCase,
    SemanticCaseOutcome,
    dataset_fingerprint,
    development_semantic_dataset,
    format_semantic_report,
    score_semantic_judge,
    semantic_report_json,
)

__all__ = [
    "SEMANTIC_DATASET_VERSION",
    "AnswerEvaluation",
    "SemanticBenchmarkRun",
    "SemanticCase",
    "SemanticCaseOutcome",
    "dataset_fingerprint",
    "development_semantic_dataset",
    "duplicate_count",
    "evaluate_answer",
    "format_semantic_report",
    "mean_reciprocal_rank",
    "measure_latency",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "score_semantic_judge",
    "semantic_report_json",
    "unresolved_reference_count",
]
