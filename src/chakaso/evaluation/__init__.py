"""Evaluation: metric functions and the answer-evaluation combiner.

This package turns measured results into numbers and combines the answer's separate
quality dimensions. It reports no headline score and no model-quality result: there is no
calibrated evaluator or general benchmark here, and inventing a figure would be the
fabrication the project's rules forbid. The versioned retrieval *benchmark* the metric
functions run against lives in `chakaso.benchmark`; `answer.py` composes citation and
grounding results into an `AnswerEvaluation` that keeps dimensions distinct.

The full evaluation layer — a model-quality benchmark, harness, regression gates — remains
later work (``docs/evaluation.md``). This is the honest, measurable subset of it.
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

__all__ = [
    "AnswerEvaluation",
    "duplicate_count",
    "evaluate_answer",
    "mean_reciprocal_rank",
    "measure_latency",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "unresolved_reference_count",
]
