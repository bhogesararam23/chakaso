"""Evaluation: metric functions, not a benchmark.

This package exists to make retrieval *measurable*, not to report measurements. There is no
benchmark, no evaluation dataset, no task version and no measured number here or anywhere
in the repository — inventing a figure would be the fabrication the project's rules forbid.
What is provided are pure, deterministic metric functions that a real benchmark can be run
through once one exists.

The full evaluation layer — dataset, harness, regression gates — is later work
(``docs/evaluation.md``). This is the smallest honest piece of it.
"""

from __future__ import annotations

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
    "duplicate_count",
    "mean_reciprocal_rank",
    "measure_latency",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "unresolved_reference_count",
]
