"""Rendering a benchmark run as a report, for a person and for a machine.

A run's numbers are only meaningful with their provenance attached, so every report
carries the dataset identity, version and content fingerprint, the configuration
(top-k, retriever), and the environment it was produced in. It also states, in its own
words, that it is a *development* benchmark over synthetic fixtures — a tiny fixed set
that cannot support a claim about general performance, and over which no confidence
interval or significance test is computed because doing so would be theatre.

The report renders retrieval scores only; it does not evaluate an answer, because there
is no model to answer. `report_to_dict` is deterministic and JSON-serializable so the
same run always produces the same bytes, apart from environment metadata that is
intentionally recorded.
"""

from __future__ import annotations

import json
from typing import Final

from chakaso.benchmark.runner import BenchmarkRun, CaseOutcome

__all__ = ["DEVELOPMENT_BENCHMARK_NOTE", "format_report", "report_json", "report_to_dict"]

#: Stated in every report so a number cannot be lifted out of context and read as a
#: general-capability claim.
DEVELOPMENT_BENCHMARK_NOTE: Final[str] = (
    "Development benchmark over a small, synthetic, hand-built fixture corpus. It scores "
    "retrieval only, evaluates no model answer, and is too small to support any claim "
    "about general performance. No confidence interval or significance test is reported."
)


def report_to_dict(run: BenchmarkRun) -> dict[str, object]:
    """Convert a run into a deterministic, JSON-serializable mapping."""
    return {
        "benchmark": {
            "kind": "development",
            "dataset_id": run.dataset_id,
            "dataset_version": run.dataset_version,
            "dataset_content_fingerprint": run.dataset_content_fingerprint,
            "note": DEVELOPMENT_BENCHMARK_NOTE,
        },
        "configuration": {
            "retriever": run.retriever_name,
            "top_k": run.top_k,
            **run.metadata,
        },
        "metrics": {
            "scored_cases": run.metrics.scored_cases,
            "abstention_cases": run.metrics.abstention_cases,
            "total_cases": run.metrics.total_cases,
            "recall_at_k": run.metrics.mean_recall_at_k,
            "precision_at_k": run.metrics.mean_precision_at_k,
            "mrr": run.metrics.mrr,
            "hit_rate": run.metrics.hit_rate,
            "forbidden_hits": run.metrics.forbidden_hits,
            "false_retrievals": run.metrics.false_retrievals,
            "duplicates": run.metrics.duplicates,
        },
        "cases": [_case_dict(outcome) for outcome in run.outcomes],
    }


def _case_dict(outcome: CaseOutcome) -> dict[str, object]:
    return {
        "case_id": str(outcome.case_id),
        "category": outcome.category.value,
        "case_version": outcome.case_version,
        "retrieved_ids": list(outcome.retrieved_ids),
        "gold_ids": list(outcome.gold_ids),
        "forbidden_ids": list(outcome.forbidden_ids),
        "recall_at_k": outcome.recall_at_k,
        "precision_at_k": outcome.precision_at_k,
        "reciprocal_rank": outcome.reciprocal_rank,
        "hit": outcome.hit,
        "forbidden_hit": outcome.forbidden_hit,
        "abstained": outcome.abstained,
    }


def report_json(run: BenchmarkRun) -> str:
    """Render the report as deterministic JSON (sorted keys, fixed separators)."""
    return json.dumps(report_to_dict(run), sort_keys=True, indent=2, ensure_ascii=False)


def format_report(run: BenchmarkRun) -> str:
    """Render the report as human-readable text."""
    lines = [
        f"Retrieval benchmark report — {run.dataset_id} v{run.dataset_version}",
        f"content fingerprint: {run.dataset_content_fingerprint}",
        f"retriever: {run.retriever_name}  top_k: {run.top_k}",
        DEVELOPMENT_BENCHMARK_NOTE,
        "",
        _format_metrics(run),
        "",
        "Per case:",
    ]
    for outcome in run.outcomes:
        verdict = "hit" if outcome.hit else ("abstained" if outcome.abstained else "miss")
        flag = " [FORBIDDEN HIT]" if outcome.forbidden_hit else ""
        lines.append(
            f"  {outcome.case_id} ({outcome.category.value}): {verdict}"
            f"  recall@k={outcome.recall_at_k:.2f}{flag}"
        )
    return "\n".join(lines)


def _format_metrics(run: BenchmarkRun) -> str:
    metrics = run.metrics
    scored = metrics.scored_cases
    return "\n".join(
        [
            f"Scored cases: {scored}   Abstention cases: {metrics.abstention_cases}",
            f"Recall@k: {metrics.mean_recall_at_k:.4f}   Precision@k: {metrics.mean_precision_at_k:.4f}",
            f"MRR: {metrics.mrr:.4f}   Hit rate: {metrics.hit_rate:.4f}",
            f"Forbidden hits: {metrics.forbidden_hits}   False retrievals: {metrics.false_retrievals}"
            f"   Duplicates: {metrics.duplicates}",
        ]
    )
