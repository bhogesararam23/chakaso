"""Rendering a grounded-answer evaluation run, for a person and for a machine.

Every report carries the provenance a number needs (ADR-0031's reporting discipline): the dataset
identity, version and content fingerprint, the configuration (top-k, retriever, split), the provider
that produced the answers *and whether it was an oracle*, and whatever reproducibility metadata the
run recorded (code revision, environment, dependency versions). It renders the retrieval and answer
dimensions separately — never merged — so one cannot hide another.

The note stated in every report is the load-bearing honesty line: a run through the reference oracle
measures the harness, and a run through a non-answering provider measures that no model answers.
Neither is a claim about answer capability on real language, and no confidence interval or
significance test is computed over this tiny self-authored workload because doing so would be
theatre. The JSON is deterministic (sorted keys, fixed separators) so the same run always yields the
same bytes, apart from the environment metadata that is intentionally recorded.
"""

from __future__ import annotations

import json
from typing import Final

from chakaso.benchmark.grounded_runner import GroundedCaseOutcome, GroundedRun, StageFailure

__all__ = [
    "GROUNDED_EVALUATION_NOTE",
    "format_grounded_report",
    "grounded_report_json",
    "grounded_report_to_dict",
]

#: Stated in every report so a number cannot be lifted out and read as a capability claim.
GROUNDED_EVALUATION_NOTE: Final[str] = (
    "Development evaluation over a small, self-authored, human-inspectable workload. It scores "
    "retrieval and answer structure separately and merges them into no single quality score. A run "
    "through the reference oracle measures the harness, not a model; a run through the "
    "non-answering provider measures that no model answers. Neither is evidence of answer "
    "capability on real language, and no confidence interval or significance test is reported."
)


def _outcome_dict(outcome: GroundedCaseOutcome) -> dict[str, object]:
    return {
        "case_id": outcome.case_id,
        "category": outcome.category.value,
        "split": outcome.split.value,
        "case_version": outcome.case_version,
        "failure": outcome.failure.value,
        "retrieved_ids": list(outcome.retrieved_ids),
        "gold_ids": list(outcome.gold_ids),
        "forbidden_ids": list(outcome.forbidden_ids),
        "recall_at_k": outcome.recall_at_k,
        "precision_at_k": outcome.precision_at_k,
        "reciprocal_rank": outcome.reciprocal_rank,
        "hit": outcome.hit,
        "forbidden_hit": outcome.forbidden_hit,
        "answer_label": outcome.answer_label,
        "claims": outcome.claims,
        "citation_valid": outcome.citation_valid,
        "citation_unknown": outcome.citation_unknown,
        "citation_irrelevant": outcome.citation_irrelevant,
        "citation_uncited": outcome.citation_uncited,
        "supported_claims": outcome.supported_claims,
        "unsupported_claims": outcome.unsupported_claims,
        "evidence_coverage": outcome.evidence_coverage,
        "is_grounded": outcome.is_grounded,
        "required_facts_present": outcome.required_facts_present,
        "required_facts_total": outcome.required_facts_total,
        "prohibited_present": outcome.prohibited_present,
        "prohibited_total": outcome.prohibited_total,
        "abstention_expected": outcome.abstention_expected,
        "abstention_observed": outcome.abstention_observed,
        "abstention_correct": outcome.abstention_correct,
        "sufficiency_decision": outcome.sufficiency_decision,
        "sufficiency_correct": outcome.sufficiency_correct,
    }


def grounded_report_to_dict(run: GroundedRun) -> dict[str, object]:
    """Convert a run into a deterministic, JSON-serializable mapping."""
    return {
        "evaluation": {
            "kind": "development-grounded-answer",
            "dataset_id": run.dataset_id,
            "dataset_version": run.dataset_version,
            "dataset_content_fingerprint": run.dataset_content_fingerprint,
            "note": GROUNDED_EVALUATION_NOTE,
        },
        "configuration": {
            "retriever": run.retriever_name,
            "top_k": run.top_k,
            "split_filter": run.split_filter,
            "provider": run.provider_label,
            "provider_is_oracle": run.provider_is_oracle,
            **run.metadata,
        },
        "retrieval_metrics": {
            "scored_cases": run.retrieval_metrics.scored_cases,
            "abstention_cases": run.retrieval_metrics.abstention_cases,
            "recall_at_k": run.retrieval_metrics.mean_recall_at_k,
            "precision_at_k": run.retrieval_metrics.mean_precision_at_k,
            "mrr": run.retrieval_metrics.mrr,
            "hit_rate": run.retrieval_metrics.hit_rate,
            "forbidden_hits": run.retrieval_metrics.forbidden_hits,
            "false_retrievals": run.retrieval_metrics.false_retrievals,
        },
        "answer_metrics": {
            "answered_cases": run.answer_metrics.answered_cases,
            "mean_evidence_coverage": run.answer_metrics.mean_evidence_coverage,
            "citation_validity": run.answer_metrics.citation_validity,
            "unsupported_claim_rate": run.answer_metrics.unsupported_claim_rate,
            "required_fact_coverage": run.answer_metrics.required_fact_coverage,
            "prohibited_claim_rate": run.answer_metrics.prohibited_claim_rate,
            "appropriate_abstention_rate": run.answer_metrics.appropriate_abstention_rate,
            "grounded_answer_rate": run.answer_metrics.grounded_answer_rate,
        },
        "failures": {
            "retrieval": run.failures.retrieval,
            "generation": run.failures.generation,
            "evaluation": run.failures.evaluation,
            "total": run.failures.total,
        },
        "sufficiency_metrics": {
            "assessed_cases": run.sufficiency_metrics.assessed_cases,
            "accuracy": run.sufficiency_metrics.accuracy,
            "false_abstentions": run.sufficiency_metrics.false_abstentions,
            "missed_abstentions": run.sufficiency_metrics.missed_abstentions,
        },
        "per_category": {category: dict(values) for category, values in run.per_category.items()},
        "cases": [_outcome_dict(outcome) for outcome in run.outcomes],
    }


def grounded_report_json(run: GroundedRun) -> str:
    """Render the report as deterministic JSON (sorted keys, fixed separators)."""
    return json.dumps(grounded_report_to_dict(run), sort_keys=True, indent=2, ensure_ascii=False)


def format_grounded_report(run: GroundedRun) -> str:
    """Render the report as human-readable text."""
    oracle = (
        "oracle (harness self-check, not a model)" if run.provider_is_oracle else run.provider_label
    )
    lines = [
        f"Grounded-answer evaluation — {run.dataset_id} v{run.dataset_version}",
        f"content fingerprint: {run.dataset_content_fingerprint}",
        f"retriever: {run.retriever_name}  top_k: {run.top_k}  split: {run.split_filter}",
        f"answer provider: {oracle}",
        GROUNDED_EVALUATION_NOTE,
        "",
        "Retrieval:",
        f"  scored={run.retrieval_metrics.scored_cases} "
        f"abstention={run.retrieval_metrics.abstention_cases} "
        f"recall@k={run.retrieval_metrics.mean_recall_at_k:.4f} "
        f"precision@k={run.retrieval_metrics.mean_precision_at_k:.4f}",
        f"  mrr={run.retrieval_metrics.mrr:.4f} hit_rate={run.retrieval_metrics.hit_rate:.4f} "
        f"forbidden_hits={run.retrieval_metrics.forbidden_hits} "
        f"false_retrievals={run.retrieval_metrics.false_retrievals}",
        "",
        "Answer:",
        f"  answered={run.answer_metrics.answered_cases} "
        f"evidence_coverage={run.answer_metrics.mean_evidence_coverage:.4f} "
        f"citation_validity={run.answer_metrics.citation_validity:.4f}",
        f"  unsupported_claim_rate={run.answer_metrics.unsupported_claim_rate:.4f} "
        f"required_fact_coverage={run.answer_metrics.required_fact_coverage:.4f} "
        f"prohibited_claim_rate={run.answer_metrics.prohibited_claim_rate:.4f}",
        f"  appropriate_abstention_rate={run.answer_metrics.appropriate_abstention_rate:.4f} "
        f"grounded_answer_rate={run.answer_metrics.grounded_answer_rate:.4f}",
        "",
        f"Failures: retrieval={run.failures.retrieval} generation={run.failures.generation} "
        f"evaluation={run.failures.evaluation}",
        "",
        _format_sufficiency(run),
        "",
        "Per case:",
    ]
    for outcome in run.outcomes:
        lines.append(_format_outcome(outcome))
    return "\n".join(lines)


def _format_sufficiency(run: GroundedRun) -> str:
    metrics = run.sufficiency_metrics
    if metrics.assessed_cases == 0:
        return "Sufficiency: not assessed (no policy enabled — this is the E2603 baseline path)"
    return (
        f"Sufficiency: assessed={metrics.assessed_cases} accuracy={metrics.accuracy:.4f} "
        f"false_abstentions={metrics.false_abstentions} missed_abstentions={metrics.missed_abstentions}"
    )


def _format_outcome(outcome: GroundedCaseOutcome) -> str:
    if outcome.failure is not StageFailure.NONE:
        return f"  {outcome.case_id} ({outcome.category.value}): {outcome.failure.value.upper()}"
    grounded = "n/a" if outcome.is_grounded is None else str(outcome.is_grounded)
    flag = " [FORBIDDEN HIT]" if outcome.forbidden_hit else ""
    suff = (
        ""
        if outcome.sufficiency_decision is None
        else f" sufficiency={outcome.sufficiency_decision}"
    )
    return (
        f"  {outcome.case_id} ({outcome.category.value}): "
        f"recall@k={outcome.recall_at_k:.2f} grounded={grounded} "
        f"abstain(expected={outcome.abstention_expected},observed={outcome.abstention_observed})"
        f"{suff}{flag}"
    )
