"""Tests for the semantic-evaluation development dataset and its deterministic scorer.

These assert the *instrument*, never a capability. The dataset is fixed, versioned and labelled;
the scorer computes accuracy, per-class precision/recall, a confusion matrix and unresolved/error
counts; and the honesty guarantee is that the fixture — which is handed the expected relations —
scores perfectly and that is documented as measuring the harness, not semantic understanding. The
probes exist to prove the scorer can tell an over-claiming judge from an abstaining one.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from chakaso.claims.model import Claim
from chakaso.evaluation import (
    SEMANTIC_DATASET_VERSION,
    SemanticBenchmarkRun,
    dataset_fingerprint,
    development_semantic_dataset,
    format_semantic_report,
    score_semantic_judge,
    semantic_report_json,
)
from chakaso.evaluation.semantic_benchmark import (
    ProbeSupportAllJudge,
    ProbeUncertainJudge,
    materialize_case,
)
from chakaso.evidence import EvidencePack
from chakaso.grounding import (
    FixtureSemanticJudge,
    SemanticEvaluation,
    SemanticJudgeError,
    SemanticJudgement,
)

_CASES = development_semantic_dataset()


def _fixture_judge() -> FixtureSemanticJudge:
    return FixtureSemanticJudge(
        {materialize_case(case)[0].claim_id: case.expected for case in _CASES}
    )


# -- the dataset ---------------------------------------------------------------


def test_dataset_is_versioned_nonempty_with_unique_case_ids() -> None:
    ids = [case.case_id for case in _CASES]

    assert SEMANTIC_DATASET_VERSION
    assert len(_CASES) >= 12
    assert len(set(ids)) == len(ids)
    assert all(case.evidence_texts for case in _CASES)


def test_dataset_covers_the_required_situations() -> None:
    categories = {case.category for case in _CASES}

    for required in (
        "direct_support",
        "direct_contradiction",
        "partial_support",
        "unsupported_claim",
        "irrelevant_evidence",
        "negation",
        "numbers_and_units",
        "entity_mismatch",
        "conflicting_evidence",
        "ambiguous_or_insufficient",
        "multi_sentence_evidence",
        "citation_evidence_mismatch",
    ):
        assert required in categories


def test_fingerprint_is_deterministic_and_sensitive_to_changes() -> None:
    first = dataset_fingerprint(_CASES)
    second = dataset_fingerprint(development_semantic_dataset())

    assert first == second  # same content, same fingerprint
    edited = replace(_CASES[0], claim_text=_CASES[0].claim_text + " (edited)")
    assert dataset_fingerprint((edited, *_CASES[1:])) != first


def test_case_validation_rejects_bad_shape() -> None:
    from chakaso.evaluation.semantic_benchmark import SemanticCase

    with pytest.raises(ValueError, match="at least one evidence"):
        SemanticCase("x", "cat", "claim", (), SemanticJudgement.SUPPORTED)
    with pytest.raises(ValueError, match="out-of-range conflict"):
        SemanticCase("x", "cat", "claim", ("evidence",), SemanticJudgement.SUPPORTED, (5,))


# -- materialisation -----------------------------------------------------------


def test_materialize_case_is_deterministic() -> None:
    case = _CASES[0]

    claim_a, pack_a = materialize_case(case)
    claim_b, pack_b = materialize_case(case)

    assert claim_a.claim_id == claim_b.claim_id
    assert [c.chunk_id for c in pack_a.chunks] == [c.chunk_id for c in pack_b.chunks]
    assert isinstance(claim_a, Claim)
    assert isinstance(pack_a, EvidencePack)


# -- scoring: the fixture validates the harness, nothing more ------------------


def test_the_fixture_scores_perfectly_and_is_only_structure_replay() -> None:
    run = score_semantic_judge(_fixture_judge(), cases=_CASES)

    # The fixture was handed the expected relations, so it is correct by construction; this proves
    # the scorer's arithmetic, not any semantic capability.
    assert run.accuracy == 1.0
    assert run.error_count == 0
    assert all(outcome.correct for outcome in run.outcomes)
    # "Unresolved" counts cases the judge left undecided; a correct fixture reproduces the dataset's
    # own unresolved labels, so the count equals those cases — not zero.
    expected_unresolved = sum(
        1
        for case in _CASES
        if case.expected in (SemanticJudgement.UNCERTAIN, SemanticJudgement.NOT_EVALUATED)
    )
    assert run.unresolved_count == expected_unresolved
    # every confusion cell is diagonal (expected == predicted)
    assert all(key.split("->")[0] == key.split("->")[1] for key in run.confusion)


def test_per_class_metrics_on_a_correct_run_are_all_one() -> None:
    run = score_semantic_judge(_fixture_judge(), cases=_CASES)

    for metrics in run.per_class().values():
        if metrics["support"]:
            assert metrics["precision"] == 1.0
            assert metrics["recall"] == 1.0


def test_support_all_probe_is_scored_as_over_claiming() -> None:
    run = score_semantic_judge(ProbeSupportAllJudge(), cases=_CASES)

    # A judge that claims support for everything is wrong on every non-support case.
    assert run.accuracy < 1.0
    assert all(outcome.predicted is SemanticJudgement.SUPPORTED for outcome in run.outcomes)


def test_uncertain_probe_is_scored_as_fully_unresolved() -> None:
    run = score_semantic_judge(ProbeUncertainJudge(), cases=_CASES)

    assert run.unresolved_count == run.size
    assert run.accuracy < 1.0


def test_an_evaluator_exception_counts_as_a_failure_not_support() -> None:
    class _Raiser:
        name = "raiser"
        version = "0.1"

        def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:
            message = "cannot decide"
            raise SemanticJudgeError(message)

    run = score_semantic_judge(_Raiser(), cases=_CASES)

    assert run.error_count == run.size
    assert all(outcome.predicted is SemanticJudgement.NOT_EVALUATED for outcome in run.outcomes)
    assert all(outcome.evaluator_error for outcome in run.outcomes)


# -- reports -------------------------------------------------------------------


def test_run_is_deterministic_across_calls() -> None:
    first = score_semantic_judge(_fixture_judge(), cases=_CASES)
    second = score_semantic_judge(_fixture_judge(), cases=_CASES)

    assert first.to_dict() == second.to_dict()
    assert format_semantic_report(first) == format_semantic_report(second)


def test_text_report_labels_itself_a_development_instrument() -> None:
    run = score_semantic_judge(_fixture_judge(), cases=_CASES)

    report = format_semantic_report(run)
    assert "development report" in report
    assert "NOT semantic understanding" in report
    assert run.evaluator_name in report


def test_json_report_is_valid_and_carries_the_honesty_note() -> None:
    run = score_semantic_judge(_fixture_judge(), cases=_CASES)

    payload = json.loads(semantic_report_json(run))
    assert payload["benchmark"]["kind"] == "development"
    assert "not" in payload["benchmark"]["note"]
    assert "accuracy" in payload["metrics"]
    assert isinstance(run, SemanticBenchmarkRun)
