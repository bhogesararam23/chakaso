"""Tests for the correction benchmark cases and dataset versioning.

Versioning is the reproducibility guarantee: a case's version moves only when its definition does,
the dataset fingerprint is order-insensitive but judgement-sensitive, and the curated set must
actually cover every decision the correction rule can reach. Validation is the integrity guard:
impossible cases (no claims, a required position that does not exist, a duplicate id) fail rather
than become silently meaningless measurements.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from chakaso.benchmark.correction import (
    CorrectionCase,
    CorrectionClaim,
    CorrectionDataset,
    development_correction_benchmark,
)
from chakaso.benchmark.errors import (
    BenchmarkError,
    DuplicateCaseError,
    InvalidCaseError,
    UnknownCaseError,
)
from chakaso.benchmark.identity import CaseId, DatasetId
from chakaso.claims.model import ClaimStatus
from chakaso.correction.decision import CorrectionDecision
from chakaso.grounding.semantic import SemanticJudgement

CORRECT_CLAIM = CorrectionClaim(
    "The budget is 12000 credits.",
    ClaimStatus.SUPPORTED,
    cite_label="budget-2024",
    semantic_relation=SemanticJudgement.CONTRADICTED,
)


def _case(**overrides: object) -> CorrectionCase:
    base: dict[str, object] = {
        "case_id": CaseId("c1"),
        "description": "a scenario",
        "claims": (CORRECT_CLAIM,),
        "reassessment_labels": ("budget-2024",),
        "expected_decision": CorrectionDecision.CORRECT,
    }
    base.update(overrides)
    return CorrectionCase(**base)  # type: ignore[arg-type]


def test_the_curated_dataset_covers_every_reachable_decision() -> None:
    dataset = development_correction_benchmark()

    expected = {case.expected_decision for case in dataset.cases}
    assert expected == {
        CorrectionDecision.RETAIN,
        CorrectionDecision.CORRECT,
        CorrectionDecision.QUALIFY,
        CorrectionDecision.NEEDS_REVIEW,
        CorrectionDecision.ABSTAIN,
    }


def test_case_version_is_stable_and_judgement_sensitive() -> None:
    case = _case()
    assert case.version == _case().version

    softened = replace(case, expected_decision=CorrectionDecision.RETAIN)
    assert softened.version != case.version

    relation_changed = replace(
        case, claims=(replace(CORRECT_CLAIM, semantic_relation=SemanticJudgement.UNCERTAIN),)
    )
    assert relation_changed.version != case.version


def test_dataset_fingerprint_is_order_insensitive_but_edit_sensitive() -> None:
    dataset = development_correction_benchmark()
    reordered = replace(dataset, cases=tuple(reversed(dataset.cases)))
    assert dataset.content_fingerprint == reordered.content_fingerprint

    edited_case = replace(dataset.cases[0], description="changed the wording")
    edited = replace(dataset, cases=(edited_case, *dataset.cases[1:]))
    assert dataset.content_fingerprint != edited.content_fingerprint


def test_case_labels_are_normalised_and_required_positions_are_range_checked() -> None:
    normalised = _case(reassessment_labels=("z", "a", "z"))
    assert normalised.reassessment_labels == ("a", "z")

    with pytest.raises(InvalidCaseError, match="unknown claim positions"):
        _case(required_positions=(5,))


def test_impossible_cases_and_datasets_are_refused() -> None:
    with pytest.raises(InvalidCaseError):
        _case(claims=())
    with pytest.raises(InvalidCaseError):
        _case(description="   ")

    dataset = development_correction_benchmark()
    with pytest.raises(DuplicateCaseError):
        CorrectionDataset(
            dataset_id=DatasetId("x"),
            version="1",
            cases=(dataset.cases[0], dataset.cases[0]),
        )

    with pytest.raises(UnknownCaseError):
        dataset.case(CaseId("not-a-case"))


def test_dataset_rejects_a_blank_version() -> None:
    with pytest.raises(BenchmarkError, match="non-empty version"):
        CorrectionDataset(
            dataset_id=DatasetId("x"), version="  ", cases=development_correction_benchmark().cases
        )
