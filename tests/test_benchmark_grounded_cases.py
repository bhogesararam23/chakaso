"""Tests for the grounded-answer case, source and provenance schema."""

from __future__ import annotations

import pytest

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.errors import InvalidCaseError, MissingProvenanceError
from chakaso.benchmark.grounded_cases import (
    CaseSplit,
    GroundedCase,
    GroundedProvenance,
    GroundedSource,
)
from chakaso.benchmark.identity import CaseId

PROVENANCE = GroundedProvenance(
    origin="Authored for Chakaso.",
    licence="Apache-2.0.",
    collection_method="self-authored",
    redistribution="no restriction.",
)


def _case(**overrides: object) -> GroundedCase:
    base: dict[str, object] = {
        "case_id": CaseId("ga-direct-001"),
        "category": BenchmarkCategory.DIRECT_RETRIEVAL,
        "split": CaseSplit.DEVELOPMENT,
        "question": "what is the appeal deadline",
        "rationale": "checks single-source direct factual retrieval and citation",
        "gold_labels": ("deadlines",),
    }
    base.update(overrides)
    return GroundedCase(**base)  # type: ignore[arg-type]


class TestProvenance:
    def test_requires_every_field_nonempty(self) -> None:
        for field in ("origin", "licence", "collection_method", "redistribution"):
            values = {
                "origin": "o",
                "licence": "l",
                "collection_method": "c",
                "redistribution": "r",
            }
            values[field] = "  "
            with pytest.raises(MissingProvenanceError):
                GroundedProvenance(**values)  # type: ignore[arg-type]

    def test_created_on_is_optional(self) -> None:
        provenance = GroundedProvenance(
            origin="o",
            licence="l",
            collection_method="c",
            redistribution="r",
        )
        assert provenance.created_on == ""

    def test_canonical_definition_is_stable(self) -> None:
        assert PROVENANCE.canonical_definition() == PROVENANCE.canonical_definition()


class TestSource:
    def test_valid_source(self) -> None:
        source = GroundedSource(
            label="deadlines",
            title="Deadlines",
            text="The deadline is 14 April.",
            provenance=PROVENANCE,
        )
        assert source.label == "deadlines"

    def test_rejects_bad_slug_label(self) -> None:
        with pytest.raises(InvalidCaseError):
            GroundedSource(label="Bad Label", title="t", text="body", provenance=PROVENANCE)

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(InvalidCaseError):
            GroundedSource(label="ok", title="t", text="   ", provenance=PROVENANCE)

    def test_rejects_blank_title(self) -> None:
        with pytest.raises(InvalidCaseError):
            GroundedSource(label="ok", title=" ", text="body", provenance=PROVENANCE)


class TestCaseConstruction:
    def test_canonicalizes_and_sorts_collections(self) -> None:
        case = _case(
            gold_labels=("b", "a", "a"),
            forbidden_labels=("z",),
            required_evidence_labels=("a",),
            required_facts=("two", "one"),
            prohibited_claims=("bad",),
        )
        assert case.gold_labels == ("a", "b")
        assert case.required_facts == ("one", "two")

    def test_rejects_blank_question(self) -> None:
        with pytest.raises(InvalidCaseError):
            _case(question="   ")

    def test_requires_rationale(self) -> None:
        with pytest.raises(InvalidCaseError):
            _case(rationale="   ")

    def test_rejects_gold_forbidden_overlap(self) -> None:
        with pytest.raises(InvalidCaseError):
            _case(gold_labels=("shared",), forbidden_labels=("shared",))

    def test_rejects_required_evidence_outside_gold(self) -> None:
        with pytest.raises(InvalidCaseError):
            _case(gold_labels=("a",), required_evidence_labels=("b",))

    def test_rejects_abstain_with_gold(self) -> None:
        with pytest.raises(InvalidCaseError):
            _case(expected_behavior=ExpectedBehavior.ABSTAIN, gold_labels=("a",))

    def test_rejects_answer_without_gold(self) -> None:
        with pytest.raises(InvalidCaseError):
            _case(expected_behavior=ExpectedBehavior.ANSWER, gold_labels=())

    def test_rejects_abstain_with_required_facts(self) -> None:
        with pytest.raises(InvalidCaseError):
            _case(
                expected_behavior=ExpectedBehavior.ABSTAIN,
                gold_labels=(),
                required_facts=("a fact",),
            )

    def test_abstain_case_is_valid(self) -> None:
        case = _case(
            case_id=CaseId("ga-abstain-001"),
            category=BenchmarkCategory.MISSING_EVIDENCE,
            expected_behavior=ExpectedBehavior.ABSTAIN,
            gold_labels=(),
        )
        assert case.answerable is False

    def test_answer_case_is_answerable(self) -> None:
        assert _case().answerable is True


class TestCaseVersioning:
    def test_renaming_keeps_version(self) -> None:
        original = _case(case_id=CaseId("ga-direct-001"))
        renamed = _case(case_id=CaseId("totally-different-name"))
        assert original.version == renamed.version

    def test_editing_a_judgement_changes_version(self) -> None:
        original = _case()
        edited = _case(gold_labels=("other",))
        assert original.version != edited.version

    def test_version_is_order_independent(self) -> None:
        one = _case(gold_labels=("a", "b"))
        other = _case(gold_labels=("b", "a"))
        assert one.version == other.version
