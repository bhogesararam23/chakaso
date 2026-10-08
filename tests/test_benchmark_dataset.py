"""Tests for the benchmark dataset container.

The dataset is the address a result is reported against, so identity, uniqueness,
versioning and lookup all have to be provable: two cases cannot share an identifier, a
content fingerprint must be order-independent but judgement-sensitive, and asking for a
case that is not there must fail rather than return a neighbour.
"""

from __future__ import annotations

import pytest

from chakaso.benchmark import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkDataset,
    CaseId,
    DatasetId,
    DuplicateCaseError,
    UnknownCaseError,
)
from chakaso.benchmark.errors import BenchmarkError
from chakaso.core.identifiers import ChunkId


def case(name: str, *, gold: tuple[int, ...] = (1,)) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=CaseId(name),
        category=BenchmarkCategory.DIRECT_RETRIEVAL,
        query=f"query for {name}",
        gold_chunk_ids=tuple(ChunkId(f"chk_{n:016x}") for n in gold),
    )


def dataset(*cases: BenchmarkCase) -> BenchmarkDataset:
    return BenchmarkDataset(
        dataset_id=DatasetId("dev-corpus"),
        version="1.0.0",
        cases=cases or (case("a"), case("b")),
    )


def test_dataset_reports_its_cases_in_order() -> None:
    d = dataset(case("a"), case("b"), case("c"))

    assert len(d) == 3
    assert d.case_ids == (CaseId("a"), CaseId("b"), CaseId("c"))
    assert [c.case_id for c in d] == [CaseId("a"), CaseId("b"), CaseId("c")]


def test_lookup_by_identifier_accepts_both_the_type_and_its_string() -> None:
    d = dataset(case("wanted"))

    assert d.case(CaseId("wanted")).query == "query for wanted"
    assert d.case("wanted").case_id == CaseId("wanted")


def test_unknown_case_is_refused() -> None:
    with pytest.raises(UnknownCaseError, match="no case"):
        dataset(case("here")).case("absent")


def test_duplicate_case_identifiers_are_refused() -> None:
    with pytest.raises(DuplicateCaseError, match="duplicate case id"):
        dataset(case("same"), case("same"))


@pytest.mark.parametrize("version", ["", "   "])
def test_a_blank_dataset_version_is_refused(version: str) -> None:
    with pytest.raises(BenchmarkError, match="non-empty version"):
        BenchmarkDataset(dataset_id=DatasetId("d"), version=version, cases=(case("a"),))


def test_content_fingerprint_is_order_independent() -> None:
    forward = dataset(case("a"), case("b"))
    reordered = BenchmarkDataset(
        dataset_id=DatasetId("dev-corpus"),
        version="1.0.0",
        cases=(case("b"), case("a")),
    )

    assert forward.content_fingerprint == reordered.content_fingerprint


def test_content_fingerprint_changes_when_a_case_judgement_changes() -> None:
    before = dataset(case("a", gold=(1,)))
    after = dataset(case("a", gold=(1, 2)))

    assert before.content_fingerprint != after.content_fingerprint
