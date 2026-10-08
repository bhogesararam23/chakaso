"""Tests for the benchmark case schema.

A case is a judgement, so the tests check that the judgement is what survives: collections
are canonicalized, a broken case (blank query, a chunk that is both gold and forbidden) is
refused, and the version tracks the judgements rather than the name or the input order.
"""

from __future__ import annotations

import pytest

from chakaso.benchmark import (
    BenchmarkCase,
    BenchmarkCategory,
    CaseId,
    ExpectedBehavior,
    InvalidCaseError,
)
from chakaso.core.identifiers import ChunkId, SourceId


def chk(n: int) -> ChunkId:
    return ChunkId(f"chk_{n:016x}")


def src(n: int) -> SourceId:
    return SourceId(f"src_{n:016x}")


def make_case(**overrides: object) -> BenchmarkCase:
    fields: dict[str, object] = {
        "case_id": CaseId("case-1"),
        "category": BenchmarkCategory.DIRECT_RETRIEVAL,
        "query": "when is the deadline",
        "gold_chunk_ids": (chk(1),),
    }
    fields.update(overrides)
    return BenchmarkCase(**fields)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction and canonicalization
# ---------------------------------------------------------------------------


def test_query_is_trimmed() -> None:
    assert make_case(query="  padded  ").query == "padded"


def test_collections_are_deduplicated_and_ordered() -> None:
    case = make_case(gold_chunk_ids=(chk(3), chk(1), chk(3), chk(2)))

    assert case.gold_chunk_ids == (chk(1), chk(2), chk(3))


def test_membership_helpers_expose_the_sets() -> None:
    case = make_case(gold_chunk_ids=(chk(1),), forbidden_chunk_ids=(chk(9),))

    assert chk(1) in case.relevant_chunks
    assert chk(9) in case.unacceptable_chunks
    assert chk(5) not in case.relevant_chunks


# ---------------------------------------------------------------------------
# Refusals — each one deliberately broken to prove the validator bites
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_a_blank_query_is_refused(blank: str) -> None:
    with pytest.raises(InvalidCaseError, match="blank query"):
        make_case(query=blank)


def test_a_chunk_cannot_be_both_gold_and_forbidden() -> None:
    with pytest.raises(InvalidCaseError, match="both gold and forbidden"):
        make_case(gold_chunk_ids=(chk(1),), forbidden_chunk_ids=(chk(1),))


def test_a_blank_claim_is_refused() -> None:
    with pytest.raises(InvalidCaseError, match="blank required or forbidden claim"):
        make_case(required_claims=("real claim", "   "))


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def test_version_ignores_collection_input_order() -> None:
    a = make_case(gold_chunk_ids=(chk(1), chk(2)))
    b = make_case(gold_chunk_ids=(chk(2), chk(1)))

    assert a.version == b.version


def test_version_changes_when_a_judgement_changes() -> None:
    before = make_case(gold_chunk_ids=(chk(1),))
    after = make_case(gold_chunk_ids=(chk(1), chk(2)))

    assert before.version != after.version


def test_renaming_a_case_keeps_its_version() -> None:
    # The identifier is the address, not part of the judgement.
    first = make_case(case_id=CaseId("old-name"))
    second = make_case(case_id=CaseId("new-name"))

    assert first.version == second.version


def test_equal_cases_compare_equal_and_hash_equal() -> None:
    first = make_case()
    second = make_case()

    assert first == second
    assert len({first, second}) == 1


def test_expected_behavior_defaults_to_answer_and_is_typed() -> None:
    assert make_case().expected_behavior is ExpectedBehavior.ANSWER
    assert make_case(expected_behavior=ExpectedBehavior.ABSTAIN).expected_behavior is (
        ExpectedBehavior.ABSTAIN
    )
