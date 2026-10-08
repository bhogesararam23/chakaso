"""Tests for benchmark loading.

A loader that tolerates slop produces a trustworthy-looking number over a case nobody
wrote, so every malformed shape has a test that deliberately creates it and confirms the
loader refuses by name. The happy path round-trips from JSONL and from a manifest directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chakaso.benchmark import (
    BenchmarkCategory,
    CaseId,
    ExpectedBehavior,
    MalformedCaseEntryError,
    dataset_from_rows,
    load_cases,
    load_dataset_from_directory,
    parse_case,
    parse_jsonl,
)
from chakaso.benchmark.errors import BenchmarkFileError, InvalidCaseError

VALID = {
    "case_id": "c1",
    "category": "direct-retrieval",
    "query": "when is the deadline",
    "gold_chunk_ids": ["chk_0000000000000001"],
}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_minimal_row_parses_with_defaults() -> None:
    case = parse_case(VALID)

    assert case.case_id == CaseId("c1")
    assert case.category is BenchmarkCategory.DIRECT_RETRIEVAL
    assert case.expected_behavior is ExpectedBehavior.ANSWER
    assert case.requires_citation is True


def test_jsonl_round_trips_into_cases() -> None:
    text = "\n".join(
        [
            '{"case_id":"a","category":"multi-source","query":"q1","gold_source_ids":["src_0000000000000000"]}',
            "",
            '{"case_id":"b","category":"distractor","query":"q2"}',
        ],
    )

    cases = load_cases(text)

    assert [c.case_id for c in cases] == [CaseId("a"), CaseId("b")]


def test_parse_jsonl_skips_blank_lines() -> None:
    assert len(parse_jsonl('{"case_id":"a"}\n\n')) == 1


def test_dataset_from_rows_carries_identity_and_version() -> None:
    dataset = dataset_from_rows([VALID], dataset_id="dev", version="1.0.0")

    assert str(dataset.dataset_id) == "dev"
    assert dataset.version == "1.0.0"
    assert len(dataset) == 1


# ---------------------------------------------------------------------------
# Refusals — each malformed shape created deliberately
# ---------------------------------------------------------------------------


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(MalformedCaseEntryError, match="unknown benchmark case field"):
        parse_case({**VALID, "extra": 1})


def test_a_missing_required_field_is_refused() -> None:
    row = {k: v for k, v in VALID.items() if k != "query"}
    with pytest.raises(MalformedCaseEntryError, match="query"):
        parse_case(row)


def test_an_unknown_category_is_refused() -> None:
    with pytest.raises(MalformedCaseEntryError, match="unknown category"):
        parse_case({**VALID, "category": "not-a-real-category"})


def test_an_invalid_identifier_is_refused() -> None:
    with pytest.raises(MalformedCaseEntryError, match="invalid identifier"):
        parse_case({**VALID, "gold_chunk_ids": ["chk_not-hex"]})


def test_a_non_string_list_element_is_refused() -> None:
    with pytest.raises(MalformedCaseEntryError, match="must contain strings"):
        parse_case({**VALID, "gold_chunk_ids": [123]})


def test_a_non_boolean_citation_flag_is_refused() -> None:
    with pytest.raises(MalformedCaseEntryError, match="must be a boolean"):
        parse_case({**VALID, "requires_citation": "yes"})


def test_a_structurally_valid_but_inconsistent_case_is_refused() -> None:
    with pytest.raises(InvalidCaseError, match="both gold and forbidden"):
        parse_case(
            {
                **VALID,
                "forbidden_chunk_ids": ["chk_0000000000000001"],  # same as gold
            }
        )


def test_a_malformed_json_line_names_the_line_number() -> None:
    with pytest.raises(MalformedCaseEntryError, match="line 2"):
        parse_jsonl('{"case_id":"a"}\n{oops not json}\n')


# ---------------------------------------------------------------------------
# Directory loading
# ---------------------------------------------------------------------------


def _write_dataset(directory: Path) -> None:
    (directory / "manifest.toml").write_text(
        'dataset_id = "dev-retrieval"\nversion = "1.0.0"\ndescription = "fixtures"\n',
        encoding="utf-8",
    )
    (directory / "cases.jsonl").write_text(
        '{"case_id":"a","category":"direct-retrieval","query":"q","gold_chunk_ids":["chk_0000000000000001"]}\n'
        '{"case_id":"b","category":"missing-evidence","query":"q2","expected_behavior":"abstain"}\n',
        encoding="utf-8",
    )


def test_a_directory_with_manifest_and_cases_loads(tmp_path: Path) -> None:
    _write_dataset(tmp_path)

    dataset = load_dataset_from_directory(tmp_path)

    assert str(dataset.dataset_id) == "dev-retrieval"
    assert dataset.version == "1.0.0"
    assert dataset.case("b").expected_behavior is ExpectedBehavior.ABSTAIN


def test_a_missing_manifest_is_reported(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkFileError, match="manifest not found"):
        load_dataset_from_directory(tmp_path)


def test_a_manifest_without_required_fields_is_reported(tmp_path: Path) -> None:
    (tmp_path / "manifest.toml").write_text('version = "1"\n', encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(BenchmarkFileError, match="dataset_id"):
        load_dataset_from_directory(tmp_path)
