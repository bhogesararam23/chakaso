"""Tests for the grounded-answer workload loader: strict parsing and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chakaso.benchmark.errors import (
    BenchmarkFileError,
    MalformedCaseEntryError,
    MissingProvenanceError,
    WorkloadValidationError,
)
from chakaso.benchmark.grounded_loader import (
    grounded_dataset_from_rows,
    load_grounded_dataset_from_directory,
    parse_grounded_case,
    parse_grounded_source,
)
from chakaso.benchmark.identity import DatasetId

PROVENANCE = {
    "origin": "Authored for Chakaso.",
    "licence": "Apache-2.0.",
    "collection_method": "self-authored",
    "redistribution": "no restriction",
}


def _source_row(label: str, text: str) -> dict[str, object]:
    return {"label": label, "title": label.title(), "text": text, "provenance": dict(PROVENANCE)}


def _case_row(case_id: str, **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "case_id": case_id,
        "category": "direct-retrieval",
        "split": "development",
        "question": "how long is the loan",
        "rationale": "single-source fact",
        "gold_labels": ["loan"],
    }
    base.update(overrides)
    return base


class TestParsing:
    def test_valid_source_parses(self) -> None:
        source = parse_grounded_source(_source_row("loan", "A loan lasts 28 days."))
        assert source.label == "loan"
        assert "Apache-2.0" in source.provenance.licence

    def test_valid_case_parses(self) -> None:
        case = parse_grounded_case(_case_row("c-1"))
        assert case.question == "how long is the loan"

    def test_unknown_source_field_refused(self) -> None:
        row = {**_source_row("loan", "text"), "extra": 1}
        with pytest.raises(MalformedCaseEntryError):
            parse_grounded_source(row)

    def test_missing_provenance_object_refused(self) -> None:
        row = {"label": "loan", "title": "Loan", "text": "x"}
        with pytest.raises(MalformedCaseEntryError):
            parse_grounded_source(row)

    def test_blank_licence_refused_as_missing_provenance(self) -> None:
        row = _source_row("loan", "text")
        row["provenance"] = {**PROVENANCE, "licence": "   "}
        with pytest.raises(MissingProvenanceError):
            parse_grounded_source(row)

    def test_unknown_category_refused(self) -> None:
        with pytest.raises(MalformedCaseEntryError):
            parse_grounded_case(_case_row("c-1", category="nonsense"))

    def test_unknown_split_refused(self) -> None:
        with pytest.raises(MalformedCaseEntryError):
            parse_grounded_case(_case_row("c-1", split="train"))


class TestDatasetFromRows:
    def test_builds_valid_dataset(self) -> None:
        dataset = grounded_dataset_from_rows(
            [_source_row("loan", "A loan lasts 28 days.")],
            [_case_row("c-1")],
            dataset_id="ga-x",
            version="1.0.0",
        )
        assert dataset.source_labels == frozenset({"loan"})

    def test_dangling_gold_label_refused(self) -> None:
        with pytest.raises(WorkloadValidationError):
            grounded_dataset_from_rows(
                [_source_row("loan", "text")],
                [_case_row("c-1", gold_labels=["ghost"])],
                dataset_id="ga-x",
                version="1.0.0",
            )

    def test_split_leakage_refused(self) -> None:
        with pytest.raises(WorkloadValidationError):
            grounded_dataset_from_rows(
                [_source_row("loan", "text")],
                [
                    _case_row("c-dev", split="development", gold_labels=["loan"]),
                    _case_row("c-held", split="heldout", gold_labels=["loan"]),
                ],
                dataset_id="ga-x",
                version="1.0.0",
            )


class TestDirectoryLoading:
    def _write(self, root: Path, sources: list[dict], cases: list[dict]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.toml").write_text(
            'dataset_id = "ga-dir"\nversion = "1.0.0"\n', encoding="utf-8"
        )
        (root / "sources.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in sources), encoding="utf-8"
        )
        (root / "cases.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8"
        )

    def test_loads_a_workload_directory(self, tmp_path: Path) -> None:
        self._write(
            tmp_path,
            [_source_row("loan", "A loan lasts 28 days.")],
            [
                _case_row("c-1"),
                _case_row(
                    "c-2", gold_labels=[], category="missing-evidence", expected_behavior="abstain"
                ),
            ],
        )
        dataset = load_grounded_dataset_from_directory(tmp_path)
        assert dataset.dataset_id == DatasetId("ga-dir")
        assert len(dataset.cases) == 2

    def test_missing_manifest_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BenchmarkFileError):
            load_grounded_dataset_from_directory(tmp_path)

    def test_malformed_jsonl_line_refused(self, tmp_path: Path) -> None:
        (tmp_path / "manifest.toml").write_text(
            'dataset_id = "ga-dir"\nversion = "1.0.0"\n', encoding="utf-8"
        )
        (tmp_path / "sources.jsonl").write_text('{"label":\n', encoding="utf-8")
        (tmp_path / "cases.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(MalformedCaseEntryError):
            load_grounded_dataset_from_directory(tmp_path)
