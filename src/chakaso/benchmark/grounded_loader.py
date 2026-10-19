"""Loading a grounded-answer workload from data, and validating it before anything runs.

The curated workload ships in code, but a workload is meant to be *data a person writes and
reviews* — a manifest naming the dataset and version, a JSONL file of source passages, a JSONL file
of cases. This module reads that shape into validated `GroundedDataset` objects and refuses anything
that is not exactly a valid workload: an unknown field, a missing provenance, an evidence label that
names no passage, an unrecognized category, split or behaviour, or a case whose judgements leak
across the development/held-out boundary.

The strictness is the whole point (ADR-0031). A workload that tolerates a blank licence, or resolves
a typo'd evidence label to nothing, produces a number computed over a case nobody meant to write and
a source the project may not even be permitted to ship. Every refusal names the record and the field
at fault. This mirrors the retrieval benchmark loader, and reuses its JSONL parser.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.errors import BenchmarkFileError, MalformedCaseEntryError
from chakaso.benchmark.grounded_cases import (
    CaseSplit,
    GroundedCase,
    GroundedProvenance,
    GroundedSource,
)
from chakaso.benchmark.grounded_dataset import GroundedDataset
from chakaso.benchmark.identity import CaseId, DatasetId
from chakaso.benchmark.loader import parse_jsonl

__all__ = [
    "grounded_dataset_from_rows",
    "load_grounded_dataset_from_directory",
    "parse_grounded_case",
    "parse_grounded_source",
]

_SOURCE_FIELDS: Final[frozenset[str]] = frozenset({"label", "title", "text", "provenance"})
_SOURCE_REQUIRED: Final[frozenset[str]] = frozenset({"label", "title", "text", "provenance"})
_PROVENANCE_FIELDS: Final[frozenset[str]] = frozenset(
    {"origin", "licence", "collection_method", "redistribution", "created_on"}
)
_PROVENANCE_REQUIRED: Final[frozenset[str]] = frozenset(
    {"origin", "licence", "collection_method", "redistribution"}
)
_CASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "case_id",
        "category",
        "split",
        "question",
        "rationale",
        "expected_behavior",
        "gold_labels",
        "forbidden_labels",
        "required_evidence_labels",
        "required_facts",
        "prohibited_claims",
    }
)
_CASE_REQUIRED: Final[frozenset[str]] = frozenset(
    {"case_id", "category", "split", "question", "rationale"}
)


def _parse_provenance(raw: object, label: str) -> GroundedProvenance:
    if not isinstance(raw, Mapping):
        message = f"source {label!r} field 'provenance' must be an object"
        raise MalformedCaseEntryError(message)
    unknown = sorted(set(raw) - _PROVENANCE_FIELDS)
    if unknown:
        message = f"source {label!r} provenance has unknown field(s) {unknown}"
        raise MalformedCaseEntryError(message)
    missing = sorted(_PROVENANCE_REQUIRED - set(raw))
    if missing:
        message = f"source {label!r} provenance is missing required field(s) {missing}"
        raise MalformedCaseEntryError(message)
    created_on = raw.get("created_on", "")
    if not isinstance(created_on, str):
        message = f"source {label!r} provenance field 'created_on' must be a string"
        raise MalformedCaseEntryError(message)
    return GroundedProvenance(
        origin=_string(raw, "origin", f"source {label}"),
        licence=_string(raw, "licence", f"source {label}"),
        collection_method=_string(raw, "collection_method", f"source {label}"),
        redistribution=_string(raw, "redistribution", f"source {label}"),
        created_on=created_on,
    )


def parse_grounded_source(raw: Mapping[str, object]) -> GroundedSource:
    """Build a validated `GroundedSource` from a serialized mapping, refusing gaps."""
    _reject_unknown(raw, _SOURCE_FIELDS, "source")
    label = _string(raw, "label", "source")
    _require_present(raw, _SOURCE_REQUIRED, f"source {label!r}")
    return GroundedSource(
        label=label,
        title=_string(raw, "title", f"source {label}"),
        text=_string(raw, "text", f"source {label}"),
        provenance=_parse_provenance(raw.get("provenance"), label),
    )


def parse_grounded_case(raw: Mapping[str, object]) -> GroundedCase:
    """Build a validated `GroundedCase` from a serialized mapping, refusing gaps."""
    _reject_unknown(raw, _CASE_FIELDS, "case")
    case_id = CaseId(_string(raw, "case_id", "case"))
    ref = f"case {case_id}"
    _require_present(raw, _CASE_REQUIRED, ref)
    return GroundedCase(
        case_id=case_id,
        category=_category(_string(raw, "category", ref), ref),
        split=_split(_string(raw, "split", ref), ref),
        question=_string(raw, "question", ref),
        rationale=_string(raw, "rationale", ref),
        expected_behavior=_behavior(raw.get("expected_behavior", "answer"), ref),
        gold_labels=_str_tuple(raw.get("gold_labels", []), "gold_labels", ref),
        forbidden_labels=_str_tuple(raw.get("forbidden_labels", []), "forbidden_labels", ref),
        required_evidence_labels=_str_tuple(
            raw.get("required_evidence_labels", []), "required_evidence_labels", ref
        ),
        required_facts=_str_tuple(raw.get("required_facts", []), "required_facts", ref),
        prohibited_claims=_str_tuple(raw.get("prohibited_claims", []), "prohibited_claims", ref),
    )


def grounded_dataset_from_rows(
    source_rows: Iterable[Mapping[str, object]],
    case_rows: Iterable[Mapping[str, object]],
    *,
    dataset_id: str,
    version: str,
) -> GroundedDataset:
    """Build a validated dataset from serialized source and case rows."""
    return GroundedDataset(
        dataset_id=DatasetId(dataset_id),
        version=version,
        sources=tuple(parse_grounded_source(row) for row in source_rows),
        cases=tuple(parse_grounded_case(row) for row in case_rows),
    )


def load_grounded_dataset_from_directory(directory: str | Path) -> GroundedDataset:
    """Load ``manifest.toml``, ``sources.jsonl`` and ``cases.jsonl`` from a workload directory.

    Raises:
        BenchmarkFileError: a file is missing, unreadable, or the manifest lacks a required field.
    """
    root = Path(directory)
    manifest = _read_manifest(root / "manifest.toml")
    dataset_id = manifest.get("dataset_id")
    version = manifest.get("version")
    if not isinstance(dataset_id, str) or not isinstance(version, str):
        message = (
            f"{root / 'manifest.toml'}: manifest must define string 'dataset_id' and 'version'"
        )
        raise BenchmarkFileError(message)

    source_rows = parse_jsonl(_read_text(root / "sources.jsonl"))
    case_rows = parse_jsonl(_read_text(root / "cases.jsonl"))
    return grounded_dataset_from_rows(
        source_rows, case_rows, dataset_id=dataset_id, version=version
    )


def _read_manifest(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        message = f"workload manifest not found: {path}"
        raise BenchmarkFileError(message)
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        message = f"workload manifest is not readable TOML: {path}: {exc}"
        raise BenchmarkFileError(message) from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"workload file could not be read: {path}: {exc}"
        raise BenchmarkFileError(message) from exc


def _reject_unknown(raw: Mapping[str, object], allowed: frozenset[str], kind: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        message = f"{kind} record has unknown field(s) {unknown}; allowed: {sorted(allowed)}"
        raise MalformedCaseEntryError(message)


def _require_present(raw: Mapping[str, object], required: frozenset[str], ref: str) -> None:
    missing = sorted(required - set(raw))
    if missing:
        message = f"{ref} is missing required field(s) {missing}"
        raise MalformedCaseEntryError(message)


def _string(raw: Mapping[str, object], key: str, ref: str) -> str:
    if key not in raw:
        message = f"{ref} is missing required field {key!r}"
        raise MalformedCaseEntryError(message)
    value = raw[key]
    if not isinstance(value, str):
        message = f"{ref} field {key!r} must be a string, got {type(value).__name__}"
        raise MalformedCaseEntryError(message)
    return value


def _str_tuple(value: object, field: str, ref: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        message = f"{ref} field {field!r} must be a list, got {type(value).__name__}"
        raise MalformedCaseEntryError(message)
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            message = f"{ref} field {field!r} must contain strings, got {type(item).__name__}"
            raise MalformedCaseEntryError(message)
        out.append(item)
    return tuple(out)


def _category(value: str, ref: str) -> BenchmarkCategory:
    try:
        return BenchmarkCategory(value)
    except ValueError as exc:
        message = f"{ref} has unknown category {value!r}"
        raise MalformedCaseEntryError(message) from exc


def _split(value: str, ref: str) -> CaseSplit:
    try:
        return CaseSplit(value)
    except ValueError as exc:
        message = f"{ref} has unknown split {value!r}"
        raise MalformedCaseEntryError(message) from exc


def _behavior(value: object, ref: str) -> ExpectedBehavior:
    if not isinstance(value, str):
        message = f"{ref} field 'expected_behavior' must be a string"
        raise MalformedCaseEntryError(message)
    try:
        return ExpectedBehavior(value)
    except ValueError as exc:
        message = f"{ref} has unknown expected_behavior {value!r}"
        raise MalformedCaseEntryError(message) from exc
