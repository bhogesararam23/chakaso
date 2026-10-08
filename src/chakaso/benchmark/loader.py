"""Loading benchmark datasets from data, not from code.

A benchmark is data a person writes and reviews: a JSONL file of cases beside a small
TOML manifest naming the dataset and its version. This module turns that into validated
`BenchmarkDataset` objects, and it refuses anything that is not exactly a valid case —
unknown fields, a bad identifier, a category that does not exist, a JSONL line that is not
an object.

The strictness is the point. A benchmark that silently tolerates a typo'd field produces a
number computed over a case nobody meant to write, and the whole value of a versioned
benchmark (ADR-0013) is that a result can be trusted to mean its label. Every refusal
names the offending case or line and the field at fault.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final, TypeVar

from chakaso.benchmark.cases import BenchmarkCase, BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.dataset import BenchmarkDataset
from chakaso.benchmark.errors import BenchmarkFileError, MalformedCaseEntryError
from chakaso.benchmark.identity import CaseId, DatasetId
from chakaso.core.errors import IdentifierError
from chakaso.core.identifiers import ChunkId, SourceId

__all__ = [
    "dataset_from_rows",
    "load_cases",
    "load_dataset_from_directory",
    "parse_case",
    "parse_jsonl",
]

_CASE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "case_id",
        "category",
        "query",
        "gold_chunk_ids",
        "forbidden_chunk_ids",
        "gold_source_ids",
        "required_claims",
        "forbidden_claims",
        "expected_behavior",
        "requires_citation",
    }
)
_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({"case_id", "category", "query"})


def parse_case(raw: Mapping[str, object]) -> BenchmarkCase:
    """Build a validated `BenchmarkCase` from a serialized mapping.

    Raises:
        MalformedCaseEntryError: a required field is missing, a field is unknown or of the
            wrong shape, or an identifier/category/behaviour value is not recognized.
        InvalidCaseError: the entry parses but the case it describes is inconsistent
            (a blank query, a chunk that is both gold and forbidden).
    """
    unknown = sorted(set(raw) - _CASE_FIELDS)
    if unknown:
        message = f"unknown benchmark case field(s) {unknown}; allowed: {sorted(_CASE_FIELDS)}"
        raise MalformedCaseEntryError(message)

    case_id = CaseId(_require_str(raw, "case_id", ref=None))
    ref = str(case_id)

    missing = sorted(_REQUIRED_FIELDS - set(raw))
    if missing:
        message = f"case {ref} is missing required field(s) {missing}"
        raise MalformedCaseEntryError(message)

    return BenchmarkCase(
        case_id=case_id,
        category=_category(_require_str(raw, "category", ref), ref),
        query=_require_str(raw, "query", ref),
        gold_chunk_ids=_id_tuple(raw.get("gold_chunk_ids", []), ChunkId, "gold_chunk_ids", ref),
        forbidden_chunk_ids=_id_tuple(
            raw.get("forbidden_chunk_ids", []), ChunkId, "forbidden_chunk_ids", ref
        ),
        gold_source_ids=_id_tuple(raw.get("gold_source_ids", []), SourceId, "gold_source_ids", ref),
        required_claims=_str_tuple(raw.get("required_claims", []), "required_claims", ref),
        forbidden_claims=_str_tuple(raw.get("forbidden_claims", []), "forbidden_claims", ref),
        expected_behavior=_behavior(raw.get("expected_behavior", "answer"), ref),
        requires_citation=_bool(raw.get("requires_citation", True), "requires_citation", ref),
    )


def parse_jsonl(text: str) -> tuple[Mapping[str, object], ...]:
    """Parse JSONL text into a tuple of object records, skipping blank lines.

    Raises:
        MalformedCaseEntryError: a line is not valid JSON or is not a JSON object, naming
            the line number so a bad file can be fixed without a search.
    """
    records: list[Mapping[str, object]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            loaded = json.loads(stripped)
        except ValueError as exc:
            message = f"case file line {number} is not valid JSON: {exc}"
            raise MalformedCaseEntryError(message) from exc
        if not isinstance(loaded, dict):
            message = f"case file line {number} is not a JSON object"
            raise MalformedCaseEntryError(message)
        records.append(loaded)
    return tuple(records)


def load_cases(text: str) -> tuple[BenchmarkCase, ...]:
    """Parse JSONL text into validated cases."""
    return tuple(parse_case(record) for record in parse_jsonl(text))


def dataset_from_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    dataset_id: str,
    version: str,
) -> BenchmarkDataset:
    """Build a dataset from serialized case rows plus dataset identity and version."""
    return BenchmarkDataset(
        dataset_id=DatasetId(dataset_id),
        version=version,
        cases=tuple(parse_case(row) for row in rows),
    )


def load_dataset_from_directory(directory: str | Path) -> BenchmarkDataset:
    """Load ``manifest.toml`` and ``cases.jsonl`` from a benchmark directory.

    Raises:
        BenchmarkFileError: a file is missing, unreadable, or the manifest lacks a required
            field.
    """
    root = Path(directory)
    manifest_path = root / "manifest.toml"
    cases_path = root / "cases.jsonl"

    manifest = _read_manifest(manifest_path)
    dataset_id = manifest.get("dataset_id")
    version = manifest.get("version")
    if not isinstance(dataset_id, str) or not isinstance(version, str):
        message = f"{manifest_path}: manifest must define string 'dataset_id' and 'version'"
        raise BenchmarkFileError(message)

    try:
        cases_text = cases_path.read_text(encoding="utf-8")
    except OSError as exc:
        message = f"benchmark case file could not be read: {cases_path}: {exc}"
        raise BenchmarkFileError(message) from exc

    return BenchmarkDataset(
        dataset_id=DatasetId(dataset_id),
        version=version,
        cases=load_cases(cases_text),
    )


def _read_manifest(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        message = f"benchmark manifest not found: {path}"
        raise BenchmarkFileError(message)
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        message = f"benchmark manifest is not readable TOML: {path}: {exc}"
        raise BenchmarkFileError(message) from exc


def _require_str(raw: Mapping[str, object], key: str, ref: str | None) -> str:
    if key not in raw:
        message = f"case {ref} is missing required field {key!r}"
        raise MalformedCaseEntryError(message)
    value = raw[key]
    if not isinstance(value, str):
        message = f"case {ref} field {key!r} must be a string, got {type(value).__name__}"
        raise MalformedCaseEntryError(message)
    return value


def _category(value: str, ref: str) -> BenchmarkCategory:
    try:
        return BenchmarkCategory(value)
    except ValueError as exc:
        message = f"case {ref} has unknown category {value!r}"
        raise MalformedCaseEntryError(message) from exc


def _behavior(value: object, ref: str) -> ExpectedBehavior:
    if not isinstance(value, str):
        message = f"case {ref} field 'expected_behavior' must be a string"
        raise MalformedCaseEntryError(message)
    try:
        return ExpectedBehavior(value)
    except ValueError as exc:
        message = f"case {ref} has unknown expected_behavior {value!r}"
        raise MalformedCaseEntryError(message) from exc


_I = TypeVar("_I", ChunkId, SourceId)


def _id_tuple(value: object, identifier: type[_I], field: str, ref: str) -> tuple[_I, ...]:
    items = _str_tuple(value, field, ref)
    ids: list[_I] = []
    for item in items:
        try:
            ids.append(identifier.parse(item))
        except IdentifierError as exc:
            message = f"case {ref} field {field!r} has an invalid identifier {item!r}"
            raise MalformedCaseEntryError(message) from exc
    return tuple(ids)


def _str_tuple(value: object, field: str, ref: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        message = f"case {ref} field {field!r} must be a list, got {type(value).__name__}"
        raise MalformedCaseEntryError(message)
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            message = f"case {ref} field {field!r} must contain strings, got {type(item).__name__}"
            raise MalformedCaseEntryError(message)
        out.append(item)
    return tuple(out)


def _bool(value: object, field: str, ref: str) -> bool:
    if not isinstance(value, bool):
        message = f"case {ref} field {field!r} must be a boolean, got {type(value).__name__}"
        raise MalformedCaseEntryError(message)
    return value
