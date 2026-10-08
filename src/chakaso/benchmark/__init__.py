"""Benchmark: versioned cases, curated fixtures, and a runnable retrieval benchmark.

This package turns the retrieval metrics in `chakaso.evaluation` into something that can be
run against a defined, versioned set of cases. A benchmark is data with an identity and a
version: a case is addressable by a stable name, its content is fingerprinted so a result
records which definition it scored, and a dataset carries the version that ties a number
to a precise set of judgements.

This is a *development* instrument. The first fixtures are small, deliberately synthetic,
and designed to pin retrieval behaviour (distractors, duplicates, contradictions, missing
evidence). Nothing here is a scientific benchmark and no result here should be read as a
claim about general performance.

What exists now is identity, versioning, the case schema, the strict loader, the curated
fixture corpus and the retrieval runner. Reports and the higher-level claim/grounding
correction work build on these and are added as they are built, each with tests, and none
of them described as present before it is.
"""

from __future__ import annotations

from chakaso.benchmark.cases import BenchmarkCase, BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.correction import (
    CorrectionCase,
    CorrectionClaim,
    CorrectionDataset,
    development_correction_benchmark,
)
from chakaso.benchmark.correction_report import (
    DEVELOPMENT_CORRECTION_NOTE,
    correction_report_json,
    correction_report_to_dict,
    format_correction_report,
)
from chakaso.benchmark.correction_runner import (
    CorrectionBenchmarkMetrics,
    CorrectionBenchmarkRun,
    CorrectionBenchmarkRunner,
    CorrectionCaseOutcome,
)
from chakaso.benchmark.dataset import BenchmarkDataset
from chakaso.benchmark.errors import (
    BenchmarkError,
    BenchmarkFileError,
    DuplicateCaseError,
    IncompatibleBenchmarkError,
    InvalidCaseError,
    InvalidIdentifierError,
    MalformedCaseEntryError,
    UnknownCaseError,
)
from chakaso.benchmark.fixtures import (
    FIXTURE_SOURCES,
    FixtureSource,
    build_corpus,
    development_benchmark,
    load_development_benchmark,
    resolved_chunk_ids,
)
from chakaso.benchmark.identity import (
    CaseId,
    DatasetId,
    canonical_json,
    content_version,
)
from chakaso.benchmark.loader import (
    dataset_from_rows,
    load_cases,
    load_dataset_from_directory,
    parse_case,
    parse_jsonl,
)
from chakaso.benchmark.report import (
    DEVELOPMENT_BENCHMARK_NOTE,
    format_report,
    report_json,
    report_to_dict,
)
from chakaso.benchmark.runner import (
    BenchmarkRun,
    CaseOutcome,
    RetrievalBenchmarkRunner,
    RetrievalMetrics,
)

__all__ = [
    "DEVELOPMENT_BENCHMARK_NOTE",
    "DEVELOPMENT_CORRECTION_NOTE",
    "FIXTURE_SOURCES",
    "BenchmarkCase",
    "BenchmarkCategory",
    "BenchmarkDataset",
    "BenchmarkError",
    "BenchmarkFileError",
    "BenchmarkRun",
    "CaseId",
    "CaseOutcome",
    "CorrectionBenchmarkMetrics",
    "CorrectionBenchmarkRun",
    "CorrectionBenchmarkRunner",
    "CorrectionCase",
    "CorrectionCaseOutcome",
    "CorrectionClaim",
    "CorrectionDataset",
    "DatasetId",
    "DuplicateCaseError",
    "ExpectedBehavior",
    "FixtureSource",
    "IncompatibleBenchmarkError",
    "InvalidCaseError",
    "InvalidIdentifierError",
    "MalformedCaseEntryError",
    "RetrievalBenchmarkRunner",
    "RetrievalMetrics",
    "UnknownCaseError",
    "build_corpus",
    "canonical_json",
    "content_version",
    "correction_report_json",
    "correction_report_to_dict",
    "dataset_from_rows",
    "development_benchmark",
    "development_correction_benchmark",
    "format_correction_report",
    "format_report",
    "load_cases",
    "load_dataset_from_directory",
    "load_development_benchmark",
    "parse_case",
    "parse_jsonl",
    "report_json",
    "report_to_dict",
    "resolved_chunk_ids",
]
