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
    MissingProvenanceError,
    UnknownCaseError,
    WorkloadValidationError,
)
from chakaso.benchmark.fixtures import (
    FIXTURE_SOURCES,
    FixtureSource,
    build_corpus,
    development_benchmark,
    load_development_benchmark,
    resolved_chunk_ids,
)
from chakaso.benchmark.grounded_cases import (
    CaseSplit,
    GroundedCase,
    GroundedProvenance,
    GroundedSource,
)
from chakaso.benchmark.grounded_dataset import GroundedDataset
from chakaso.benchmark.grounded_loader import (
    grounded_dataset_from_rows,
    load_grounded_dataset_from_directory,
    parse_grounded_case,
    parse_grounded_source,
)
from chakaso.benchmark.grounded_report import (
    GROUNDED_EVALUATION_NOTE,
    format_grounded_report,
    grounded_report_json,
    grounded_report_to_dict,
)
from chakaso.benchmark.grounded_runner import (
    FailureCounts,
    GeneratedAnswer,
    GroundedAnswerMetrics,
    GroundedCaseOutcome,
    GroundedEvaluationRunner,
    GroundedRetrievalMetrics,
    GroundedRun,
    NonAnsweringProvider,
    ReferenceAnswerProvider,
    StageFailure,
    environment_metadata,
)
from chakaso.benchmark.grounded_workload import (
    GROUNDED_SOURCES,
    SELF_AUTHORED_PROVENANCE,
    case_evidence,
    development_grounded_workload,
    load_development_grounded_workload,
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
    "GROUNDED_EVALUATION_NOTE",
    "GROUNDED_SOURCES",
    "SELF_AUTHORED_PROVENANCE",
    "BenchmarkCase",
    "BenchmarkCategory",
    "BenchmarkDataset",
    "BenchmarkError",
    "BenchmarkFileError",
    "BenchmarkRun",
    "CaseId",
    "CaseOutcome",
    "CaseSplit",
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
    "FailureCounts",
    "FixtureSource",
    "GeneratedAnswer",
    "GroundedAnswerMetrics",
    "GroundedCase",
    "GroundedCaseOutcome",
    "GroundedDataset",
    "GroundedEvaluationRunner",
    "GroundedProvenance",
    "GroundedRetrievalMetrics",
    "GroundedRun",
    "GroundedSource",
    "IncompatibleBenchmarkError",
    "InvalidCaseError",
    "InvalidIdentifierError",
    "MalformedCaseEntryError",
    "MissingProvenanceError",
    "NonAnsweringProvider",
    "ReferenceAnswerProvider",
    "RetrievalBenchmarkRunner",
    "RetrievalMetrics",
    "StageFailure",
    "UnknownCaseError",
    "WorkloadValidationError",
    "build_corpus",
    "canonical_json",
    "case_evidence",
    "content_version",
    "correction_report_json",
    "correction_report_to_dict",
    "dataset_from_rows",
    "development_benchmark",
    "development_correction_benchmark",
    "development_grounded_workload",
    "environment_metadata",
    "format_correction_report",
    "format_grounded_report",
    "format_report",
    "grounded_dataset_from_rows",
    "grounded_report_json",
    "grounded_report_to_dict",
    "load_cases",
    "load_dataset_from_directory",
    "load_development_benchmark",
    "load_development_grounded_workload",
    "load_grounded_dataset_from_directory",
    "parse_case",
    "parse_grounded_case",
    "parse_grounded_source",
    "parse_jsonl",
    "report_json",
    "report_to_dict",
    "resolved_chunk_ids",
]
