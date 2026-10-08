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

What exists now is identity and versioning. The case schema, the dataset loader, the
curated fixture corpus, the retrieval runner and the reports are added as they are built,
each with tests, and none of them described as present before it is.
"""

from __future__ import annotations

from chakaso.benchmark.errors import BenchmarkError, InvalidIdentifierError
from chakaso.benchmark.identity import (
    CaseId,
    DatasetId,
    canonical_json,
    content_version,
)

__all__ = [
    "BenchmarkError",
    "CaseId",
    "DatasetId",
    "InvalidIdentifierError",
    "canonical_json",
    "content_version",
]
