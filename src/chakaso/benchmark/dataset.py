"""A benchmark dataset: an identified, versioned collection of cases.

A dataset is the unit a result is reported against, so it carries the four identities the
project's reproducibility rule needs kept distinct (ADR-0013): dataset identity, dataset
version, and — per case — case identity and case version. The dataset `version` is an
author-changed string; a `content_fingerprint` is also computed from the cases so an
accidental edit to any case is visible without the author having remembered to bump the
version.

Case identifiers are unique within a dataset. That is not a tidiness rule: a per-case
metric is reported against an identifier, and two cases sharing one make the number
ambiguous about which case it describes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chakaso.benchmark.cases import BenchmarkCase
from chakaso.benchmark.errors import BenchmarkError, DuplicateCaseError, UnknownCaseError
from chakaso.benchmark.identity import CaseId, DatasetId, content_version

__all__ = ["BenchmarkDataset"]


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """A versioned, identified set of benchmark cases."""

    dataset_id: DatasetId
    version: str
    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            message = f"dataset {self.dataset_id} must declare a non-empty version"
            raise BenchmarkError(message)

        seen: set[CaseId] = set()
        for case in self.cases:
            if case.case_id in seen:
                message = f"dataset {self.dataset_id} has a duplicate case id {case.case_id}"
                raise DuplicateCaseError(message)
            seen.add(case.case_id)

    @property
    def content_fingerprint(self) -> str:
        """A fingerprint of every case's identity and version, sorted by identifier.

        Independent of case order in the collection, so reordering the file does not
        change it; sensitive to any case's judgements, so an edit to one does.
        """
        summary = [
            {"case_id": str(case.case_id), "version": case.version}
            for case in sorted(self.cases, key=lambda case: str(case.case_id))
        ]
        return content_version(summary)

    @property
    def case_ids(self) -> tuple[CaseId, ...]:
        """The identifiers of every case, in dataset order."""
        return tuple(case.case_id for case in self.cases)

    def case(self, case_id: CaseId | str) -> BenchmarkCase:
        """Return the case named by ``case_id``.

        Raises:
            UnknownCaseError: the dataset holds no case with that identifier.
        """
        wanted = case_id if isinstance(case_id, CaseId) else CaseId(case_id)
        for case in self.cases:
            if case.case_id == wanted:
                return case
        message = f"dataset {self.dataset_id} has no case {wanted}"
        raise UnknownCaseError(message)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterable[BenchmarkCase]:
        return iter(self.cases)
