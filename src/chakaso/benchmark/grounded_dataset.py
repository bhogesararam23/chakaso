"""A grounded-answer workload: a provenance-bearing source pool and the cases over it.

A `GroundedDataset` is the unit a grounded answer result is reported against, so it keeps the
same four identities distinct that ADR-0013 fixes for the retrieval benchmark — dataset id,
dataset version, and per case a case id and a case version — and adds a `content_fingerprint`
over the cases *and* the source pool, so editing a passage is caught whether or not the author
remembered to bump a version.

It validates three things a single case cannot see on its own:

* every evidence label a case names is actually a source in the pool — a dangling reference is
  a broken workload, refused at construction, never resolved to nothing at run time;
* a case identifier is unique, so a per-case number is unambiguous;
* the development and held-out splits share no supporting evidence — a source that grounds a
  held-out case may not also ground a development case, or a "held-out" number would be a
  restatement of what was tuned on.

Evidence is named by label here and resolved to real content-derived `ChunkId`s by ingesting the
same passages a runner will search, exactly as the retrieval fixtures do: a gold identifier always
matches the evidence the corpus actually produces, and because text-source identifiers carry no
timestamp the resolution is fully deterministic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from chakaso.benchmark.errors import (
    DuplicateCaseError,
    WorkloadValidationError,
)
from chakaso.benchmark.grounded_cases import CaseSplit, GroundedCase, GroundedSource
from chakaso.benchmark.identity import CaseId, DatasetId, content_version
from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.retrieval import Corpus, ingest_text

__all__ = ["GroundedDataset", "build_corpus", "resolved_chunk_ids"]

# A fixed timestamp keeps identity out of the clock: text-source identifiers derive from content
# only, so which chunk ids the corpus yields never depends on when it is built.
DEFAULT_TIMESTAMP: datetime = datetime(2026, 10, 19, 0, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class GroundedDataset:
    """An identified, versioned, integrity-checked grounded-answer workload."""

    dataset_id: DatasetId
    version: str
    sources: tuple[GroundedSource, ...]
    cases: tuple[GroundedCase, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            message = f"dataset {self.dataset_id} must declare a non-empty version"
            raise WorkloadValidationError(message)

        labels = [source.label for source in self.sources]
        if len(labels) != len(set(labels)):
            message = f"dataset {self.dataset_id} has a duplicate source label"
            raise WorkloadValidationError(message)
        pool = set(labels)

        seen: set[CaseId] = set()
        for case in self.cases:
            if case.case_id in seen:
                message = f"dataset {self.dataset_id} has a duplicate case id {case.case_id}"
                raise DuplicateCaseError(message)
            seen.add(case.case_id)
            referenced = (
                set(case.gold_labels)
                | set(case.forbidden_labels)
                | set(case.required_evidence_labels)
            )
            dangling = referenced - pool
            if dangling:
                names = ", ".join(sorted(dangling))
                message = f"case {case.case_id} names evidence not in the pool: {names}"
                raise WorkloadValidationError(message)

        self._check_split_leakage()

    def _check_split_leakage(self) -> None:
        dev_evidence = {
            label
            for case in self.cases
            if case.split is CaseSplit.DEVELOPMENT
            for label in case.gold_labels
        }
        held_evidence = {
            label
            for case in self.cases
            if case.split is CaseSplit.HELDOUT
            for label in case.gold_labels
        }
        shared = dev_evidence & held_evidence
        if shared:
            names = ", ".join(sorted(shared))
            message = f"sources support cases on both splits, leaking into held-out: {names}"
            raise WorkloadValidationError(message)

    @property
    def source_labels(self) -> frozenset[str]:
        """The labels of every passage in the pool."""
        return frozenset(source.label for source in self.sources)

    def cases_in(self, split: CaseSplit) -> tuple[GroundedCase, ...]:
        """The cases on one side of the split, in dataset order."""
        return tuple(case for case in self.cases if case.split is split)

    @property
    def content_fingerprint(self) -> str:
        """A fingerprint over every case judgement and every source passage.

        Order-independent (both collections are canonicalized by key before hashing) so
        reordering the file does not change it, and sensitive to any edit of a case or a
        passage so an undeclared change is still visible.
        """
        case_summary = [
            {"case_id": str(case.case_id), "version": case.version}
            for case in sorted(self.cases, key=lambda case: str(case.case_id))
        ]
        source_summary = [
            {
                "label": source.label,
                "text": source.text,
                "provenance": source.provenance.canonical_definition(),
            }
            for source in sorted(self.sources, key=lambda source: source.label)
        ]
        return content_version({"cases": case_summary, "sources": source_summary})

    def source(self, label: str) -> GroundedSource:
        """Return the pool passage named by ``label``.

        Raises:
            WorkloadValidationError: no passage in the pool carries that label.
        """
        for candidate in self.sources:
            if candidate.label == label:
                return candidate
        message = f"dataset {self.dataset_id} has no source labelled {label!r}"
        raise WorkloadValidationError(message)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterable[GroundedCase]:
        return iter(self.cases)


def build_corpus(
    sources: Iterable[GroundedSource], *, now: Callable[[], datetime] | None = None
) -> Corpus:
    """Ingest every pool passage into a corpus, keyed by content-derived identifiers."""
    resolved_now = now if now is not None else (lambda: DEFAULT_TIMESTAMP)
    corpus = Corpus()
    for source in sources:
        corpus.add_document(
            ingest_text(source.text, label=source.label, title=source.title, now=resolved_now)
        )
    return corpus


def resolved_chunk_ids(
    dataset: GroundedDataset, corpus: Corpus
) -> dict[str, tuple[SourceId, tuple[ChunkId, ...]]]:
    """Map each pool label to its source identifier and its chunk identifiers.

    Matched by the canonical ``text:<label>`` reference rather than iteration order, so the
    mapping is defined by identity. A label present in the pool but missing from the corpus (or
    the reverse) means the dataset and the corpus disagree — a defect, surfaced here rather than
    producing a silently empty gold set.
    """
    wanted = {f"text:{source.label}": source.label for source in dataset.sources}
    found: dict[str, tuple[SourceId, tuple[ChunkId, ...]]] = {}
    for source in corpus.sources:
        label = wanted.get(source.canonical_reference)
        if label is not None:
            found[label] = (
                source.source_id,
                tuple(chunk.chunk_id for chunk in corpus.chunks_for(source.source_id)),
            )
    missing = sorted(set(wanted.values()) - set(found))
    if missing:
        names = ", ".join(missing)
        message = f"the corpus did not yield these pool labels: {names}"
        raise WorkloadValidationError(message)
    return found
