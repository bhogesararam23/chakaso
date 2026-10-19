"""Tests for the grounded-answer dataset: integrity, versioning and evidence resolution."""

from __future__ import annotations

import pytest

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.errors import DuplicateCaseError, WorkloadValidationError
from chakaso.benchmark.grounded_cases import (
    CaseSplit,
    GroundedCase,
    GroundedProvenance,
    GroundedSource,
)
from chakaso.benchmark.grounded_dataset import (
    GroundedDataset,
    build_corpus,
    resolved_chunk_ids,
)
from chakaso.benchmark.identity import CaseId, DatasetId

PROVENANCE = GroundedProvenance(
    origin="Authored for Chakaso.",
    licence="Apache-2.0.",
    collection_method="self-authored",
    redistribution="no restriction.",
)


def _source(label: str, text: str) -> GroundedSource:
    return GroundedSource(label=label, title=label.title(), text=text, provenance=PROVENANCE)


def _pool() -> tuple[GroundedSource, ...]:
    return (
        _source("deadline", "The submission deadline is 14 April."),
        _source("panel", "Appeals are heard by a panel of three."),
        _source("window", "Appeals must be filed within thirty days."),
        _source("unrelated", "The cafeteria serves lunch at noon."),
    )


def _case(
    case_id: str,
    split: CaseSplit,
    *,
    gold: tuple[str, ...] = ("deadline",),
    behavior: ExpectedBehavior = ExpectedBehavior.ANSWER,
) -> GroundedCase:
    return GroundedCase(
        case_id=CaseId(case_id),
        category=BenchmarkCategory.DIRECT_RETRIEVAL,
        split=split,
        question="when is the deadline",
        rationale="checks a direct single-source grounded answer",
        expected_behavior=behavior,
        gold_labels=gold,
    )


def _dataset(
    cases: tuple[GroundedCase, ...], sources: tuple[GroundedSource, ...] | None = None
) -> GroundedDataset:
    return GroundedDataset(
        dataset_id=DatasetId("ga-dev"),
        version="1.0.0",
        sources=sources if sources is not None else _pool(),
        cases=cases,
    )


class TestIntegrity:
    def test_valid_dataset_constructs(self) -> None:
        dataset = _dataset((_case("ga-1", CaseSplit.DEVELOPMENT),))
        assert len(dataset) == 1

    def test_empty_version_refused(self) -> None:
        with pytest.raises(WorkloadValidationError):
            GroundedDataset(
                dataset_id=DatasetId("ga-dev"),
                version="  ",
                sources=_pool(),
                cases=(_case("ga-1", CaseSplit.DEVELOPMENT),),
            )

    def test_duplicate_case_id_refused(self) -> None:
        with pytest.raises(DuplicateCaseError):
            _dataset(
                (
                    _case("ga-same", CaseSplit.DEVELOPMENT),
                    _case("ga-same", CaseSplit.HELDOUT),
                )
            )

    def test_duplicate_source_label_refused(self) -> None:
        pool = (_source("dup", "one sentence."), _source("dup", "another sentence."))
        with pytest.raises(WorkloadValidationError):
            _dataset((_case("ga-1", CaseSplit.DEVELOPMENT),), sources=pool)

    def test_dangling_evidence_label_refused(self) -> None:
        with pytest.raises(WorkloadValidationError):
            _dataset((_case("ga-1", CaseSplit.DEVELOPMENT, gold=("not-in-pool",)),))

    def test_split_leakage_refused(self) -> None:
        # The same gold source supports a development case and a held-out case.
        with pytest.raises(WorkloadValidationError):
            _dataset(
                (
                    _case("ga-dev", CaseSplit.DEVELOPMENT, gold=("deadline",)),
                    _case("ga-held", CaseSplit.HELDOUT, gold=("deadline",)),
                )
            )

    def test_disjoint_splits_are_allowed(self) -> None:
        dataset = _dataset(
            (
                _case("ga-dev", CaseSplit.DEVELOPMENT, gold=("deadline",)),
                _case("ga-held", CaseSplit.HELDOUT, gold=("panel",)),
            )
        )
        assert dataset.cases_in(CaseSplit.DEVELOPMENT) == tuple(
            case for case in dataset.cases if case.split is CaseSplit.DEVELOPMENT
        )


class TestVersioning:
    def test_fingerprint_is_deterministic(self) -> None:
        cases = (_case("ga-1", CaseSplit.DEVELOPMENT),)
        assert _dataset(cases).content_fingerprint == _dataset(cases).content_fingerprint

    def test_fingerprint_ignores_case_order(self) -> None:
        a = _case("ga-a", CaseSplit.DEVELOPMENT, gold=("deadline",))
        b = _case("ga-b", CaseSplit.DEVELOPMENT, gold=("panel",))
        forward = _dataset((a, b))
        backward = _dataset((b, a))
        assert forward.content_fingerprint == backward.content_fingerprint

    def test_editing_a_source_bumps_fingerprint(self) -> None:
        cases = (_case("ga-1", CaseSplit.DEVELOPMENT),)
        original = _dataset(cases)
        edited_pool = (*original.sources[:-1], _source("unrelated", "Totally different text."))
        edited = _dataset(cases, sources=edited_pool)
        assert original.content_fingerprint != edited.content_fingerprint

    def test_editing_a_case_bumps_fingerprint(self) -> None:
        original = _dataset((_case("ga-1", CaseSplit.DEVELOPMENT, gold=("deadline",)),))
        edited = _dataset((_case("ga-1", CaseSplit.DEVELOPMENT, gold=("panel",)),))
        assert original.content_fingerprint != edited.content_fingerprint


class TestEvidenceResolution:
    def test_labels_resolve_to_single_deterministic_chunks(self) -> None:
        dataset = _dataset((_case("ga-1", CaseSplit.DEVELOPMENT),))
        corpus = build_corpus(dataset.sources)
        resolved = resolved_chunk_ids(dataset, corpus)
        # Each authored passage is a single chunk, so a label maps to exactly one chunk id.
        assert set(resolved) == set(dataset.source_labels)
        for _, chunks in resolved.values():
            assert len(chunks) == 1
        # Rebuilding from the same pool reproduces identical identifiers.
        again = resolved_chunk_ids(dataset, build_corpus(dataset.sources))
        assert {k: (str(s), tuple(str(c) for c in v)) for k, (s, v) in resolved.items()} == {
            k: (str(s), tuple(str(c) for c in v)) for k, (s, v) in again.items()
        }

    def test_unknown_source_lookup_refused(self) -> None:
        dataset = _dataset((_case("ga-1", CaseSplit.DEVELOPMENT),))
        with pytest.raises(WorkloadValidationError):
            dataset.source("does-not-exist")
