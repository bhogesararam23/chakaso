"""Tests for the curated self-authored grounded-answer workload."""

from __future__ import annotations

from chakaso.benchmark.cases import BenchmarkCategory, ExpectedBehavior
from chakaso.benchmark.grounded_cases import CaseSplit
from chakaso.benchmark.grounded_dataset import resolved_chunk_ids
from chakaso.benchmark.grounded_workload import (
    GROUNDED_SOURCES,
    SELF_AUTHORED_PROVENANCE,
    case_evidence,
    development_grounded_workload,
    load_development_grounded_workload,
)

# The exact fingerprint of the shipped workload. It is pinned so an unintended edit to any case or
# passage is a failing test rather than a silently changed benchmark (the same discipline the
# retrieval fixtures use). A deliberate change must update this value in the same commit.
PINNED_FINGERPRINT = "d2d2ee00d117b88a"


def test_workload_builds_with_both_splits() -> None:
    dataset = development_grounded_workload()
    assert len(dataset.cases_in(CaseSplit.DEVELOPMENT)) >= 3
    assert len(dataset.cases_in(CaseSplit.HELDOUT)) >= 2


def test_categories_cover_the_distinct_grounding_situations() -> None:
    dataset = development_grounded_workload()
    present = {case.category for case in dataset.cases}
    for required in (
        BenchmarkCategory.DIRECT_RETRIEVAL,
        BenchmarkCategory.MULTI_SOURCE,
        BenchmarkCategory.MISSING_EVIDENCE,
        BenchmarkCategory.CONFLICTING,
        BenchmarkCategory.AMBIGUOUS,
        BenchmarkCategory.UNSUPPORTED_PREMISE,
        BenchmarkCategory.CITATION_CHALLENGE,
    ):
        assert required in present, f"workload is missing the {required.value} situation"


def test_every_passage_is_self_authored_apache_2_0() -> None:
    for source in GROUNDED_SOURCES:
        assert source.provenance is SELF_AUTHORED_PROVENANCE
        assert "Apache-2.0" in source.provenance.licence
        assert "no restriction" in source.provenance.redistribution.lower()


def test_no_abstain_case_carries_required_facts() -> None:
    for case in development_grounded_workload().cases:
        if case.expected_behavior is ExpectedBehavior.ABSTAIN:
            assert case.gold_labels == ()
            assert case.required_facts == ()


def test_splits_share_no_grounding_passage() -> None:
    dataset = development_grounded_workload()
    dev = {label for c in dataset.cases_in(CaseSplit.DEVELOPMENT) for label in c.gold_labels}
    held = {label for c in dataset.cases_in(CaseSplit.HELDOUT) for label in c.gold_labels}
    assert dev & held == set()


def test_every_passage_resolves_to_a_single_chunk() -> None:
    dataset = development_grounded_workload()
    corpus, ds = load_development_grounded_workload()
    resolved = resolved_chunk_ids(ds, corpus)
    assert set(resolved) == dataset.source_labels
    for _, chunks in resolved.values():
        assert len(chunks) == 1


def test_required_facts_are_literally_supported_by_gold_evidence() -> None:
    """A required fact that the gold passage does not contain would make the case unanswerable.

    This is the workload's central self-consistency check: the evidence a case calls acceptable
    must actually state the facts a correct answer is required to carry.
    """
    dataset = development_grounded_workload()
    text_by_label = {source.label: source.text for source in dataset.sources}
    for case in dataset.cases:
        gold_text = " ".join(text_by_label[label] for label in case.gold_labels)
        for fact in case.required_facts:
            assert fact.lower() in gold_text.lower(), (
                f"case {case.case_id} requires fact {fact!r} absent from its gold evidence"
            )


def test_prohibited_claims_are_not_present_in_gold_evidence() -> None:
    dataset = development_grounded_workload()
    text_by_label = {source.label: source.text for source in dataset.sources}
    for case in dataset.cases:
        gold_text = " ".join(text_by_label[label] for label in case.gold_labels)
        for claim in case.prohibited_claims:
            assert claim.lower() not in gold_text.lower(), (
                f"case {case.case_id} prohibits a claim its gold evidence states"
            )


def test_case_evidence_resolves_labels_to_chunks() -> None:
    corpus, dataset = load_development_grounded_workload()
    resolved = resolved_chunk_ids(dataset, corpus)
    evidence = case_evidence(dataset, resolved)
    direct = evidence["ga-direct-loan"]
    assert len(direct[0]) == 1  # one gold chunk
    assert direct[2] == direct[0]  # required evidence equals the single gold chunk
    abstain = evidence["ga-missing-dinosaur"]
    assert abstain[0] == ()  # an abstention names no gold


def test_content_fingerprint_is_pinned() -> None:
    assert development_grounded_workload().content_fingerprint == PINNED_FINGERPRINT
