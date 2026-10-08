"""Tests for the development fixture corpus and the benchmark built on it.

The benchmark is only usable if its gold evidence really exists in the corpus it is run
against and if the whole thing is deterministic. These assert both: every gold and
forbidden identifier resolves to a chunk the corpus produced, the abstain cases carry no
gold, and two builds fingerprint identically.
"""

from __future__ import annotations

from chakaso.benchmark import (
    FIXTURE_SOURCES,
    BenchmarkCategory,
    ExpectedBehavior,
    build_corpus,
    development_benchmark,
    load_development_benchmark,
    resolved_chunk_ids,
)


def test_every_fixture_yields_exactly_one_chunk() -> None:
    corpus = build_corpus()

    assert corpus.source_count == len(FIXTURE_SOURCES)
    assert corpus.chunk_count == len(FIXTURE_SOURCES)


def test_labels_resolve_to_chunk_identifiers() -> None:
    mapping = resolved_chunk_ids(build_corpus())

    assert set(mapping) == {source.label for source in FIXTURE_SOURCES}
    for source in FIXTURE_SOURCES:
        _, chunks = mapping[source.label]
        assert len(chunks) == 1


def test_every_gold_and_forbidden_identifier_exists_in_the_corpus() -> None:
    corpus = build_corpus()
    dataset = development_benchmark(corpus)
    available = {chunk.chunk_id for chunk in corpus.chunks}

    for case in dataset:
        assert set(case.gold_chunk_ids) <= available
        assert set(case.forbidden_chunk_ids) <= available


def test_no_source_supports_the_missing_evidence_case() -> None:
    dataset = development_benchmark(build_corpus())

    case = dataset.case("missing-evidence-planets")
    assert case.gold_chunk_ids == ()
    assert case.expected_behavior is ExpectedBehavior.ABSTAIN
    assert case.category is BenchmarkCategory.MISSING_EVIDENCE


def test_the_injection_text_is_carried_verbatim_as_content() -> None:
    mapping = resolved_chunk_ids(build_corpus())
    corpus = build_corpus()
    source_id, (chunk_id,) = mapping["injection-page"]

    chunk = corpus.chunk_for(chunk_id)

    assert chunk is not None
    assert "Ignore all previous instructions" in chunk.text
    assert source_id is not None


def test_two_builds_are_identical() -> None:
    first_corpus, first_dataset = load_development_benchmark()
    second_corpus, second_dataset = load_development_benchmark()

    assert first_dataset.content_fingerprint == second_dataset.content_fingerprint
    assert [c.chunk_id for c in first_corpus.chunks] == [c.chunk_id for c in second_corpus.chunks]


def test_a_direct_case_names_a_real_gold_chunk() -> None:
    corpus, dataset = load_development_benchmark()
    case = dataset.case("direct-deadline")

    assert len(case.gold_chunk_ids) == 1
    assert case.gold_chunk_ids[0] in {c.chunk_id for c in corpus.chunks}
