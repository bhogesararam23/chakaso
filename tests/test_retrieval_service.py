"""Tests for retrieval orchestration: query to a valid evidence pack.

The contract under test is the one the evidence layer depends on: a pack produced from a
retrieval is valid, preserves provenance and scores, keeps its sources distinct, and —
when nothing matches — comes back empty rather than fabricated. The retriever boundary is
also exercised with a stand-in, because being replaceable is the point of it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.evidence import EvidenceChunk, resolve_citations
from chakaso.retrieval import (
    ChunkingConfig,
    Corpus,
    RetrievalError,
    RetrievalResult,
    RetrievalService,
    RetrievalStrategy,
    ingest_text,
)

RETRIEVED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
ONE_CHUNK_PER_PARAGRAPH = ChunkingConfig(target_characters=1, max_characters=2000)


def doc(text: str, label: str):
    return ingest_text(text, label=label, config=ONE_CHUNK_PER_PARAGRAPH, now=lambda: RETRIEVED)


def service(*documents) -> RetrievalService:
    return RetrievalService(Corpus.of(documents))


# ---------------------------------------------------------------------------
# A successful retrieval produces a valid, scored pack
# ---------------------------------------------------------------------------


def test_a_match_builds_a_pack_from_the_corpus() -> None:
    outcome = service(doc("The deadline is in April.", "policy")).search("deadline", top_k=5)

    assert outcome.pack.is_empty is False
    assert len(outcome.pack.chunks) == 1
    chunk = outcome.pack.chunks[0]
    assert chunk.retrieval_score is not None and chunk.retrieval_score > 0
    assert chunk.chunk_id in {c.chunk_id for c in outcome.pack.chunks}


def test_provenance_survives_retrieval() -> None:
    document = doc("Appeals go to a panel.", "appeals")
    outcome = service(document).search("appeals", top_k=5)

    chunk = outcome.pack.chunks[0]
    source = outcome.pack.source_for(chunk.source_id)
    assert source is not None
    assert source.source_id == document.source.source_id
    assert source.canonical_reference == document.source.canonical_reference


def test_the_pack_is_valid_by_construction() -> None:
    # The pack validates every chunk's source is present; a retrieved pack must satisfy
    # that rather than rely on the retriever behaving.
    outcome = service(doc("one paragraph", "a")).search("paragraph", top_k=5)
    for chunk in outcome.pack.chunks:
        assert outcome.pack.source_for(chunk.source_id) is not None


def test_results_keep_the_ranking_explanation() -> None:
    outcome = service(
        doc("deadline deadline deadline", "heavy"),
        doc("deadline appeal", "light"),
    ).search("deadline", top_k=5)

    assert [r.rank for r in outcome.results] == list(range(1, len(outcome.results) + 1))
    assert outcome.results[0].score >= outcome.results[1].score
    assert "deadline" in outcome.results[0].matched_terms


def test_citation_resolution_works_against_a_retrieved_pack() -> None:
    # The end the pipeline exists for: a chunk retrieved into a pack resolves as a real
    # citation, and an identifier outside it does not.
    outcome = service(doc("The submission window closes.", "window")).search("submission", top_k=5)
    chunk = outcome.pack.chunks[0]

    resolution = resolve_citations(
        f"See [{chunk.chunk_id}] and [src_ffffffffffffffff].", outcome.pack
    )

    assert [c.identifier for c in resolution.citations] == [str(chunk.chunk_id)]
    assert resolution.unknown_identifiers == ("src_ffffffffffffffff",)


# ---------------------------------------------------------------------------
# Duplicate sources
# ---------------------------------------------------------------------------


def test_two_chunks_of_one_source_yield_one_source_record() -> None:
    document = doc("deadline one.\n\ndeadline two.", "multi")
    outcome = service(document).search("deadline", top_k=5)

    assert len(outcome.pack.chunks) == 2
    assert len(outcome.pack.sources) == 1


# ---------------------------------------------------------------------------
# Empty retrieval is honest
# ---------------------------------------------------------------------------


def test_a_query_matching_nothing_yields_an_empty_pack_not_invented_evidence() -> None:
    outcome = service(doc("only words here", "a")).search("elephant", top_k=5)

    assert outcome.results == ()
    assert outcome.pack.is_empty is True
    assert outcome.pack.supplied_identifiers == frozenset()


def test_retrieving_from_an_empty_corpus_is_empty() -> None:
    outcome = RetrievalService(Corpus()).search("anything", top_k=5)

    assert outcome.pack.is_empty is True


# ---------------------------------------------------------------------------
# top_k and source filtering
# ---------------------------------------------------------------------------


def test_top_k_bounds_the_pack() -> None:
    documents = [doc("deadline alpha", f"a{i}") for i in range(5)]
    outcome = service(*documents).search("deadline", top_k=2)

    assert len(outcome.pack.chunks) == 2


def test_source_filter_restricts_the_pack() -> None:
    first = doc("deadline here", "first")
    second = doc("deadline there", "second")
    outcome = service(first, second).search(
        "deadline", top_k=5, source_ids=[first.source.source_id]
    )

    assert {chunk.source_id for chunk in outcome.pack.chunks} == {first.source.source_id}


def test_top_k_below_one_is_rejected() -> None:
    with pytest.raises(RetrievalError, match="at least 1"):
        service(doc("a word", "a")).search("word", top_k=0)


# ---------------------------------------------------------------------------
# The retriever boundary is replaceable
# ---------------------------------------------------------------------------


class _StubRetriever:
    """A stand-in Retriever that always returns the first chunk, for boundary tests."""

    def __init__(self, chunk: EvidenceChunk) -> None:
        self._chunk = chunk

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        source_ids=None,
    ) -> tuple[RetrievalResult, ...]:
        if top_k < 1:
            return ()
        return (
            RetrievalResult(
                chunk=self._chunk,
                score=99.0,
                rank=1,
                strategy=RetrievalStrategy.LEXICAL,
                matched_terms=("stub",),
            ),
        )


def test_a_custom_retriever_can_be_injected() -> None:
    document = doc("irrelevant content", "x")
    stub = _StubRetriever(document.chunks[0])
    outcome = RetrievalService(Corpus.of([document]), retriever=stub, retriever_name="stub").search(
        "anything", top_k=3
    )

    assert outcome.pack.chunks[0].retrieval_score == 99.0
    assert outcome.retrieval_config["retriever"] == "stub"


def test_the_pack_records_retrieval_provenance() -> None:
    outcome = service(doc("deadline", "a")).search("deadline", top_k=7)

    assert outcome.retrieval_config == {"retriever": "lexical", "top_k": "7"}
    assert outcome.pack.retrieval_config["retriever"] == "lexical"


def test_chunk_identifiers_are_not_changed_by_scoring() -> None:
    document = doc("deadline", "a")
    original_id: ChunkId = document.chunks[0].chunk_id
    outcome = service(document).search("deadline", top_k=5)

    # Recording a score makes a copy; the identifier a citation refers to is unchanged.
    assert outcome.pack.chunks[0].chunk_id == original_id


def test_a_source_the_corpus_lacks_is_caught() -> None:
    # A chunk referencing an absent source is a broken corpus, not something to render.
    orphan = EvidenceChunk.create(source_id=SourceId("src_0000000000000000"), text="x", position=0)
    corpus = Corpus()
    corpus.add(doc("seed", "seed").source)

    class _OrphanRetriever:
        def retrieve(self, query, *, top_k, source_ids=None):
            return (
                RetrievalResult(
                    chunk=orphan, score=1.0, rank=1, strategy=RetrievalStrategy.LEXICAL
                ),
            )

    with pytest.raises(RetrievalError, match="no source"):
        RetrievalService(corpus, retriever=_OrphanRetriever()).search("x", top_k=1)
