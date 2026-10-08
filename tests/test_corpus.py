"""Tests for the in-memory corpus.

The corpus is a membership container, so the assertions are about membership and order:
duplicate additions do not grow it, a chunk cannot hide behind a source that does not own
it, and enumeration does not depend on the order things were added.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.core.identifiers import SourceId
from chakaso.evidence import SourceRecord
from chakaso.retrieval import ChunkingConfig, Corpus, CorpusError, ingest_text

RETRIEVED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
# A one-character target keeps every paragraph its own chunk, so the counts asserted
# below are predictable rather than dependent on the default packer merging them.
ONE_CHUNK_PER_PARAGRAPH = ChunkingConfig(target_characters=1, max_characters=2000)


def doc(text: str, label: str):
    return ingest_text(text, label=label, config=ONE_CHUNK_PER_PARAGRAPH, now=lambda: RETRIEVED)


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


def test_adding_documents_populates_sources_and_chunks() -> None:
    corpus = Corpus.of([doc("Alpha.\n\nBeta.", "a"), doc("Gamma.", "g")])

    assert corpus.source_count == 2
    assert corpus.chunk_count == len(corpus.chunks) == 3
    assert corpus.is_empty is False


def test_an_empty_corpus_is_empty() -> None:
    corpus = Corpus()

    assert corpus.is_empty is True
    assert len(corpus) == 0
    assert corpus.chunks == ()
    assert corpus.sources == ()


def test_a_source_with_no_chunks_can_be_added() -> None:
    record = SourceRecord.create_text(content="", retrieved_at=RETRIEVED, label="empty")
    corpus = Corpus()
    corpus.add(record)

    assert corpus.source_count == 1
    assert corpus.chunk_count == 0
    assert corpus.source_for(record.source_id) == record


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------


def test_re_adding_the_same_document_is_idempotent() -> None:
    document = doc("One body.\n\nTwo paragraphs.", "one")
    corpus = Corpus.of([document, document])

    assert corpus.source_count == 1
    assert corpus.chunk_count == 2


def test_identical_content_under_a_different_label_is_two_sources() -> None:
    # A label changes the reference, so the identifier differs and they stay distinct.
    corpus = Corpus.of([doc("Same body.", "left"), doc("Same body.", "right")])

    assert corpus.source_count == 2


# ---------------------------------------------------------------------------
# Provenance guard
# ---------------------------------------------------------------------------


def test_a_chunk_cannot_be_added_under_a_source_that_does_not_own_it() -> None:
    document = doc("Orphan chunk.", "owner")
    stranger = SourceRecord.create_text(content="other", retrieved_at=RETRIEVED, label="other")

    corpus = Corpus()
    with pytest.raises(CorpusError, match="not to the source"):
        corpus.add(stranger, document.chunks)


# ---------------------------------------------------------------------------
# Deterministic enumeration
# ---------------------------------------------------------------------------


def test_enumeration_does_not_depend_on_insertion_order() -> None:
    first = doc("Aardvark note.", "a")
    second = doc("Zebra note.", "z")

    forward = Corpus.of([first, second])
    backward = Corpus.of([second, first])

    assert [chunk.chunk_id for chunk in forward.chunks] == [
        chunk.chunk_id for chunk in backward.chunks
    ]
    assert [source.source_id for source in forward.sources] == [
        source.source_id for source in backward.sources
    ]


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


def test_lookup_by_identifier_returns_the_record_or_none() -> None:
    document = doc("Findable content.", "find")
    corpus = Corpus.of([document])

    assert corpus.source_for(document.source.source_id) == document.source
    assert corpus.chunk_for(document.chunks[0].chunk_id) == document.chunks[0]
    assert corpus.source_for(SourceId("src_0000000000000000")) is None


def test_chunks_for_a_source_are_in_document_order() -> None:
    document = doc("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.", "ordered")
    corpus = Corpus.of([document])

    chunks = corpus.chunks_for(document.source.source_id)

    assert [chunk.position for chunk in chunks] == list(range(len(chunks)))
