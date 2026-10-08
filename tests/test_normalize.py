"""Tests for deterministic content normalization.

The property that matters is that equivalent content, reached a different way, becomes
the same text — because normalization feeds the content hash and the chunk identifiers,
so two formatting variants of one document must produce one set of identifiers, not two.
"""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime

from chakaso.core.identifiers import derive_source_id
from chakaso.evidence import SourceRecord
from chakaso.retrieval import chunk_document, normalize_document

RETRIEVED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_carriage_returns_become_line_feeds() -> None:
    assert normalize_document("a\r\nb\rc\nd") == "a\nb\nc\nd"


def test_trailing_spaces_and_tabs_are_removed() -> None:
    assert normalize_document("a   \nb\t\t") == "a\nb"


def test_leading_indentation_is_preserved() -> None:
    # Lists and code blocks mean something by their indentation.
    assert normalize_document("    indented") == "    indented"


def test_runs_of_blank_lines_collapse_to_one() -> None:
    assert normalize_document("one\n\n\n\n\n\ntwo") == "one\n\ntwo"


def test_blank_lines_at_the_edges_are_dropped() -> None:
    # Surrounding blank lines go, but the leading indentation of a content line does
    # not — that one is preserved on purpose (test_leading_indentation_is_preserved).
    assert normalize_document("\n\nhello\n\n\n") == "hello"


# ---------------------------------------------------------------------------
# Unicode
# ---------------------------------------------------------------------------


def test_decomposed_text_is_composed_to_nfc() -> None:
    decomposed = "e\u0301"  # 'e' + combining acute
    assert normalize_document(decomposed) == "\u00e9"


def test_nfc_input_is_left_unchanged() -> None:
    text = "café naïve"
    assert normalize_document(text) == text
    assert normalize_document(unicodedata.normalize("NFD", text)) == text


# ---------------------------------------------------------------------------
# Non-destruction
# ---------------------------------------------------------------------------


def test_normalization_does_not_change_case_or_wording() -> None:
    text = "The Deadline Is REAL for Section 4."
    assert normalize_document(text) == text


def test_normalization_does_not_rewrap_lines() -> None:
    text = "short line\nnext line"
    assert normalize_document(text) == text


def test_empty_and_whitespace_only_text_normalize_to_empty() -> None:
    assert normalize_document("") == ""
    assert normalize_document("   \n\t\n \n") == ""


# ---------------------------------------------------------------------------
# Determinism and idempotence
# ---------------------------------------------------------------------------


def test_normalization_is_idempotent() -> None:
    text = "  a\r\n\r\n\r\nb  \n"
    once = normalize_document(text)
    assert normalize_document(once) == once


def test_equivalent_representations_produce_identical_chunk_identifiers() -> None:
    # The reason normalization exists ahead of chunking: a Windows file and a paste of
    # the same body must not become two different sets of evidence.
    pasted = "# Title\n\nFirst paragraph.\n\nSecond paragraph.\n"
    with_crlf = pasted.replace("\n", "\r\n")

    source_id = derive_source_id("text:", normalize_document(pasted))
    pasted_chunks = chunk_document(source_id, normalize_document(pasted))
    crlf_chunks = chunk_document(source_id, normalize_document(with_crlf))

    assert [chunk.chunk_id for chunk in pasted_chunks] == [chunk.chunk_id for chunk in crlf_chunks]


def test_a_normalized_document_is_ingested_through_a_record() -> None:
    raw = "Alpha.\r\n\r\n\r\n\r\nBeta.  \n"
    record = SourceRecord.create_text(
        content=normalize_document(raw), retrieved_at=RETRIEVED, label="doc"
    )

    assert record.canonical_reference == "text:doc"
    assert normalize_document(raw) == "Alpha.\n\nBeta."
