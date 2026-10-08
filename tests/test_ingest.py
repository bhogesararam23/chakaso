"""Tests for local ingestion: reading one explicit source into evidence records.

Two families of assertion matter here. The pipeline must produce the records the
evidence layer already defines, deterministically. And the filesystem safety rules must
actually refuse — a validator that never rejects anything it is supposed to reject is
not enforcing anything. Each rejection is exercised deliberately.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from chakaso.evidence import SourceKind
from chakaso.retrieval import (
    ContentDecodingError,
    DocumentTooLargeError,
    DocumentUnavailableError,
    UnsupportedDocumentError,
    ingest_file,
    ingest_text,
)

RETRIEVED = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)

MARKDOWN = """\
# 4. Deadlines

The submission window closes on 14 April.

## 4.1 Extensions

Requests must be made in writing.
"""


def frozen_clock() -> datetime:
    return RETRIEVED


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_ingest_markdown_file_produces_a_source_and_chunks(tmp_path: Path) -> None:
    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")

    result = ingest_file(document, now=frozen_clock)

    assert result.source.kind is SourceKind.FILE
    assert result.source.title == "spec.md"
    assert result.source.metadata["format"] == "markdown"
    assert result.source.raw_text_path == str(document.resolve())
    assert result.chunks
    assert all(chunk.source_id == result.source.source_id for chunk in result.chunks)


def test_sections_are_preserved_through_ingestion(tmp_path: Path) -> None:
    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")

    result = ingest_file(document, now=frozen_clock)

    headings = {chunk.section for chunk in result.chunks}
    assert "4. Deadlines" in headings
    assert "4. Deadlines > 4.1 Extensions" in headings


def test_ingest_plain_text_file(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("Just prose with no headings at all.\n", encoding="utf-8")

    result = ingest_file(document, now=frozen_clock)

    assert result.source.metadata["format"] == "text"
    assert result.chunks[0].section is None
    assert result.chunks[0].text == "Just prose with no headings at all."


def test_title_can_be_supplied(tmp_path: Path) -> None:
    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")

    result = ingest_file(document, title="Grievance Policy", now=frozen_clock)

    assert result.source.title == "Grievance Policy"


def test_a_custom_chunking_config_is_respected(tmp_path: Path) -> None:
    from chakaso.retrieval import ChunkingConfig

    document = tmp_path / "long.md"
    document.write_text("word " * 200, encoding="utf-8")

    fine = ingest_file(
        document,
        config=ChunkingConfig(target_characters=80, max_characters=100),
        now=frozen_clock,
    )
    coarse = ingest_file(document, now=frozen_clock)

    assert len(fine.chunks) > len(coarse.chunks)


# ---------------------------------------------------------------------------
# Empty and content-preserving
# ---------------------------------------------------------------------------


def test_an_empty_file_is_a_valid_source_with_no_chunks(tmp_path: Path) -> None:
    # A file with nothing in it cannot support a claim; it is a real source with zero
    # chunks, not an error and not a fabricated empty chunk.
    document = tmp_path / "empty.md"
    document.write_text("", encoding="utf-8")

    result = ingest_file(document, now=frozen_clock)

    assert result.chunks == ()
    assert result.source.source_id is not None


def test_every_word_of_a_file_survives_into_a_chunk(tmp_path: Path) -> None:
    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")

    joined = " ".join(chunk.text for chunk in ingest_file(document, now=frozen_clock).chunks)

    for phrase in ("closes on 14 April", "made in writing", "submission window"):
        assert phrase in joined


def test_supplied_text_is_ingested_without_a_filesystem() -> None:
    result = ingest_text("# Title\n\nBody sentence.", label="clause", now=frozen_clock)

    assert result.source.kind is SourceKind.TEXT
    assert result.source.canonical_reference == "text:clause"
    assert [chunk.text for chunk in result.chunks] == ["Body sentence."]


# ---------------------------------------------------------------------------
# Normalization happens inside ingestion
# ---------------------------------------------------------------------------


def test_ingestion_normalizes_line_endings_and_trailing_space(tmp_path: Path) -> None:
    document = tmp_path / "crlf.md"
    document.write_bytes(b"# Title\r\n\r\nBody.  \r\n")

    result = ingest_file(document, now=frozen_clock)

    assert result.chunks[0].text == "Body."


def test_crlf_and_lf_text_produce_identical_identifiers() -> None:
    # The point of normalizing during ingestion: two representations of one body are
    # one source with one set of chunks, not two.
    crlf = ingest_text("# T\r\n\r\nBody.", now=frozen_clock)
    lf = ingest_text("# T\n\nBody.", now=frozen_clock)

    assert crlf.source.source_id == lf.source.source_id
    assert [c.chunk_id for c in crlf.chunks] == [c.chunk_id for c in lf.chunks]


def test_a_byte_order_mark_is_not_treated_as_content(tmp_path: Path) -> None:
    document = tmp_path / "bom.txt"
    document.write_bytes(b"\xef\xbb\xbfAlpha.\n")

    result = ingest_file(document, now=frozen_clock)

    assert result.chunks[0].text == "Alpha."


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_ingesting_the_same_file_twice_is_deterministic(tmp_path: Path) -> None:
    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")

    first = ingest_file(document, now=frozen_clock)
    second = ingest_file(document, now=frozen_clock)

    assert first.source.source_id == second.source.source_id
    assert [c.chunk_id for c in first.chunks] == [c.chunk_id for c in second.chunks]


def test_a_dot_dot_path_and_its_resolved_form_are_one_source(tmp_path: Path) -> None:
    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")
    (tmp_path / "sub").mkdir()
    indirection = tmp_path / "sub" / ".." / "spec.md"

    direct = ingest_file(document, now=frozen_clock)
    # ``..`` is normalized away when the reference is built, so a path that winds
    # through a parent cannot fork one document into two sources.
    indirect = ingest_file(indirection, now=frozen_clock)

    assert direct.source.source_id == indirect.source.source_id


def test_a_str_and_a_path_name_the_same_source(tmp_path: Path) -> None:
    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")

    assert (
        ingest_file(str(document), now=frozen_clock).source.source_id
        == ingest_file(document, now=frozen_clock).source.source_id
    )


# ---------------------------------------------------------------------------
# Filesystem safety: each rejection is exercised deliberately
# ---------------------------------------------------------------------------


def test_nonexistent_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DocumentUnavailableError, match="does not exist"):
        ingest_file(tmp_path / "absent.md", now=frozen_clock)


def test_a_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(DocumentUnavailableError, match="not a regular file"):
        ingest_file(tmp_path, now=frozen_clock)


def test_an_empty_path_string_is_refused() -> None:
    with pytest.raises(DocumentUnavailableError, match="non-empty path"):
        ingest_file("   ", now=frozen_clock)


def test_an_unsupported_extension_is_refused(tmp_path: Path) -> None:
    document = tmp_path / "notes.rst"
    document.write_text("ReStructuredText", encoding="utf-8")

    with pytest.raises(UnsupportedDocumentError, match=r"\.rst"):
        ingest_file(document, now=frozen_clock)


def test_the_private_docx_format_is_refused_on_format(tmp_path: Path) -> None:
    # The private planning pack is .docx; ingestion never reads it because the format
    # allowlist admits only plain text and Markdown, not because of any filename check.
    document = tmp_path / "plan.docx"
    document.write_bytes(b"PK\x03\x04 binary-ish")

    with pytest.raises(UnsupportedDocumentError, match=r"\.docx"):
        ingest_file(document, now=frozen_clock)


def test_a_file_over_the_limit_is_refused_without_being_read(tmp_path: Path) -> None:
    document = tmp_path / "big.txt"
    document.write_bytes(b"x" * 1000)

    with pytest.raises(DocumentTooLargeError, match="over the"):
        ingest_file(document, max_bytes=512, now=frozen_clock)


def test_a_file_under_the_limit_is_read(tmp_path: Path) -> None:
    document = tmp_path / "big.txt"
    document.write_bytes(b"Small enough.\n")

    assert ingest_file(document, max_bytes=512, now=frozen_clock).chunks


def test_invalid_utf8_is_refused_rather_than_replaced(tmp_path: Path) -> None:
    document = tmp_path / "bad.txt"
    document.write_bytes(b"caf\xe9 is not valid utf-8")

    with pytest.raises(ContentDecodingError, match="not valid UTF-8"):
        ingest_file(document, now=frozen_clock)


def test_unicode_content_and_a_unicode_path_are_ingested(tmp_path: Path) -> None:
    document = tmp_path / "résumé.md"
    document.write_text("Café na\xefve content.", encoding="utf-8")

    result = ingest_file(document, now=frozen_clock)

    assert result.source.canonical_reference.startswith("file:///")
    assert result.chunks[0].text == "Café na\xefve content."


def test_the_clock_must_be_time_aware(tmp_path: Path) -> None:
    from chakaso.retrieval import IngestionError

    document = tmp_path / "spec.md"
    document.write_text(MARKDOWN, encoding="utf-8")

    with pytest.raises(IngestionError, match="timezone-aware"):
        ingest_file(document, now=lambda: datetime(2026, 10, 8, 12, 0))
