"""Tests for ingesting an external text into a pinned corpus.

The property is provenance, not parsing cleverness: what the record says about a corpus has to be what the
bytes are. So the wrapper rule, the chunking rule and the digest each get a case that would fail if the code
guessed instead of observed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chakaso.core.hashing import sha256_hex
from chakaso.training.errors import CorpusError
from chakaso.training.manifest import read_snapshot
from chakaso.training.source_corpus import (
    DEFAULT_MAX_CHARACTERS_PER_DOCUMENT,
    IngestedSource,
    build_corpus_manifest,
    chunk_paragraphs,
    ingest_project_gutenberg,
    pin_project_gutenberg,
    strip_gutenberg_wrappers,
)

_START = "*** START OF THE PROJECT GUTENBERG EBOOK TEST TEXT ***"
_END = "*** END OF THE PROJECT GUTENBERG EBOOK TEST TEXT ***"

_HEADER = """The Project Gutenberg eBook of Test Text

This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever.

Title: Test Text
Author: A. Author
"""

_BODY = "First paragraph.\n\nSecond paragraph, which is longer than the first and carries the story on.\n\nThird paragraph.\n"


def _text(body: str = _BODY, *, start: str = _START, end: str = _END) -> str:
    return f"{_HEADER}\n{start}\n\n{body}\n{end}\n"


# ------------------------------------------------------------------ wrappers


def test_the_wrappers_are_stripped_by_the_sources_own_markers() -> None:
    header, body = strip_gutenberg_wrappers(_text(), slug="pg0")

    assert "Title: Test Text" in header
    assert body == _BODY
    assert _START not in body and _END not in body


def test_a_file_without_both_markers_is_refused_rather_than_trusted() -> None:
    missing_end = f"{_HEADER}\n{_START}\n\n{_BODY}\n"

    with pytest.raises(CorpusError, match="does not carry both"):
        strip_gutenberg_wrappers(missing_end, slug="pg0")
    with pytest.raises(CorpusError, match="does not carry both"):
        strip_gutenberg_wrappers("just some text\n", slug="pg0")


def test_markers_in_the_wrong_order_are_refused() -> None:
    inverted = f"{_HEADER}\n{_END}\n\n{_BODY}\n{_START}\n"

    with pytest.raises(CorpusError, match="does not carry both"):
        strip_gutenberg_wrappers(inverted, slug="pg0")


# ------------------------------------------------------------------ ingestion


def test_ingestion_reads_the_provenance_the_file_states() -> None:
    source = ingest_project_gutenberg(_text(), slug="pg0", ebook_number=0)

    assert source.title == "Test Text"
    assert source.author == "A. Author"
    assert "for the use of anyone anywhere in the United States" in source.licence_statement
    assert source.url.endswith("/0/pg0.txt")
    assert source.raw_characters == len(_text())


def test_the_digest_recorded_is_the_one_the_caller_supplied() -> None:
    """The byte digest, not a digest of the text after newline translation, is what a download check needs."""
    claimed = sha256_hex(b"raw bytes as they arrived")

    source = ingest_project_gutenberg(_text(), slug="pg0", ebook_number=0, source_digest=claimed)

    assert source.source_digest == claimed


def test_a_source_with_an_empty_body_is_refused() -> None:
    empty = _text("\n   \n")

    with pytest.raises(CorpusError, match="no text"):
        ingest_project_gutenberg(empty, slug="pg0", ebook_number=0)


def test_a_slug_that_cannot_be_a_path_is_refused() -> None:
    with pytest.raises(CorpusError, match="path prefix"):
        IngestedSource(
            slug="nested/pg0",
            ebook_number=0,
            title="t",
            author="a",
            licence_statement="l",
            source_digest="0" * 64,
            raw_characters=4,
            text="body",
        )


# ------------------------------------------------------------------ chunking


def test_chunks_respect_the_bound_and_never_split_a_paragraph() -> None:
    long_paragraph = "word " * 500
    text = f"A short paragraph.\n\n{long_paragraph.strip()}\n\nAnother short one.\n"

    chunks = chunk_paragraphs(text, max_characters=64)

    assert chunks[0] == "A short paragraph."
    assert chunks[1] == long_paragraph.strip()  # oversized, and kept whole rather than cut
    assert chunks[2] == "Another short one."
    assert long_paragraph.strip() in chunks[1]


def test_chunking_groups_paragraphs_up_to_the_bound_in_reading_order() -> None:
    paragraphs = [f"Paragraph number {index}." for index in range(1, 8)]
    text = "\n\n".join(paragraphs) + "\n"

    chunks = chunk_paragraphs(text, max_characters=60)

    assert "\n\n".join(chunks).split("\n\n") == paragraphs
    assert all(len(chunk) <= 60 for chunk in chunks[:-1])


def test_chunking_is_deterministic_and_refuses_degenerate_input() -> None:
    text = _BODY

    assert chunk_paragraphs(text, max_characters=16384) == chunk_paragraphs(
        text, max_characters=16384
    )
    with pytest.raises(CorpusError, match="no paragraphs"):
        chunk_paragraphs("   \n\n  \n", max_characters=10)
    with pytest.raises(CorpusError, match="must be positive"):
        chunk_paragraphs("a\n", max_characters=0)


# ------------------------------------------------------------------ manifest and pinning


def test_the_manifest_names_the_sources_it_came_from() -> None:
    first = ingest_project_gutenberg(_text(), slug="pg1", ebook_number=1)
    second = ingest_project_gutenberg(_text(), slug="pg2", ebook_number=2)

    manifest = build_corpus_manifest(
        [first, second],
        name="test-corpus",
        version="1.0.0",
        licence="Public domain",
        source="two synthetic texts",
        retrieved="2026-10-21",
        max_characters_per_document=16384,
    )

    assert manifest.source.endswith("(retrieved 2026-10-21)")
    assert manifest.include == ("pg1/", "pg2/")
    assert [document.path for document in manifest.documents] == [
        "pg1/part-0001.md",
        "pg2/part-0001.md",
    ]
    assert (
        manifest.fingerprint
        == build_corpus_manifest(
            [first, second],
            name="test-corpus",
            version="1.0.0",
            licence="Public domain",
            source="two synthetic texts",
            retrieved="2026-10-21",
            max_characters_per_document=16384,
        ).fingerprint
    )


def test_two_sources_sharing_a_slug_are_refused() -> None:
    first = ingest_project_gutenberg(_text(), slug="pg1", ebook_number=1)
    same = ingest_project_gutenberg(_text(), slug="pg1", ebook_number=1)

    with pytest.raises(CorpusError, match="share a slug"):
        build_corpus_manifest(
            [first, same],
            name="test-corpus",
            version="1.0.0",
            licence="Public domain",
            source="two files, one name",
            retrieved="2026-10-21",
        )


def _write_sources(tmp_path: Path) -> dict[str, tuple[int, Path]]:
    directory = tmp_path / "downloads"
    directory.mkdir()
    files: dict[str, tuple[int, Path]] = {}
    for slug, number in (("pg1", 1), ("pg2", 2)):
        path = directory / f"{slug}.txt"
        path.write_bytes(_text(f"Body of {slug}.\n\nMore of {slug}.\n").encode("utf-8"))
        files[slug] = (number, path)
    return files


def test_pinning_writes_a_snapshot_that_loads_back_verified(tmp_path: Path) -> None:
    files = _write_sources(tmp_path)
    snapshot = tmp_path / "data" / "corpus-snapshot.json"

    manifest, checksum = pin_project_gutenberg(
        files,
        name="test-corpus",
        version="1.0.0",
        licence="Public domain",
        source="two synthetic texts",
        retrieved="2026-10-21",
        snapshot=snapshot,
    )

    loaded = read_snapshot(snapshot)
    assert loaded.checksum == checksum
    assert loaded.fingerprint == manifest.fingerprint
    assert loaded.acquisition["method"] == "chakaso-project-gutenberg"
    assert loaded.acquisition["retrieved"] == "2026-10-21"
    assert loaded.acquisition["source_digests"].startswith("pg1=")
    assert len(loaded.manifest.documents) == 2
    for _number, path in files.values():
        assert sha256_hex(path.read_bytes()) in loaded.acquisition["source_digests"]


def test_pinning_refuses_to_replace_a_recorded_snapshot(tmp_path: Path) -> None:
    files = _write_sources(tmp_path)
    snapshot = tmp_path / "snapshot.json"
    pin_project_gutenberg(
        files,
        name="test-corpus",
        version="1.0.0",
        licence="Public domain",
        source="two synthetic texts",
        retrieved="2026-10-21",
        snapshot=snapshot,
    )

    with pytest.raises(CorpusError, match="not replaced unless"):
        pin_project_gutenberg(
            files,
            name="test-corpus",
            version="1.0.0",
            licence="Public domain",
            source="two synthetic texts",
            retrieved="2026-10-21",
            snapshot=snapshot,
        )


def test_a_file_that_is_not_utf8_is_reported_as_a_read_failure(tmp_path: Path) -> None:
    path = tmp_path / "binary.txt"
    path.write_bytes(b"\xff\xfe not a text file at all")

    with pytest.raises(UnicodeDecodeError):
        pin_project_gutenberg(
            {"pg1": (1, path)},
            name="test-corpus",
            version="1.0.0",
            licence="Public domain",
            source="one undecodable file",
            retrieved="2026-10-21",
            snapshot=tmp_path / "snapshot.json",
        )


def test_the_default_bound_is_chapter_sized_and_recorded_in_the_snapshot(tmp_path: Path) -> None:
    files = _write_sources(tmp_path)

    pin_project_gutenberg(
        files,
        name="test-corpus",
        version="1.0.0",
        licence="Public domain",
        source="two synthetic texts",
        retrieved="2026-10-21",
        snapshot=tmp_path / "s.json",
    )

    acquisition = read_snapshot(tmp_path / "s.json").acquisition

    assert acquisition["characters_per_document_bound"] == str(DEFAULT_MAX_CHARACTERS_PER_DOCUMENT)
