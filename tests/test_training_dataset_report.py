"""Tests for corpus composition and duplication — the measurements a data card is allowed to state."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chakaso.core.hashing import sha256_hex
from chakaso.training.dataset_catalog import CatalogSource, DatasetCatalog, gutenberg_url
from chakaso.training.dataset_report import (
    DEFAULT_SHINGLE_WORDS,
    DatasetReport,
    bottom_k,
    build_report,
    candidate_pairs,
    near_duplicate_groups,
    render_data_card,
    report_to_file,
    word_shingles,
    write_data_card,
)
from chakaso.training.errors import CorpusError
from chakaso.training.manifest import CorpusDocument, CorpusManifest, normalize_text

_ALICE = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to "
    "do: once or twice she had peeped into the book her sister was reading, but it had no pictures or "
    "conversations in it, and what is the use of a book, thought Alice, without pictures or conversations? "
    "So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy "
    "and stupid, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and "
    "picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her."
)
_MELVILLE = (
    "Call me Ishmael. Some years ago, never minding how long I was to be, there came a period of detached "
    "and casual browsing in a quiet town beside the old harbour, where the shipping news filled every column "
    "of the local paper and the sea wind rattled the windows of the lodging house where I had put up for the "
    "night, and I found myself wishing for a wide open stretch of water and a long voyage to pass it."
)
# A near duplicate of the first text: the opening is shared word for word, then it diverges.
# The same passage with an added tail: most of the opening is shared, so the pair is a near duplicate by
# the rule stated in the module, and the tail keeps the two documents from being byte-identical.
_ALICE_COPY = _ALICE + (
    " Then the rabbit took a watch out of its waistcoat-pocket, and, starting up again and hurrying on, "
    "Alice got to her feet and followed it across the field, thinking as she ran that she had never seen "
    "anything so very remarkable before in all her life."
)
_SHORT = "Very short indeed."


def _document(path: str, text: str) -> CorpusDocument:
    normalized = normalize_text(text)
    return CorpusDocument(
        path=path,
        sha256=sha256_hex(normalized.encode("utf-8")),
        byte_count=len(normalized.encode("utf-8")),
        char_count=len(normalized),
        text=normalized,
    )


def _manifest(documents: list[CorpusDocument]) -> CorpusManifest:
    return CorpusManifest(
        name="measured-corpus",
        version="1.0.0",
        licence="Public domain in the United States",
        source="Project Gutenberg (retrieved 2026-01-05)",
        include=("pg1/", "pg2/"),
        extensions=(".md",),
        exact_paths=(),
        max_document_bytes=65_536,
        normalization="crlf-and-cr-to-lf",
        documents=tuple(documents),
    )


def _catalog() -> DatasetCatalog:
    return DatasetCatalog(
        name="measured-corpus",
        version="1.0.0",
        licence="Public domain in the United States",
        source="Project Gutenberg",
        retrieved="2026-01-05",
        sources=(
            CatalogSource(
                slug="pg1",
                ebook_number=1,
                url=gutenberg_url(1),
                sha256="a" * 64,
                title="Alice",
                author="Lewis Carroll",
            ),
            CatalogSource(
                slug="pg2",
                ebook_number=2,
                url=gutenberg_url(2),
                sha256="b" * 64,
                title="Moby Dick",
                author="Herman Melville",
            ),
        ),
        max_characters_per_document=16_384,
    )


def test_a_shingle_set_is_every_consecutive_word_run_of_the_declared_size() -> None:
    shingles = word_shingles("one two three four five", size=3)

    assert shingles == frozenset({"one two three", "two three four", "three four five"})


def test_a_document_shorter_than_the_shingle_size_produces_no_shingles() -> None:
    assert word_shingles("only three words", size=8) == frozenset()


def test_an_unusable_shingle_size_is_refused() -> None:
    with pytest.raises(CorpusError, match="shingle size"):
        word_shingles("any text", size=0)


def test_a_sketch_is_the_smallest_values_of_a_stable_hash_and_not_of_process_hashing() -> None:
    shingles = word_shingles(
        " ".join(f"word{index}" for index in range(40)), size=DEFAULT_SHINGLE_WORDS
    )

    values = bottom_k(shingles, width=4)

    assert len(values) == 4
    assert values == bottom_k(shingles, width=4)
    assert values < bottom_k(shingles, width=64)


def test_a_sketch_of_nothing_is_empty_rather_than_an_arbitrary_number() -> None:
    assert bottom_k(set(), width=8) == frozenset()


def test_a_reprinted_document_is_found_as_a_near_duplicate() -> None:
    groups = near_duplicate_groups(
        {"pg1/part-0001.md": _ALICE, "pg2/part-0007.md": _ALICE_COPY, "pg3/part-0001.md": _MELVILLE}
    )

    assert [group for group, _ in groups] == [("pg1/part-0001.md", "pg2/part-0007.md")]
    assert groups[0][1] > 0.5


def test_unrelated_documents_are_not_reported_as_duplicates() -> None:
    groups = near_duplicate_groups({"a/x.md": _ALICE, "b/y.md": _MELVILLE})

    assert groups == ()


def test_a_fragment_too_short_to_measure_is_never_a_candidate() -> None:
    assert near_duplicate_groups({"a/x.md": _ALICE, "b/y.md": _SHORT}) == ()


def test_transitively_matching_documents_are_reported_as_one_group() -> None:
    middle = _ALICE[:420] + _MELVILLE[:120]
    groups = near_duplicate_groups(
        {"a.md": _ALICE, "b.md": middle, "c.md": _ALICE_COPY},
        similarity=0.3,
    )

    assert len(groups) == 1
    assert set(groups[0][0]) == {"a.md", "b.md", "c.md"}


def test_an_empty_document_set_has_nothing_to_compare() -> None:
    assert near_duplicate_groups({}) == ()
    assert candidate_pairs({}) == 0


def test_a_candidate_filter_below_one_shared_value_is_refused() -> None:
    with pytest.raises(CorpusError, match="min_shared"):
        near_duplicate_groups({"a.md": _ALICE, "b.md": _MELVILLE}, min_shared=0)


def test_a_bucket_cap_below_two_documents_is_refused() -> None:
    with pytest.raises(CorpusError, match="max_bucket"):
        near_duplicate_groups({"a.md": _ALICE, "b.md": _MELVILLE}, max_bucket=1)


def test_a_threshold_that_is_not_a_fraction_is_refused() -> None:
    with pytest.raises(CorpusError, match="similarity must be"):
        near_duplicate_groups({"a.md": _ALICE}, similarity=0.0)


def test_a_report_counts_documents_characters_and_sources() -> None:
    manifest = _manifest(
        [
            _document("pg1/part-0001.md", _ALICE),
            _document("pg1/part-0002.md", _ALICE_COPY),
            _document("pg2/part-0001.md", _MELVILLE),
        ]
    )

    report = build_report(manifest)

    assert report.name == "measured-corpus"
    assert report.documents == 3
    assert report.characters == sum(item.char_count for item in manifest.documents)
    assert dict(report.documents_by_source) == {"pg1": 2, "pg2": 1}
    assert report.characters_by_source[0][1] > 0
    assert (
        report.document_characters["min"]
        <= report.document_characters["p50"]
        <= (report.document_characters["p100"])
    )
    assert report.distinct_symbols > 20


def test_a_report_carries_the_duplication_findings_it_measured() -> None:
    manifest = _manifest(
        [_document("pg1/part-0001.md", _ALICE), _document("pg2/part-0001.md", _ALICE_COPY)]
    )

    report = build_report(manifest)

    assert len(report.near_duplicate_groups) == 1
    assert report.pairs_examined >= 1
    assert report.shingle_words == DEFAULT_SHINGLE_WORDS


def test_exact_duplicate_paths_are_reported_from_the_manifest() -> None:
    one = _document("pg1/part-0001.md", _ALICE)
    manifest = _manifest([one, _document("pg1/part-0002.md", _ALICE)])

    report = build_report(manifest)

    assert report.exact_duplicate_groups == (("pg1/part-0001.md", "pg1/part-0002.md"),)


def test_a_report_serialises_deterministically() -> None:
    manifest = _manifest(
        [_document("pg1/part-0001.md", _ALICE), _document("pg2/part-0001.md", _MELVILLE)]
    )

    first = json.dumps(build_report(manifest).as_dict(), sort_keys=True)
    second = json.dumps(build_report(manifest).as_dict(), sort_keys=True)

    assert first == second


def test_a_data_card_states_provenance_composition_and_limits() -> None:
    manifest = _manifest(
        [_document("pg1/part-0001.md", _ALICE), _document("pg2/part-0001.md", _MELVILLE)]
    )
    report = build_report(manifest)

    card = render_data_card(
        report,
        catalog=_catalog(),
        snapshot_checksum="c" * 32,
        catalog_sha256="d" * 64,
        sources=[("pg1", "Alice", gutenberg_url(1), "a" * 64)],
        prepared_with="chakaso corpus prepare --catalog configs/corpora/x.toml",
        limitations=["Nineteenth-century and classical prose only."],
    )

    assert "# Data card: measured-corpus 1.0.0" in card
    assert "Public domain in the United States" in card
    assert "- **Retrieved:** 2026-01-05" in card
    assert "`pg1` | Alice" in card
    assert "Near-duplicate screening" in card
    assert "chakaso corpus prepare" in card
    assert "Nineteenth-century and classical prose only." in card


def test_a_card_without_a_catalogue_still_describes_the_corpus() -> None:
    report = build_report(_manifest([_document("pg1/part-0001.md", _ALICE)]))

    card = render_data_card(report)

    assert "Retrieved" not in card
    assert "Nothing was measured about the texts beyond what is recorded above." in card


def test_a_data_card_is_not_written_over_unasked(tmp_path: Path) -> None:
    report = build_report(_manifest([_document("pg1/part-0001.md", _ALICE)]))
    card = render_data_card(report)
    destination = tmp_path / "DATA_CARD.md"
    write_data_card(card, destination)

    with pytest.raises(CorpusError, match="allow_overwrite"):
        write_data_card(card, destination)

    write_data_card(card, destination, allow_overwrite=True)
    assert destination.read_text(encoding="utf-8") == card


def test_a_machine_readable_report_lands_beside_the_card(tmp_path: Path) -> None:
    report = build_report(_manifest([_document("pg1/part-0001.md", _ALICE)]))
    destination = tmp_path / "report.json"
    report_to_file(report, destination)

    loaded = json.loads(destination.read_text(encoding="utf-8"))

    assert loaded["documents"] == 1
    assert loaded["document_characters"]["p50"] == report.document_characters["p50"]
    with pytest.raises(CorpusError, match="already exists"):
        report_to_file(report, destination)


def test_a_report_holds_its_own_data_rather_than_reading_a_global() -> None:
    manifest = _manifest([_document("pg1/part-0001.md", _ALICE)])
    report = build_report(manifest)

    assert isinstance(report, DatasetReport)
    assert report.as_dict()["characters"] == report.characters
