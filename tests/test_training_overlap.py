"""Tests for the near-duplicate leakage measure.

The measure is only useful if it finds the leaks an exact-digest screen cannot, and reports nothing for texts that
merely share vocabulary, so both directions are built here on purpose.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")

from chakaso.training.overlap import overlap_report, word_ngram_hashes

_BOOK = (
    "the old miller walked down to the river every morning before the sun was up and counted the barges that "
    "had tied up overnight while the mist lay along the water and the first birds began to call from the reeds"
)


def test_the_same_words_hash_the_same_in_every_process_and_case() -> None:
    first = word_ngram_hashes(_BOOK, 8)
    second = word_ngram_hashes(_BOOK.upper().replace(" ", "   "), 8)

    assert first.dtype.name == "uint64"
    assert list(first) == list(second), "case and spacing are not part of the text"
    assert len(first) == len(set(first))


def test_a_text_shorter_than_the_window_has_no_windows_and_is_reported_as_such() -> None:
    assert word_ngram_hashes("too few words here", 8).size == 0
    report = overlap_report([_BOOK], {"short": "too few words here"}, n=8)

    assert report["units"]["short"] == {"windows": 0, "shared": 0, "ratio": None}
    assert report["pooled"]["ratio"] is None


def test_an_edition_with_a_different_header_is_found_though_its_digest_differs() -> None:
    edition = "PREFACE TO THE SECOND EDITION.\n\n" + _BOOK + "\n\nTHE END."
    assert edition != _BOOK

    report = overlap_report([_BOOK], {"second-edition": edition}, n=8)

    unit = report["units"]["second-edition"]
    assert unit["ratio"] > 0.8, "almost every window of a reissued text is already in training"


def test_a_text_that_only_shares_vocabulary_shares_no_window() -> None:
    unrelated = (
        "a different story about sailors who crossed the water and counted the stars while the wind lay along "
        "the deck and the first light began to call across the waves over the quiet harbour"
    )

    report = overlap_report([_BOOK], {"unrelated": unrelated}, n=8)

    assert report["units"]["unrelated"]["ratio"] == 0.0
    assert report["units"]["unrelated"]["windows"] > 0


def test_a_partial_quotation_is_reported_in_proportion() -> None:
    half = (
        " ".join(_BOOK.split()[:20])
        + " "
        + "entirely new words follow here and continue for a good while longer"
    )

    ratio = overlap_report([_BOOK], {"half": half}, n=8)["units"]["half"]["ratio"]

    assert 0.2 < ratio < 0.8


def test_the_pooled_ratio_weights_long_units_more_than_short_ones() -> None:
    long_text = _BOOK + " " + _BOOK  # fully overlapped, twice the windows of the novel one
    novel = "nothing at all in common with the training text appears in this particular sentence of words here"

    report = overlap_report([_BOOK], {"long": long_text, "novel": novel}, n=8)

    assert (
        report["pooled"]["windows"]
        == report["units"]["long"]["windows"] + report["units"]["novel"]["windows"]
    )
    assert report["pooled"]["shared"] == report["units"]["long"]["shared"]
    assert 0.0 < report["pooled"]["ratio"] < 1.0


def test_an_empty_training_side_is_all_novel_and_a_bad_window_is_refused() -> None:
    assert overlap_report([], {"u": _BOOK})["units"]["u"]["ratio"] == 0.0
    with pytest.raises(ValueError, match="positive integer"):
        word_ngram_hashes(_BOOK, 0)
    with pytest.raises(ValueError, match="positive integer"):
        word_ngram_hashes(_BOOK, True)  # type: ignore[arg-type]
