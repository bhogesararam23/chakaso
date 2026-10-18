"""Tests for the experiment corpus's provenance and integrity."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from chakaso.training.corpus import CORPUS, Corpus, corpus_text


def test_the_corpus_carries_explicit_provenance() -> None:
    # The record must cite where the data came from and under what terms; those are required fields,
    # not left to be inferred from the text constant.
    assert CORPUS.name == "chakaso-toy-char-v1"
    assert CORPUS.version
    assert "Authored" in CORPUS.source
    assert CORPUS.licence.startswith("Apache-2.0")


def test_the_corpus_text_is_non_empty_and_reachable_both_ways() -> None:
    assert CORPUS.char_count == len(CORPUS.text)
    assert corpus_text() == CORPUS.text
    assert CORPUS.char_count > 0


def test_the_corpus_is_frozen_again_mutation() -> None:
    with pytest.raises(FrozenInstanceError):
        CORPUS.text = "changed"  # type: ignore[misc]


def test_an_empty_corpus_is_refused_rather_than_silently_accepted() -> None:
    with pytest.raises(ValueError, match="text"):
        Corpus(name="x", version="1", source="s", licence="l", text="")


def test_a_nameless_corpus_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        Corpus(name="", version="1", source="s", licence="l", text="abc")
