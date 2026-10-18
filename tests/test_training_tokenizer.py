"""Tests for the deterministic character vocabulary.

These cover the properties ADR-0030's first experiment depends on: stable identifier assignment,
honest unknown-character handling, a lossless-for-known-characters round trip, and a fingerprint that
identifies a vocabulary exactly.
"""

from __future__ import annotations

import pytest

from chakaso.training.tokenizer import UNK_ID, CharVocabulary


def test_vocabulary_is_fit_only_from_training_characters() -> None:
    vocab = CharVocabulary(("abc",))

    assert set(vocab.chars) == {"a", "b", "c"}
    # Id 0 is reserved for unknown, real characters start at 1.
    assert vocab.size == 4
    assert vocab.id_of("a") >= 1


def test_identifier_assignment_is_by_codepoint_order_not_insertion_order() -> None:
    first = CharVocabulary(("cba",))
    second = CharVocabulary(("abc",))

    assert first.chars == second.chars == ("a", "b", "c")
    assert first.id_of("a") == 1 and first.id_of("c") == 3


def test_identical_vocabularies_share_a_fingerprint_and_different_ones_do_not() -> None:
    a = CharVocabulary(("hello",))
    b = CharVocabulary(("olleh",))  # same character set
    c = CharVocabulary(("help",))  # different character set

    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != c.fingerprint


def test_unknown_characters_map_to_the_reserved_unknown_id() -> None:
    vocab = CharVocabulary(("abc",))

    assert vocab.id_of("z") == UNK_ID
    encoded = vocab.encode("abz")
    assert encoded == (vocab.id_of("a"), vocab.id_of("b"), UNK_ID)


def test_encode_is_total_and_decode_is_lossless_for_known_characters() -> None:
    text = "the quick red fox"
    vocab = CharVocabulary((text,))

    encoded = vocab.encode(text)
    assert vocab.decode(encoded) == text


def test_decode_skips_unknown_ids_rather_than_inventing_a_character() -> None:
    vocab = CharVocabulary(("abc",))
    # An unknown id (0) must not decode to any real character.
    assert vocab.decode((vocab.id_of("a"), UNK_ID, vocab.id_of("b"))) == "ab"


def test_char_of_returns_none_for_unknown_and_out_of_range_ids() -> None:
    vocab = CharVocabulary(("abc",))

    assert vocab.char_of(UNK_ID) is None
    assert vocab.char_of(-1) is None
    assert vocab.char_of(vocab.size + 5) is None


def test_id_of_rejects_a_string_that_is_not_a_single_character() -> None:
    vocab = CharVocabulary(("abc",))

    with pytest.raises(ValueError, match="single character"):
        vocab.id_of("ab")


def test_unknown_fraction_reports_the_holdout_share_explicitly() -> None:
    vocab = CharVocabulary(("abc",))

    assert vocab.unknown_fraction("abc") == 0.0
    assert vocab.unknown_fraction("") == 0.0
    assert vocab.unknown_fraction("azzz") == pytest.approx(0.75)


def test_an_empty_or_characterless_training_input_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        CharVocabulary(())
    with pytest.raises(ValueError, match="no characters"):
        CharVocabulary(("", ""))
