"""Tests for content hashing."""

from __future__ import annotations

import hashlib

import pytest

from chakaso.core.hashing import (
    SHA256_HEX_LENGTH,
    HashingError,
    join_parts,
    sha256_hex,
    short_sha256_hex,
)


def test_matches_the_standard_library_for_bytes() -> None:
    # The whole value of this module is that its digest is the digest everyone
    # else computes, so it is checked against hashlib directly rather than against
    # a recorded string that could be wrong in both places.
    assert sha256_hex(b"chakaso") == hashlib.sha256(b"chakaso").hexdigest()


def test_text_is_encoded_as_utf8() -> None:
    assert sha256_hex("chakaso") == sha256_hex(b"chakaso")


def test_non_ascii_text_is_hashed_as_utf8_not_something_else() -> None:
    assert sha256_hex("チャカソ") == hashlib.sha256("チャカソ".encode()).hexdigest()


def test_digest_is_a_full_sha256() -> None:
    digest = sha256_hex("anything")

    assert len(digest) == SHA256_HEX_LENGTH
    assert digest == digest.lower()
    assert all(character in "0123456789abcdef" for character in digest)


def test_empty_input_is_hashable() -> None:
    # An empty page is a real retrieval outcome, not an error, and its hash must
    # be well defined so that the source can still be identified.
    assert sha256_hex("") == hashlib.sha256(b"").hexdigest()


def test_different_content_gives_different_digests() -> None:
    assert sha256_hex("a") != sha256_hex("b")


def test_short_digest_is_a_prefix_of_the_full_digest() -> None:
    assert sha256_hex("chakaso").startswith(short_sha256_hex("chakaso", length=16))
    assert len(short_sha256_hex("chakaso", length=16)) == 16


@pytest.mark.parametrize("length", [0, -1, SHA256_HEX_LENGTH + 1])
def test_impossible_digest_length_is_rejected(length: int) -> None:
    with pytest.raises(HashingError, match="between 1 and"):
        short_sha256_hex("chakaso", length=length)


def test_hashing_error_is_a_value_error() -> None:
    # Untrusted input validation should be catchable as the built-in type.
    assert issubclass(HashingError, ValueError)


def test_part_separator_prevents_concatenation_ambiguity() -> None:
    # Without a separator, hashing ("ab", "c") and ("a", "bc") would agree, and
    # two different sources could claim the same identity.
    assert join_parts("ab", "c") != join_parts("a", "bc")
    assert sha256_hex(join_parts("ab", "c")) != sha256_hex(join_parts("a", "bc"))


def test_part_separator_joins_in_order() -> None:
    assert join_parts("a", "b") != join_parts("b", "a")


def test_join_parts_with_nothing_is_empty() -> None:
    assert join_parts() == ""
