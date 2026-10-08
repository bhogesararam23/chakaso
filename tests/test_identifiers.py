"""Tests for evidence identifier derivation and validation.

Identifier format is a contract that stored records, generated citations and
evaluation datasets all depend on, so the properties asserted here are the ones a
change would break silently.
"""

from __future__ import annotations

import pytest

from chakaso.core.errors import ChakasoError, IdentifierError
from chakaso.core.identifiers import (
    CHUNK_ID_PREFIX,
    SOURCE_ID_PREFIX,
    ChunkId,
    SourceId,
    derive_chunk_id,
    derive_source_id,
)

URL = "https://example.org/article"
OTHER_URL = "https://example.org/other"
CONTENT = "The specification says the window closes on 14 April."


def test_derived_source_id_has_the_documented_shape() -> None:
    source_id = derive_source_id(URL, CONTENT)

    assert source_id.value.startswith(SOURCE_ID_PREFIX)
    assert len(source_id.value) == len(SOURCE_ID_PREFIX) + 16
    assert SourceId.pattern().fullmatch(source_id.value)


def test_derived_chunk_id_has_the_documented_shape() -> None:
    chunk_id = derive_chunk_id(derive_source_id(URL, CONTENT), 0, CONTENT)

    assert chunk_id.value.startswith(CHUNK_ID_PREFIX)
    assert len(chunk_id.value) == len(CHUNK_ID_PREFIX) + 16
    assert ChunkId.pattern().fullmatch(chunk_id.value)


def test_derivation_is_deterministic() -> None:
    # Reproducibility is the point: replaying a run must produce the same
    # identifiers, or an evaluation cannot compare evidence across runs.
    assert derive_source_id(URL, CONTENT) == derive_source_id(URL, CONTENT)
    chunk_a = derive_chunk_id(derive_source_id(URL, CONTENT), 3, "text")
    chunk_b = derive_chunk_id(derive_source_id(URL, CONTENT), 3, "text")
    assert chunk_a == chunk_b


def test_repeated_retrieval_of_identical_content_deduplicates() -> None:
    assert derive_source_id(URL, CONTENT) == derive_source_id(URL, CONTENT)


def test_changed_content_produces_a_distinguishable_source() -> None:
    # ADR-0007: an earlier answer must keep pointing at the bytes it actually
    # used, so a changed page cannot reuse the old identifier.
    original = derive_source_id(URL, "The deadline is in March.")
    revised = derive_source_id(URL, "The deadline is 14 April.")

    assert original != revised


def test_different_urls_with_identical_content_are_different_sources() -> None:
    # Identity includes the URL, otherwise two sites republishing one document
    # would collapse into a single source and citations would lose their domain.
    assert derive_source_id(URL, CONTENT) != derive_source_id(OTHER_URL, CONTENT)


def test_chunk_identity_depends_on_position() -> None:
    source_id = derive_source_id(URL, CONTENT)

    assert derive_chunk_id(source_id, 0, "same text") != derive_chunk_id(source_id, 1, "same text")


def test_chunk_identity_depends_on_text() -> None:
    # Re-chunking must produce new identifiers rather than redefining what an
    # existing identifier refers to.
    source_id = derive_source_id(URL, CONTENT)

    assert derive_chunk_id(source_id, 0, "one") != derive_chunk_id(source_id, 0, "two")


def test_chunk_identity_depends_on_the_source() -> None:
    first = derive_source_id(URL, "one")
    second = derive_source_id(URL, "two")

    assert derive_chunk_id(first, 0, "text") != derive_chunk_id(second, 0, "text")


def test_chunk_positions_are_not_ambiguous_across_digits() -> None:
    # 1 + "2text" and 12 + "text" must not collide, which is what the part
    # separator in the hash input prevents.
    source_id = derive_source_id(URL, CONTENT)

    assert derive_chunk_id(source_id, 1, "2text") != derive_chunk_id(source_id, 12, "text")


def test_bytes_and_text_content_agree() -> None:
    assert derive_source_id(URL, CONTENT) == derive_source_id(URL, CONTENT.encode("utf-8"))


def test_empty_canonical_reference_is_rejected() -> None:
    # An empty reference means the source was never obtained, not that it is empty,
    # and deriving an identifier from nothing would give unrelated failures one name.
    with pytest.raises(IdentifierError, match="empty canonical reference"):
        derive_source_id("", CONTENT)


def test_negative_chunk_position_is_rejected() -> None:
    source_id = derive_source_id(URL, CONTENT)

    with pytest.raises(IdentifierError, match="must not be negative"):
        derive_chunk_id(source_id, -1, CONTENT)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "src_",
        "src_0123456789abcdef",  # chunk prefix is required instead
        "chk_0123456789abcdef0",  # too long
        "chk_0123456789abcde",  # too short
        "chk_0123456789ABCDEF",  # uppercase is not part of the format
        "chk_0123456789abcdeg",  # not hexadecimal
        "chk-0123456789abcdef",
        "0123456789abcdef",
        "chk_0123456789abcdef\n",
    ],
)
def test_malformed_identifiers_are_rejected_on_construction(value: str) -> None:
    with pytest.raises(IdentifierError) as info:
        ChunkId(value)

    message = str(info.value)
    assert repr(value) in message
    assert "chk_0000000000000000" in message


def test_error_message_names_the_expected_format() -> None:
    with pytest.raises(IdentifierError, match="chunk identifier"):
        ChunkId("nonsense")


def test_identifier_error_is_a_value_error_and_a_chakaso_error() -> None:
    assert issubclass(IdentifierError, ValueError)
    assert issubclass(IdentifierError, ChakasoError)


def test_identifiers_of_different_types_are_not_equal() -> None:
    # A source and a chunk never share an identifier, and the prefixes are what
    # make that true rather than a convention somebody has to remember.
    digest = "0123456789abcdef"
    assert SourceId(f"{SOURCE_ID_PREFIX}{digest}") != ChunkId(f"{CHUNK_ID_PREFIX}{digest}")


def test_identifiers_are_hashable_and_sortable() -> None:
    first = derive_source_id(URL, "a")
    second = derive_source_id(URL, "b")

    assert len({first, second}) == 2
    assert sorted([second, first]) == sorted([first, second])


def test_string_form_is_the_bare_identifier() -> None:
    # Identifiers are interpolated into prompts and printed next to citations,
    # where the dataclass repr would be noise.
    source_id = derive_source_id(URL, CONTENT)

    assert str(source_id) == source_id.value
    assert f"see {source_id}" == f"see {source_id.value}"


def test_parse_validates_untrusted_input() -> None:
    # This is the path generated text takes: validate before any lookup.
    assert SourceId.parse("src_0123456789abcdef") == SourceId("src_0123456789abcdef")

    with pytest.raises(IdentifierError):
        SourceId.parse("https://example.org/not-an-identifier")
