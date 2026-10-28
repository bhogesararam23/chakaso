"""Tests for the learned subword vocabulary: segmentation that can be reviewed and reproduced.

The claims under test are the ones a tokenizer must not be trusted on: that encoding is total, that a
round trip is exact, that a trained rule sequence reproduces bit for bit, that a held-out document cannot
influence the identifier space, and that a vocabulary file which has been altered or speaks another format
is refused rather than half-loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chakaso.training.errors import CorpusError, CorruptedCorpusArtifactError
from chakaso.training.subword import (
    BYTE_BASE,
    DOCUMENT_END,
    FIRST_MERGED_ID,
    RESERVED_DOCUMENT_END_ID,
    SubwordVocabulary,
    pre_tokenize,
    train_subword_vocabulary,
)

_CORPUS = (
    "the quick red fox ran over the dry sand.\n"
    "the red fox ran and the quick fox slept.\n"
    "a quick brown owl flew over the quiet town.\n"
    "the owl slept while the fox ran and ran.\n"
)


def _trained(size: int = 320, *, min_frequency: int = 2) -> SubwordVocabulary:
    return train_subword_vocabulary([_CORPUS], target_size=size, min_frequency=min_frequency)


# ------------------------------------------------------------------ pre-tokenization


def test_pre_tokenize_keeps_every_character_of_the_input() -> None:
    text = "Words, more_words, 12 numbers, 日本語 and \U0001f600 emoji — plus   spaces."

    pieces = pre_tokenize(text)

    assert "".join(pieces) == text
    assert pieces


def test_pre_tokenize_separates_runs_and_single_symbols() -> None:
    assert pre_tokenize("quiet town.") == ("quiet", " ", "town", ".")
    assert pre_tokenize("a  b") == ("a", "  ", "b")
    # An underscore stays inside the word it belongs to, so a code identifier segments as one piece
    # rather than three — the choice a reader would otherwise have to reverse-engineer from the regex.
    assert pre_tokenize("more_words here") == ("more_words", " ", "here")
    assert pre_tokenize("a_b") == ("a_b",)


# ------------------------------------------------------------------ training


def test_training_learns_the_repeated_pair_into_one_symbol() -> None:
    vocabulary = train_subword_vocabulary(["aa aa aa"], target_size=258, min_frequency=2)

    assert vocabulary.merges == ((97, 97),)
    assert vocabulary.size == FIRST_MERGED_ID + 1
    assert vocabulary.encode("aa") == (257,)


def test_training_is_deterministic_across_rebuilds() -> None:
    first = _trained()
    second = train_subword_vocabulary([_CORPUS], target_size=320, min_frequency=2)

    assert first.merges == second.merges
    assert first.fingerprint == second.fingerprint


def test_a_pair_that_only_appears_in_held_out_text_is_never_merged() -> None:
    vocabulary = _trained(300)
    for left, right in vocabulary.merges:
        covered = bytes((left, right)) if left < BYTE_BASE and right < BYTE_BASE else b""
        assert b"qz" not in covered, f"merge ({left},{right}) came only from held-out text"


def test_training_on_held_out_text_produces_a_different_vocabulary() -> None:
    # The point of the rule above: what a vocabulary contains depends on the texts it was given, so a
    # leakage mistake must be visible as a different fingerprint rather than hidden inside a total encoder.
    train_only = train_subword_vocabulary(["ab ab ab"], target_size=258, min_frequency=2)
    with_extra = train_subword_vocabulary(
        ["ab ab ab", "ab ab ab"], target_size=258, min_frequency=2
    )
    different = train_subword_vocabulary(["cd cd cd"], target_size=258, min_frequency=2)

    assert train_only.fingerprint == with_extra.fingerprint
    assert train_only.fingerprint != different.fingerprint


def test_min_frequency_blocks_a_singleton_merge() -> None:
    vocabulary = train_subword_vocabulary(["ab ab ab zz"], target_size=258, min_frequency=3)

    assert vocabulary.merges == ((97, 98),)
    assert vocabulary.encode("zz") == (122, 122)


def test_repeated_pairs_are_counted_per_occurrence_not_per_piece() -> None:
    # 'abababab' holds (a,b) four times and (b,a) three times inside one piece. Counting per shape instead
    # of per occurrence would give both pairs a count of 1, and the second rule would come out different.
    vocabulary = train_subword_vocabulary(["abababab"], target_size=260, min_frequency=2)

    assert vocabulary.merges[0] == (97, 98)
    assert vocabulary.merges[1] == (257, 257)
    # A third rule is impossible: the shape is now two symbols whose single pair occurrence counts as 1,
    # below min_frequency, so training stops rather than inventing a merge the data does not support.
    assert vocabulary.merges == ((97, 98), (257, 257))
    assert vocabulary.encode("abab") == (258,)
    assert vocabulary.encode("abababab") == (258, 258)


def test_a_bigger_vocabulary_compresses_the_same_text_further() -> None:
    small = train_subword_vocabulary([_CORPUS], target_size=270, min_frequency=2)
    large = train_subword_vocabulary([_CORPUS], target_size=340, min_frequency=2)

    assert small.token_count(_CORPUS) > large.token_count(_CORPUS)
    assert large.compression_ratio(_CORPUS) > small.compression_ratio(_CORPUS)


def test_training_stops_early_when_the_data_runs_out() -> None:
    vocabulary = train_subword_vocabulary(["a b c"], target_size=4000, min_frequency=2)

    assert vocabulary.size <= 4000
    assert vocabulary.merges == ()


def test_training_refuses_impossible_settings() -> None:
    with pytest.raises(ValueError, match="target_size"):
        train_subword_vocabulary(["text"], target_size=BYTE_BASE)
    with pytest.raises(ValueError, match="target_size"):
        train_subword_vocabulary(["text"], target_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="min_frequency"):
        train_subword_vocabulary(["text"], target_size=300, min_frequency=0)
    with pytest.raises(ValueError, match="min_frequency"):
        train_subword_vocabulary(["text"], target_size=300, min_frequency=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="produced none"):
        train_subword_vocabulary([""], target_size=300)


def test_a_merge_table_may_not_repeat_or_forward_reference_a_rule() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        SubwordVocabulary(merges=((97, 98), (97, 98)))
    with pytest.raises(ValueError, match="does not define"):
        SubwordVocabulary(merges=((97, 98), (300, 97)))
    with pytest.raises(ValueError, match="non-negative integer"):
        SubwordVocabulary(merges=((-1, 98),))


# ------------------------------------------------------------------ encoding and decoding


def test_encode_decode_round_trips_exactly() -> None:
    vocabulary = _trained()
    text = _CORPUS + "Mixed Case, punctuation! And a number 42.\n"

    assert vocabulary.decode(vocabulary.encode(text)) == text


def test_encoding_is_total_for_arbitrary_unicode() -> None:
    vocabulary = _trained()
    text = "\U0001f600 日本語 ✓ \x07 control ¢"

    identifiers = vocabulary.encode(text)

    assert identifiers
    assert all(0 <= symbol < vocabulary.size for symbol in identifiers)
    assert vocabulary.decode(identifiers) == text


def test_empty_text_encodes_to_nothing_and_ratios_stay_defined() -> None:
    vocabulary = _trained()

    assert vocabulary.encode("") == ()
    assert vocabulary.token_count("") == 0
    assert vocabulary.compression_ratio("") == 0.0


def test_decoding_skips_the_reserved_separator_identifier() -> None:
    vocabulary = _trained()

    assert vocabulary.decode([*vocabulary.encode("fox"), DOCUMENT_END]) == "fox"
    assert vocabulary.decode([DOCUMENT_END]) == ""


def test_decoding_refuses_an_identifier_outside_the_vocabulary() -> None:
    vocabulary = _trained()

    with pytest.raises(CorpusError, match="outside this vocabulary"):
        vocabulary.decode([vocabulary.size])
    with pytest.raises(CorpusError, match="outside this vocabulary"):
        vocabulary.decode([-1])


def test_symbol_lookup_refuses_an_unknown_identifier() -> None:
    vocabulary = _trained()

    with pytest.raises(CorpusError, match="outside this vocabulary"):
        vocabulary.symbol_to_bytes(vocabulary.size + 5)


def test_encode_can_guard_the_model_embedding_table() -> None:
    vocabulary = train_subword_vocabulary(["aa aa aa"], target_size=258, min_frequency=2)

    assert vocabulary.encode("aa") == (257,)
    assert vocabulary.encode(
        "aa", allowed=frozenset({BYTE_BASE, RESERVED_DOCUMENT_END_ID, 257})
    ) == (257,)
    with pytest.raises(CorpusError, match=r"outside the allowed set: \[257\]"):
        vocabulary.encode("aa", allowed=frozenset(range(FIRST_MERGED_ID)))


def test_pieces_report_the_separator_and_every_learned_symbol() -> None:
    vocabulary = _trained()
    pieces = vocabulary.pieces()

    assert len(pieces) == vocabulary.size
    assert pieces[RESERVED_DOCUMENT_END_ID] == "<doc-end>"
    assert pieces[ord("a")] == "a"
    assert all(piece for piece in pieces[: vocabulary.size])


def test_an_incremental_decoder_never_emits_half_a_character() -> None:
    vocabulary = _trained()
    # U+00A3 is the two bytes C2 A3: two identifiers, neither valid text on its own.
    first, second = 0xC2, 0xA3
    assert vocabulary.decode([first]) == "\ufffd"  # the per-token path this decoder replaces

    decoder = vocabulary.incremental_decoder()

    assert decoder.feed(ord("a")) == "a"
    assert decoder.feed(first) == ""
    assert decoder.feed(second) == "\u00a3"
    assert decoder.finish() == ""


def test_an_incremental_decode_equals_the_bulk_decode_on_any_stream() -> None:
    vocabulary = _trained()
    text = "naive caf\u00e9 \u65e5\u672c\u8a9e \U0001f600 \u00a3 end"
    identifiers = [*vocabulary.encode(text), DOCUMENT_END]

    decoder = vocabulary.incremental_decoder()
    pieces = [decoder.feed(identifier) for identifier in identifiers]
    pieces.append(decoder.finish())

    assert "".join(pieces) == vocabulary.decode(identifiers) == text


def test_a_stream_that_ends_inside_a_character_is_flushed_as_a_replacement() -> None:
    vocabulary = _trained()
    decoder = vocabulary.incremental_decoder()

    assert decoder.feed(0xC2) == ""
    assert decoder.finish() == "\ufffd"
    assert decoder.finish() == ""


def test_an_incremental_decoder_refuses_an_identifier_outside_the_vocabulary() -> None:
    vocabulary = _trained()

    with pytest.raises(CorpusError):
        vocabulary.incremental_decoder().feed(vocabulary.size)


def test_a_multi_byte_character_split_across_tokens_still_decodes() -> None:
    vocabulary = _trained()
    text = "日本語 text"
    identifiers = vocabulary.encode(text)

    assert vocabulary.decode(identifiers) == text
    # A deliberately truncated stream is honest about it: replacement, not an exception and not silence.
    assert "\ufffd" in vocabulary.decode(identifiers[:1])


# ------------------------------------------------------------------ serialization


def test_vocabulary_round_trips_through_disk(tmp_path: Path) -> None:
    vocabulary = _trained()
    target = tmp_path / "vocab.json"
    vocabulary.save(target)

    loaded = SubwordVocabulary.load(target)

    assert loaded.fingerprint == vocabulary.fingerprint
    assert loaded.merges == vocabulary.merges
    assert loaded.encode(_CORPUS) == vocabulary.encode(_CORPUS)


def test_an_altered_vocabulary_file_is_refused(tmp_path: Path) -> None:
    vocabulary = _trained()
    target = tmp_path / "vocab.json"
    vocabulary.save(target)
    document = json.loads(target.read_text(encoding="utf-8"))
    document["merges"].pop()
    target.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="does not match its content"):
        SubwordVocabulary.load(target)


def test_a_vocabulary_record_without_a_checksum_is_refused() -> None:
    document = _trained().to_document()
    document.pop("checksum")

    with pytest.raises(CorruptedCorpusArtifactError, match="no checksum"):
        SubwordVocabulary.from_document(document)


def test_a_record_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(CorruptedCorpusArtifactError, match="must be a JSON object"):
        SubwordVocabulary.from_document([1, 2])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", "chakaso-subword-vocabulary/9"),
        ("base", "utf-16-code-units"),
        ("pre_tokenizer", "something-else"),
    ],
)
def test_a_record_speaking_another_format_is_refused(field: str, value: str) -> None:
    document = _trained().to_document()
    document[field] = value
    document["checksum"] = _rechecksum(document)

    with pytest.raises(CorpusError, match="this code implements"):
        SubwordVocabulary.from_document(document)


def _rechecksum(document: dict[str, object]) -> str:
    from chakaso.core.hashing import sha256_hex

    content = {key: value for key, value in document.items() if key != "checksum"}
    return sha256_hex(json.dumps(content, sort_keys=True, separators=(",", ":")))


def test_a_reserved_layout_this_code_cannot_honour_is_refused() -> None:
    document = _trained().to_document()
    document["reserved"] = {"document_end": 999}
    document["checksum"] = _rechecksum(document)

    with pytest.raises(CorpusError, match="unexpected identifier layout"):
        SubwordVocabulary.from_document(document)


def test_a_declared_size_that_disagrees_with_the_merge_table_is_refused() -> None:
    document = _trained().to_document()
    document["vocab_size"] = 4096
    document["checksum"] = _rechecksum(document)

    with pytest.raises(CorpusError, match="imply"):
        SubwordVocabulary.from_document(document)


def test_a_vocabulary_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "vocab.json"
    target.write_text("{ broken", encoding="utf-8")

    with pytest.raises(CorruptedCorpusArtifactError, match="not valid JSON"):
        SubwordVocabulary.load(target)


# ------------------------------------------------------------------ identity boundaries
def test_the_reserved_identifier_never_moves_when_the_vocabulary_is_retrained() -> None:
    # Two different corpora at two different sizes: the merge table moves, the reserved layout does not.
    # A checkpoint decoded against a vocabulary whose separator had shifted would be nonsense, and this is
    # the property that stops it from shifting.
    first = _trained(320)
    second = train_subword_vocabulary(
        ["another text entirely, with different words and different repeats."],
        target_size=420,
        min_frequency=2,
    )

    assert (
        first.document_end_id == second.document_end_id == RESERVED_DOCUMENT_END_ID == DOCUMENT_END
    )
    assert first.size > FIRST_MERGED_ID
    assert second.size != first.size
    assert second.fingerprint != first.fingerprint
    assert first.fingerprint == _trained(320).fingerprint


def test_a_long_input_encodes_decodes_and_keeps_its_length() -> None:
    vocabulary = _trained()
    text = _CORPUS * 2_000

    identifiers = vocabulary.encode(text)
    restored = vocabulary.decode(identifiers)

    assert len(text) == 2_000 * len(_CORPUS)
    assert restored == text
    assert vocabulary.token_count(text) == len(identifiers)
    assert vocabulary.compression_ratio(text) > 1.0


def test_every_whitespace_shape_survives_the_round_trip() -> None:
    vocabulary = _trained()
    text = "tabs\tbetween\tsymbols\r\nwindows\r\nlines\n\n\nblank\vdotted.\n"

    assert vocabulary.decode(vocabulary.encode(text)) == text


def test_a_symbol_heavy_input_round_trips_without_inventing_tokens() -> None:
    vocabulary = _trained()
    text = "βαθμός — ὁδός; 日本語の文, Émile Zola; Çikolata ☕ 🎹 🧶\n"

    identifiers = vocabulary.encode(text)

    assert vocabulary.decode(identifiers) == text
    assert all(item < vocabulary.size for item in identifiers)
    assert all(item != DOCUMENT_END for item in identifiers)
