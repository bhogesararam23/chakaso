"""Tests for sequence packing: window arithmetic that cannot leak.

The two properties that matter are exact: every example predicts the token immediately after its context,
and no example ever mixes documents from opposite sides of a split. Both are checked by reconstructing the
arithmetic by hand on tiny controlled inputs rather than by trusting the slice bounds.
"""

from __future__ import annotations

import pytest

from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.packing import PackedExample, count_tokens, pack_units, token_stream
from chakaso.training.subword import DOCUMENT_END, train_subword_vocabulary

_VOCABULARY = train_subword_vocabulary(
    ["the quick red fox ran over the dry sand again and again"], target_size=280, min_frequency=2
)


def _units(*texts: str) -> tuple[TextUnit, ...]:
    return tuple(TextUnit(identifier=f"doc-{index}", text=text) for index, text in enumerate(texts))


# ------------------------------------------------------------------ window arithmetic


def test_targets_are_the_input_shifted_by_exactly_one_token() -> None:
    units = _units("the quick red fox ran over the dry sand again and again")
    context_length = 8
    stream = token_stream(units, _VOCABULARY)

    batch = pack_units(units, _VOCABULARY, context_length=context_length)

    width = context_length + 1
    for index, example in enumerate(batch.examples):
        start = index * width
        assert example.input_ids == stream[start : start + context_length]
        assert example.targets == stream[start + 1 : start + width]
        # The target is the context shifted by one, so the final target is the token after the block and
        # the first context token is never scored as a target inside this example.
        assert example.targets[:-1] == example.input_ids[1:]
        assert example.targets[-1] == stream[start + width - 1]


def test_block_width_and_accounting_add_up() -> None:
    units = _units("the quick red fox ran over the dry sand again and again")
    stream = token_stream(units, _VOCABULARY)

    batch = pack_units(units, _VOCABULARY, context_length=4)

    assert batch.tokens_available == len(stream)
    assert len(batch.examples) == len(stream) // 5
    assert batch.dropped_tail == len(stream) - 5 * len(batch.examples)
    assert batch.positions == 4 * len(batch.examples)
    assert batch.context_length == 4
    assert batch.dropped_tail < 5


def test_a_long_document_is_cut_into_blocks_rather_than_discarded() -> None:
    text = "the quick red fox ran over the dry sand again and again " * 3
    units = _units(text)
    stream = token_stream(units, _VOCABULARY)

    batch = pack_units(units, _VOCABULARY, context_length=8)

    assert len(batch.examples) == len(stream) // 9
    assert len(batch.examples) > 3


def test_every_document_is_terminated_by_the_separator() -> None:
    units = _units("first document body", "second document body")

    stream = token_stream(units, _VOCABULARY)

    assert stream[-1] == DOCUMENT_END
    assert stream.count(DOCUMENT_END) == 2


def test_document_tracking_is_off_by_default() -> None:
    batch = pack_units(
        _units("first document body", "second document body"), _VOCABULARY, context_length=2
    )

    assert all(example.segment_ids is None for example in batch.examples)


def test_document_tracking_records_the_source_of_every_position_and_crosses_boundaries() -> None:
    units = _units(
        ("first document body is longer than the second " * 2).rstrip(),
        "second",
    )

    batch = pack_units(units, _VOCABULARY, context_length=3, track_documents=True)

    for example in batch.examples:
        assert example.segment_ids is not None
        assert len(example.segment_ids) == len(example.input_ids)
    # A block that spans the separator carries both document indices; that is the fact a masking run uses.
    assert any(len(set(example.segment_ids or ())) > 1 for example in batch.examples)


def test_a_segment_identifier_of_the_wrong_length_is_refused() -> None:
    with pytest.raises(CorpusError, match="segment identifiers"):
        PackedExample(input_ids=(1, 2, 3), targets=(2, 3, 4), segment_ids=(0, 0))
    with pytest.raises(CorpusError, match="non-negative"):
        PackedExample(input_ids=(1, 2, 3), targets=(2, 3, 4), segment_ids=(0, -1, 0))


def test_separators_can_be_switched_off_for_a_continuous_stream() -> None:
    units = _units("alpha beta", "gamma delta")

    with_separators = token_stream(units, _VOCABULARY)
    without = token_stream(units, _VOCABULARY, separate_documents=False)

    assert len(with_separators) == len(without) + 2
    assert DOCUMENT_END not in without


# ------------------------------------------------------------------ leakage


def test_a_packed_block_never_mixes_documents_from_opposite_sides() -> None:
    train = _units("alpha beta gamma delta epsilon zeta " * 6)
    held_out = _units("qqq www eee rrr ttt yyy " * 6)
    train_vocabulary = train_subword_vocabulary(
        ["alpha beta gamma delta epsilon zeta " * 6], target_size=270, min_frequency=2
    )

    train_batch = pack_units(train, train_vocabulary, context_length=16)
    held_out_stream = token_stream(held_out, train_vocabulary)

    for example in train_batch.examples:
        joined = example.input_ids + example.targets[-1:]
        assert not any(
            held_out_stream[index : index + len(joined)] == joined
            for index in range(len(held_out_stream) - len(joined) + 1)
        )


def test_a_held_out_marker_token_never_appears_in_packed_training_positions() -> None:
    marker_units = _units("ordinary words here zzz9 unusual marker")
    body = "ordinary words here and more ordinary words here and again"
    vocabulary = train_subword_vocabulary([body], target_size=270, min_frequency=2)

    batch = pack_units(_units(body), vocabulary, context_length=8)

    marker_stream = token_stream(marker_units, vocabulary)
    unique = {
        token
        for token in marker_stream
        if token != DOCUMENT_END and token not in token_stream(_units(body), vocabulary)
    }
    for example in batch.examples:
        assert not unique & set(example.input_ids)
        assert not unique & set(example.targets)


# ------------------------------------------------------------------ counting


def test_count_tokens_matches_the_stream_it_describes() -> None:
    units = _units("the quick red fox", "ran over the dry sand")

    assert count_tokens(units, _VOCABULARY) == len(token_stream(units, _VOCABULARY))


def test_counting_is_additive_over_documents() -> None:
    first = _units("the quick red fox")
    second = _units("ran over the dry sand")

    assert count_tokens(first, _VOCABULARY) + count_tokens(second, _VOCABULARY) == count_tokens(
        (*first, *second), _VOCABULARY
    )


# ------------------------------------------------------------------ determinism and refusal


def test_packing_is_deterministic_for_the_same_input() -> None:
    units = _units("the quick red fox ran over the dry sand again and again")

    first = pack_units(units, _VOCABULARY, context_length=6)
    second = pack_units(units, _VOCABULARY, context_length=6)

    assert [example.input_ids for example in first.examples] == [
        example.input_ids for example in second.examples
    ]


def test_a_stream_too_short_to_fill_one_block_is_refused() -> None:
    units = _units("ab")

    with pytest.raises(CorpusError, match="produced no examples"):
        pack_units(units, _VOCABULARY, context_length=32)


@pytest.mark.parametrize("context_length", [0, -1, True, "8", 2.0])
def test_an_invalid_context_length_is_refused(context_length: object) -> None:
    with pytest.raises(CorpusError, match="context_length"):
        pack_units(_units("some text here"), _VOCABULARY, context_length=context_length)  # type: ignore[arg-type]


def test_an_example_whose_input_and_target_lengths_disagree_is_refused() -> None:
    with pytest.raises(CorpusError, match="lengths disagree"):
        PackedExample(input_ids=(1, 2, 3), targets=(1, 2))
    with pytest.raises(CorpusError, match="no positions"):
        PackedExample(input_ids=(), targets=())
