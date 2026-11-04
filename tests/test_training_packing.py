"""Tests for sequence packing: window arithmetic that cannot leak.

The two properties that matter are exact: every example predicts the token immediately after its context,
and no example ever mixes documents from opposite sides of a split. Both are checked by reconstructing the
arithmetic by hand on tiny controlled inputs rather than by trusting the slice bounds.
"""

from __future__ import annotations

import pytest

from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.packing import (
    PackedBatch,
    PackedExample,
    count_tokens,
    pack_units,
    token_stream,
)
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


# ------------------------------------------------------------------ the training-token cap


def test_a_token_cap_stops_at_a_whole_unit_and_says_so() -> None:
    units = _units(*["the quick red fox ran over the dry sand again"] * 8)
    per_unit = len(token_stream(units[:1], _VOCABULARY))
    cap = per_unit * 3 + 1

    capped = pack_units(units, _VOCABULARY, context_length=4, max_tokens=cap)
    uncapped = pack_units(units, _VOCABULARY, context_length=4)

    # Four whole units reach the cap (three do not), so the stream is at least the cap and at most one
    # unit's worth over it. Nothing is half packed.
    assert per_unit * 4 <= capped.tokens_available <= cap + per_unit
    assert capped.document_count == 4
    assert capped.tokens_available < uncapped.tokens_available
    assert capped.tokens_available == len(token_stream(units[:4], _VOCABULARY))


def test_a_cap_above_the_whole_corpus_packs_everything() -> None:
    units = _units(*["the quick red fox ran over the dry sand again"] * 3)
    full = pack_units(units, _VOCABULARY, context_length=4)
    capped = pack_units(units, _VOCABULARY, context_length=4, max_tokens=10_000)

    assert capped.tokens_available == full.tokens_available
    assert capped.document_count == full.document_count


def test_the_cap_is_validated() -> None:
    units = _units("the quick red fox ran over the dry sand again")
    with pytest.raises(CorpusError, match="max_tokens"):
        pack_units(units, _VOCABULARY, context_length=4, max_tokens=0)
    with pytest.raises(CorpusError, match="max_tokens"):
        pack_units(units, _VOCABULARY, context_length=4, max_tokens=True)  # type: ignore[arg-type]


# ------------------------------------------------ selecting evaluation documents with another vocabulary (ADR-0048)


def test_selecting_by_a_reference_vocabulary_picks_the_documents_a_capped_pack_would_have() -> None:
    from chakaso.training.packing import select_units_by_token_cap

    vocabulary = train_subword_vocabulary(
        ["the quick red fox ran over the dry sand"] * 4, target_size=280, min_frequency=2
    )
    units = tuple(
        TextUnit(identifier=f"u{i}", text="the quick red fox ran. " * (i + 2)) for i in range(8)
    )

    for cap in (1, 30, 80, 200, 10_000):
        chosen = select_units_by_token_cap(units, vocabulary, cap)
        try:
            capped = pack_units(units, vocabulary, context_length=8, max_tokens=cap)
        except CorpusError:
            continue  # too small to fill a block; the selection still returns a prefix
        selected = pack_units(chosen, vocabulary, context_length=8)
        assert selected.examples == capped.examples
        assert selected.document_count == capped.document_count == len(chosen)


def test_the_documents_selected_depend_on_the_vocabulary_that_counts() -> None:
    from chakaso.training.packing import select_units_by_token_cap

    text = "the cat sat on the mat and the dog sat on the log. " * 6
    coarse = train_subword_vocabulary([text], target_size=300, min_frequency=2)
    fine = train_subword_vocabulary([text], target_size=258, min_frequency=2)
    units = tuple(TextUnit(identifier=f"u{i}", text=text) for i in range(10))

    # A vocabulary that spends more identifiers per document reaches the cap in fewer documents.
    assert len(select_units_by_token_cap(units, fine, 400)) < len(
        select_units_by_token_cap(units, coarse, 400)
    )


def test_a_cap_that_is_not_a_positive_integer_is_refused() -> None:
    from chakaso.training.packing import select_units_by_token_cap

    vocabulary = train_subword_vocabulary(["a b a b a b"], target_size=260, min_frequency=2)

    with pytest.raises(CorpusError, match="max_tokens"):
        select_units_by_token_cap((TextUnit(identifier="u", text="a b"),), vocabulary, 0)


# ------------------------------------------------------------------ what a pack is made of


def _tracked_batch(*segment_blocks: tuple[int, ...]) -> PackedBatch:
    """A pack built by hand: each block is three wide, so two blocks give six scored positions."""
    examples = tuple(
        PackedExample(input_ids=(1,) * 3, targets=(2,) * 3, segment_ids=block)
        for block in segment_blocks
    )
    return PackedBatch(
        examples=examples,
        tokens_available=7,
        dropped_tail=1,
        document_count=len({value for block in segment_blocks for value in block}),
    )


def test_a_target_belongs_to_the_document_of_the_token_it_predicts() -> None:
    from chakaso.training.packing import target_segments

    # The seam rule by hand. Block one's inputs are (0, 0, 1), so its first two targets are the second and
    # third input tokens — documents 0 and 1 — and its last target is the token *after* the block, which is
    # block two's first input, document 1. Block two's last target lies past the end of the pack, so it is
    # attributed to the document the stream was still inside: its own final segment.
    batch = _tracked_batch((0, 0, 1), (1, 1, 1))

    assert target_segments(batch.examples) == [0, 1, 1, 1, 1, 1]
    assert len(target_segments(batch.examples)) == batch.positions


def test_a_pack_without_document_identifiers_is_refused_by_name() -> None:
    from chakaso.training.packing import target_segments

    untracked = PackedBatch(
        examples=(PackedExample(input_ids=(1, 2, 3), targets=(2, 3, 4)),),
        tokens_available=4,
        dropped_tail=1,
        document_count=1,
    )

    with pytest.raises(CorpusError, match="track_documents"):
        target_segments(untracked.examples)


def test_blocks_of_mixed_width_have_no_seam_rule_and_are_refused() -> None:
    from chakaso.training.packing import target_segments

    mixed = (
        PackedExample(input_ids=(1, 2), targets=(2, 3), segment_ids=(0, 1)),
        PackedExample(input_ids=(1, 2, 3), targets=(2, 3, 4), segment_ids=(0, 1, 1)),
    )

    with pytest.raises(CorpusError, match="one block width"):
        target_segments(mixed)


def test_the_composition_counts_the_registers_the_scored_positions_came_from() -> None:
    from chakaso.training.packing import sample_composition

    units = (
        TextUnit(identifier="a", text="first", group="prose-anauthor"),
        TextUnit(identifier="b", text="second", group="mathematics-anauthor"),
    )
    composition = sample_composition(_tracked_batch((0, 0, 1), (1, 1, 1)), units)

    assert composition["documents"] == 2
    assert composition["documents_available"] == 2
    assert composition["works"] == 2
    assert composition["works_available"] == 2
    assert composition["positions_scored"] == 6
    assert composition["registers"]["prose-"] == {
        "documents": 1,
        "documents_available": 1,
        "positions": 1,
        "position_share": 0.1667,
        "works": 1,
        "works_available": 1,
    }
    assert composition["registers"]["mathematics-"]["positions"] == 5
    assert composition["registers"]["mathematics-"]["position_share"] == 0.8333
    # The registers partition the pack: every scored position is counted once, in exactly one register.
    assert (
        sum(entry["positions"] for entry in composition["registers"].values())
        == composition["positions_scored"]
    )


def test_a_register_the_cap_never_reached_is_stated_as_an_absence() -> None:
    """The fact K-033 found was that medicine is *not in the pack at all*, and a missing key is a poor way to say it."""
    from chakaso.training.packing import sample_composition

    units = (
        TextUnit(identifier="a", text="first", group="prose-anauthor"),
        TextUnit(identifier="b", text="second", group="medicine-anauthor"),
    )
    composition = sample_composition(_tracked_batch((0, 0, 0), (0, 0, 0)), units)

    assert composition["registers"]["medicine-"] == {
        "documents": 0,
        "documents_available": 1,
        "positions": 0,
        "position_share": 0.0,
        "works": 0,
        "works_available": 1,
    }
    assert composition["documents"] == 1
    assert composition["documents_available"] == 2
    assert composition["registers"]["prose-"]["position_share"] == 1.0


def test_a_composition_that_cannot_name_its_documents_is_refused() -> None:
    from chakaso.training.packing import sample_composition

    with pytest.raises(CorpusError, match="segment index is only meaningful"):
        sample_composition(
            _tracked_batch((0, 0, 2), (2, 2, 2)),
            (TextUnit(identifier="a", text="x", group="prose-a"),),
        )


def test_a_packed_sample_names_the_documents_it_reached_and_not_the_side_it_came_from() -> None:
    from chakaso.training.packing import sample_composition

    units = tuple(
        TextUnit(
            identifier=f"doc-{index}",
            text="the quick red fox ran over the dry sand again and again",
            group=f"{'prose' if index % 2 else 'mathematics'}-work-{index}",
        )
        for index in range(8)
    )
    capped = pack_units(units, _VOCABULARY, context_length=8, track_documents=True, max_tokens=30)
    composition = sample_composition(capped, units)

    assert composition["positions_scored"] == capped.positions
    # A cap that stops mid-side leaves documents unscored — which is the fact K-033 asks the report to state.
    assert composition["documents"] < composition["documents_available"] == len(units)
    assert set(composition["registers"]) == {"mathematics-", "prose-"}
    assert (
        sum(entry["documents"] for entry in composition["registers"].values())
        == composition["documents"]
    )
    assert (
        sum(entry["works"] for entry in composition["registers"].values()) == (composition["works"])
    )


def test_an_ungrouped_document_is_counted_rather_than_dropped() -> None:
    from chakaso.training.packing import sample_composition

    units = (TextUnit(identifier="a", text="x", group=""),)

    assert sample_composition(_tracked_batch((0, 0, 0), (0, 0, 0)), units)["registers"] == {
        "(ungrouped)": {
            "documents": 1,
            "documents_available": 1,
            "positions": 6,
            "position_share": 1.0,
            "works": 1,
            "works_available": 1,
        }
    }
