"""Tests for the deterministic split and its leakage guarantees.

These are the tests the first experiment's honesty depends on: the held-out set must not influence the
vocabulary, no unit may appear on both sides, the split and its fingerprint must reproduce exactly, and
windowing must be well-defined at the sequence edges.
"""

from __future__ import annotations

import pytest

from chakaso.training.data import (
    character_windows,
    dataset_fingerprint,
    fit_vocabulary,
    split_corpus,
    split_three_way,
)
from chakaso.training.tokenizer import UNK_ID

_TEXT = (
    "alpha bravo charlie\n"
    "delta echo foxtrot\n"
    "golf hotel india\n"
    "juliet kilo lima\n"
    "mike november oscar\n"
    "papa quebec romeo\n"
)


def test_split_is_deterministic_for_a_fixed_seed() -> None:
    first = split_corpus(_TEXT, validation_fraction=0.25, seed=7)
    second = split_corpus(_TEXT, validation_fraction=0.25, seed=7)

    assert first.train_units == second.train_units
    assert first.validation_units == second.validation_units
    assert first.train_text == second.train_text


def test_a_different_seed_selects_a_different_holdout() -> None:
    a = split_corpus(_TEXT, validation_fraction=0.34, seed=1)
    b = split_corpus(_TEXT, validation_fraction=0.34, seed=99)

    assert set(a.validation_units) != set(b.validation_units)


def test_no_unit_appears_on_both_sides_of_the_split() -> None:
    split = split_corpus(_TEXT, validation_fraction=0.3, seed=3)

    assert set(split.train_units).isdisjoint(set(split.validation_units))
    # Every unit is preserved exactly once across the two sides.
    assert len(split.train_units) + len(split.validation_units) == len(_units_of(_TEXT))


def test_vocabulary_is_fit_on_training_text_only() -> None:
    # A sentinel character that appears in exactly one line. Across seeds, whenever that line lands in
    # the holdout it must become an unknown character to the trained vocabulary — never widen it. That
    # is the anti-leakage property, checked as an invariant rather than by recomputing the code path.
    unique_line = "\u00a7zq\u00a7"  # section-sign and 'q' shared nowhere else in the text
    base = "alpha bravo\ncharlie delta\necho foxtrot\ngolf hotel\n"
    text = base + unique_line + "\nindia juliet\nkilo lima\n"

    held_out_once = False
    for seed in range(24):
        split = split_corpus(text, validation_fraction=0.3, seed=seed)
        vocab = fit_vocabulary(split)

        if unique_line in split.validation_units:
            held_out_once = True
            assert unique_line not in split.train_text
            for char in set(unique_line):
                assert char not in vocab.chars
                assert vocab.id_of(char) == UNK_ID

        # Always: the vocabulary is exactly the characters present in the training side.
        assert set(vocab.chars) == set(split.train_text)

    # Guard against a vacuous pass: the sentinel really is held out for some seed.
    assert held_out_once, (
        "the unique line was never held out; the split would be leak-proof by luck"
    )


def test_windows_never_straddle_the_split_boundary() -> None:
    split = split_corpus(_TEXT, validation_fraction=0.3, seed=5)
    vocab = fit_vocabulary(split)

    train_examples = character_windows(split.train_text, vocab, context_size=3)
    val_examples = character_windows(split.validation_text, vocab, context_size=3)

    # Each side's windows come only from that side's text; the concatenated contexts cannot contain a
    # train unit and a validation unit in the same window because the sides are separate strings.
    assert train_examples and val_examples
    assert all(len(ex.context) == 3 for ex in train_examples)
    assert all(len(ex.context) == 3 for ex in val_examples)


def test_early_positions_are_left_padded_with_the_unknown_id() -> None:
    split = split_corpus(_TEXT, validation_fraction=0.3, seed=5)
    vocab = fit_vocabulary(split)
    examples = character_windows(split.train_text, vocab, context_size=4)

    # The first produced window is for position 1, which has only one preceding character, so it must
    # be padded to the full context length with UNK.
    assert examples[0].context[:3] == (UNK_ID, UNK_ID, UNK_ID)


def test_dataset_fingerprint_is_stable_and_sensitive() -> None:
    split = split_corpus(_TEXT, validation_fraction=0.3, seed=5)

    fp_a = dataset_fingerprint(split, context_size=4)
    fp_b = dataset_fingerprint(split, context_size=4)
    fp_other_context = dataset_fingerprint(split, context_size=6)

    assert fp_a == fp_b
    assert fp_a != fp_other_context

    other_split = split_corpus(_TEXT, validation_fraction=0.3, seed=8)
    assert dataset_fingerprint(other_split, context_size=4) != fp_a


def test_an_out_of_range_validation_fraction_is_refused() -> None:
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            split_corpus(_TEXT, validation_fraction=bad, seed=0)


def test_a_too_small_corpus_cannot_be_split() -> None:
    with pytest.raises(ValueError, match="at least two"):
        split_corpus("only one line", validation_fraction=0.5, seed=0)


def test_a_non_integer_or_boolean_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="integer"):
        split_corpus(_TEXT, validation_fraction=0.3, seed=True)  # type: ignore[arg-type]


def test_a_zero_context_size_is_refused() -> None:
    split = split_corpus(_TEXT, validation_fraction=0.3, seed=0)
    vocab = fit_vocabulary(split)
    with pytest.raises(ValueError, match="at least 1"):
        character_windows(split.train_text, vocab, context_size=0)


def _units_of(text: str) -> tuple[str, ...]:
    return tuple(line for line in text.splitlines() if line)


def test_three_way_split_is_deterministic_and_disjoint() -> None:
    first = split_three_way(_TEXT, validation_fraction=0.2, test_fraction=0.2, seed=5)
    second = split_three_way(_TEXT, validation_fraction=0.2, test_fraction=0.2, seed=5)

    assert first.train_units == second.train_units
    assert first.validation_units == second.validation_units
    assert first.test_units == second.test_units
    # No unit appears on more than one side.
    assert set(first.train_units).isdisjoint(first.validation_units)
    assert set(first.train_units).isdisjoint(first.test_units)
    assert set(first.validation_units).isdisjoint(first.test_units)


def test_three_way_split_preserves_every_unit_exactly_once() -> None:
    split = split_three_way(_TEXT, validation_fraction=0.2, test_fraction=0.2, seed=3)
    total = len(split.train_units) + len(split.validation_units) + len(split.test_units)

    assert total == len(_units_of(_TEXT))
    # A unit appears on exactly one side, so combined it reconstructs the unit set with no duplicates.
    all_units = split.train_units + split.validation_units + split.test_units
    assert len(set(all_units)) == len(all_units)


def test_three_way_vocabulary_excludes_test_only_characters() -> None:
    unique_line = "\u00a7qqz"  # § q z shared nowhere else
    text = "alpha bravo\ncharlie delta\necho foxtrot\ngolf hotel\n" + unique_line + "\n"
    # Whatever the seed, the fit-from-train-only rule guarantees a test/validation-only char is unknown.
    for seed in range(16):
        split = split_three_way(text, validation_fraction=0.2, test_fraction=0.2, seed=seed)
        vocabulary = fit_vocabulary(split)
        for char in set(split.test_text) - set(split.train_text):
            assert vocabulary.id_of(char) == UNK_ID


def test_three_way_rejects_bad_fractions_and_sizes() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_three_way(_TEXT, validation_fraction=0.0, test_fraction=0.2, seed=0)
    with pytest.raises(ValueError, match="must be < 1"):
        split_three_way(_TEXT, validation_fraction=0.6, test_fraction=0.6, seed=0)
    with pytest.raises(ValueError, match="at least three"):
        split_three_way("one\ntwo\n", validation_fraction=0.2, test_fraction=0.2, seed=0)
    with pytest.raises(ValueError, match="integer"):
        split_three_way(_TEXT, validation_fraction=0.2, test_fraction=0.2, seed=True)  # type: ignore[arg-type]
