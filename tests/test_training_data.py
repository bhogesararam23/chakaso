"""Tests for the deterministic split and its leakage guarantees.

These are the tests the first experiment's honesty depends on: the held-out set must not influence the
vocabulary, no unit may appear on both sides, the split and its fingerprint must reproduce exactly, and
windowing must be well-defined at the sequence edges.
"""

from __future__ import annotations

import pytest

from chakaso.training.data import (
    TextUnit,
    character_windows,
    dataset_fingerprint,
    fit_vocabulary,
    split_corpus,
    split_text_units,
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


# ---------------------------------------------------------------------------
# Document-level splitting: the shape a real corpus needs.
# ---------------------------------------------------------------------------


def _units(count: int, *, marker: str = "doc") -> tuple[TextUnit, ...]:
    return tuple(
        TextUnit(
            identifier=f"{marker}-{index}",
            text=f"document {index} says its own thing about topic {index * 7}.\n",
        )
        for index in range(count)
    )


def test_every_document_lands_on_exactly_one_side() -> None:
    units = _units(10)

    split = split_text_units(units, validation_fraction=0.2, test_fraction=0.2, seed=3)

    assigned = split.train + split.validation + split.test
    assert len(assigned) == len(units)
    identifiers = [unit.identifier for unit in assigned]
    assert len(set(identifiers)) == len(identifiers)
    assert set(identifiers) == {unit.identifier for unit in units}


def test_document_split_is_deterministic_and_seed_dependent() -> None:
    units = _units(11)
    first = split_text_units(units, seed=5)
    second = split_text_units(units, seed=5)
    other = split_text_units(units, seed=6)

    assert first.fingerprint == second.fingerprint
    assert first.identifiers == second.identifiers
    # A seed that changed nothing would make "deterministic" and "seeded" the same claim twice; at least
    # one document must move for the seed to be doing work.
    assert other.fingerprint != first.fingerprint


def test_a_marker_unique_to_a_held_out_document_never_reaches_the_training_text() -> None:
    # The leakage property, stated so it can actually fail: each document holds a token no other document
    # has, so a token from a validation or test document appearing in the training text is a real leak.
    units = tuple(
        TextUnit(
            identifier=f"doc-{index}",
            text=f"plain shared prose about the same subject zzx{index:03d}\n",
        )
        for index in range(12)
    )
    leaked = 0
    for seed in range(24):
        split = split_text_units(units, validation_fraction=0.25, test_fraction=0.25, seed=seed)
        held_out = split.validation + split.test
        assert held_out, "a split with no held-out documents proves nothing"
        for unit in held_out:
            marker = unit.text.split("zzx")[1][:3]
            if f"zzx{marker}" in split.train_text:
                leaked += 1
    assert leaked == 0


def test_duplicate_content_is_dropped_once_and_reported() -> None:
    original = TextUnit(identifier="doc-1", text="the same paragraph, twice.\n")
    twin = TextUnit(identifier="doc-2", text="the same paragraph, twice.\n")
    others = _units(4, marker="other")

    split = split_text_units((original, *others, twin), seed=1)

    assert [unit.identifier for unit in split.dropped_duplicates] == ["doc-2"]
    assigned = {unit.identifier for unit in split.train + split.validation + split.test}
    assert "doc-2" not in assigned
    assert "doc-1" in assigned


def test_a_duplicated_document_cannot_straddle_the_split() -> None:
    repeated = TextUnit(
        identifier="copy-a", text="identical body text that must not be memorised.\n"
    )
    units = (repeated, TextUnit(identifier="copy-b", text=repeated.text), *_units(5, marker="fill"))

    for seed in range(12):
        split = split_text_units(units, seed=seed)
        held_out = split.validation + split.test
        in_train = any(unit.text == repeated.text for unit in split.train)
        in_held_out = any(unit.text == repeated.text for unit in held_out)
        assert not (in_train and in_held_out)


def test_side_texts_hold_their_own_documents_and_no_others() -> None:
    units = _units(9)

    split = split_text_units(units, seed=2)

    for unit in split.train:
        assert unit.text in split.train_text
        assert unit.text not in split.validation_text
        assert unit.text not in split.test_text
    assert split.train_text.count("\n\n") == max(0, len(split.train) - 1)


def test_document_split_rejects_bad_fractions_seeds_and_sizes() -> None:
    units = _units(6)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        split_text_units(units, validation_fraction=0.0)
    with pytest.raises(ValueError, match="must be < 1"):
        split_text_units(units, validation_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="integer"):
        split_text_units(units, seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least three distinct documents"):
        split_text_units(_units(2))


def test_document_split_refuses_empty_or_repeated_units() -> None:
    units = _units(4)
    with pytest.raises(ValueError, match="non-empty identifier"):
        split_text_units((*units, TextUnit(identifier="", text="orphan text\n")))
    with pytest.raises(ValueError, match="has no text"):
        split_text_units((*units, TextUnit(identifier="blank", text="")))
    with pytest.raises(ValueError, match="appears twice"):
        split_text_units((*units, TextUnit(identifier="doc-0", text="a different body\n")))


def test_split_fingerprint_moves_when_the_assignment_moves() -> None:
    units = _units(12)
    before = split_text_units(units, seed=1)
    after = split_text_units(
        (*units, TextUnit(identifier="doc-12", text="a new document\n")), seed=1
    )

    assert before.fingerprint != after.fingerprint
