"""Tests for share-composed training sets: the share is a number the run is told, and the arms are nested.

The properties that matter are the ones a sweep depends on and that a plausible implementation would break
without a failing run: the realised share is the declared one to within a document, a larger share of a class
contains everything a smaller one did, the endpoints are the runs filtering already produced, and a share the
pool cannot supply is refused instead of becoming a different mixture.
"""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.mixture import (
    identifiers_by_register,
    select_by_share,
    unit_identifiers,
)
from chakaso.training.subword import train_subword_vocabulary

_VOCABULARY = train_subword_vocabulary(
    ["the quick red fox ran over the dry sand again and again and over the red sand"],
    target_size=280,
    min_frequency=2,
)
_BODY = "the quick red fox ran over the dry sand again and again "


def _pool() -> tuple[TextUnit, ...]:
    """Twenty works of each kind, with lengths that differ so the last document taken overshoots a quota."""
    units: list[TextUnit] = []
    for index in range(20):
        units.append(TextUnit(f"p{index}", _BODY * (3 + index % 5), group=f"prose-{index}"))
        units.append(TextUnit(f"s{index}", _BODY * (2 + index % 7), group=f"science-{index}"))
    return tuple(units)


def _pool_identifiers(prefix: str) -> int:
    return sum(
        unit_identifiers(unit, _VOCABULARY) for unit in _pool() if unit.group.startswith(prefix)
    )


def test_the_realised_share_is_the_declared_one_to_within_the_last_document() -> None:
    budget = 600
    selection = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.25, max_tokens=budget, seed=7
    )

    longest = max(unit_identifiers(unit, _VOCABULARY) for unit in _pool())
    assert 150 <= selection.matched_identifiers < 150 + longest
    assert 450 <= selection.other_identifiers < 450 + longest
    assert selection.identifiers == sum(selection.identifiers_by_group.values())
    assert selection.realised_share == pytest.approx(0.25, abs=longest / budget)


def test_the_counts_by_work_add_up_to_what_the_selected_documents_contain() -> None:
    selection = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.5, max_tokens=500, seed=3
    )

    assert selection.identifiers == sum(
        unit_identifiers(unit, _VOCABULARY) for unit in selection.units
    )
    assert set(selection.identifiers_by_group) == {unit.group for unit in selection.units}


def test_a_larger_share_of_a_class_contains_everything_a_smaller_share_did() -> None:
    small = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.25, max_tokens=800, seed=7
    )
    large = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.75, max_tokens=800, seed=7
    )

    small_prose = {unit.identifier for unit in small.units if unit.group.startswith("prose-")}
    large_prose = {unit.identifier for unit in large.units if unit.group.startswith("prose-")}
    small_other = {unit.identifier for unit in small.units if not unit.group.startswith("prose-")}
    large_other = {unit.identifier for unit in large.units if not unit.group.startswith("prose-")}
    assert small_prose < large_prose
    assert large_other < small_other


def test_a_share_of_one_is_exactly_what_filtering_and_shuffling_gives() -> None:
    filtered = [unit for unit in _pool() if unit.group.startswith("prose-")]
    random.Random(7).shuffle(filtered)

    selection = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=1.0, max_tokens=400, seed=7
    )

    assert selection.other_identifiers == 0
    assert list(selection.units) == filtered[: len(selection.units)]


def test_a_share_of_zero_is_exactly_the_other_registers_filtered_and_shuffled() -> None:
    filtered = [unit for unit in _pool() if not unit.group.startswith("prose-")]
    random.Random(7).shuffle(filtered)

    selection = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.0, max_tokens=400, seed=7
    )

    assert selection.matched_identifiers == 0
    assert list(selection.units) == filtered[: len(selection.units)]


def test_a_mixture_interleaves_the_registers_instead_of_arriving_as_two_blocks() -> None:
    # Near the whole pool, so dozens of documents are drawn: two blocks would switch once, a shuffle ~half the time.
    selection = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.5, max_tokens=3000, seed=7
    )

    kinds = [unit.group.startswith("prose-") for unit in selection.units]
    switches = sum(1 for left, right in pairwise(kinds) if left != right)
    assert len(selection.units) > 15
    assert switches > 8


def test_one_seed_gives_one_selection_and_another_gives_another() -> None:
    first = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.5, max_tokens=600, seed=7
    )
    again = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.5, max_tokens=600, seed=7
    )
    other = select_by_share(
        _pool(), _VOCABULARY, prefix="prose-", share=0.5, max_tokens=600, seed=8
    )

    assert first == again
    assert [unit.identifier for unit in other.units] != [unit.identifier for unit in first.units]


def test_a_share_the_pool_cannot_supply_is_refused_not_quietly_changed() -> None:
    available = _pool_identifiers("prose-")

    with pytest.raises(CorpusError, match="declared share needs"):
        select_by_share(
            _pool(),
            _VOCABULARY,
            prefix="prose-",
            share=1.0,
            max_tokens=available + 1,
            seed=7,
        )


def test_a_prefix_that_matches_no_work_cannot_supply_a_positive_share() -> None:
    with pytest.raises(CorpusError, match="hold 0 identifiers"):
        select_by_share(_pool(), _VOCABULARY, prefix="poetry-", share=0.5, max_tokens=200, seed=7)


@pytest.mark.parametrize("share", [-0.1, 1.1, True, "0.5", None])
def test_a_share_outside_zero_to_one_or_not_a_number_is_refused(share: object) -> None:
    with pytest.raises(ValueError, match="share must be a number from 0 to 1"):
        select_by_share(
            _pool(),
            _VOCABULARY,
            prefix="prose-",
            share=share,  # type: ignore[arg-type]
            max_tokens=200,
            seed=7,
        )


@pytest.mark.parametrize("max_tokens", [0, -5, True, 2.5])
def test_a_budget_that_is_not_a_positive_integer_is_refused(max_tokens: object) -> None:
    with pytest.raises(ValueError, match="max_tokens must be an integer"):
        select_by_share(
            _pool(),
            _VOCABULARY,
            prefix="prose-",
            share=0.5,
            max_tokens=max_tokens,  # type: ignore[arg-type]
            seed=7,
        )


def test_an_empty_prefix_is_refused() -> None:
    with pytest.raises(ValueError, match="prefix must be a non-empty string"):
        select_by_share(_pool(), _VOCABULARY, prefix="", share=0.5, max_tokens=200, seed=7)


def test_counts_by_register_keep_the_total_and_count_an_unmatched_work_as_other() -> None:
    counts = {"prose-a": 10, "science-b": 5, "poetry-c": 3, "prose-d": 7}

    by_register = identifiers_by_register(counts, ("prose-", "science-", "medicine-"))

    assert by_register == {"prose-": 17, "science-": 5, "medicine-": 0, "other": 3}
    assert sum(by_register.values()) == sum(counts.values())
