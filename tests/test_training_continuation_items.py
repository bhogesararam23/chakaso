"""The continuation-discrimination probe's items and scoring: deterministic, held-out only, and honest about chance."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from chakaso.training.continuation_items import (
    build_items,
    items_digest,
    score_items,
)
from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.subword import train_subword_vocabulary

_WORDS = ["river", "stone", "field", "bird", "horse", "door", "letter", "ship", "tree", "king"]


def _text(seed: int, length: int) -> str:
    """Pseudo-random words (by hash), so no two passages repeat one another by accident of a short period."""
    return " ".join(
        _WORDS[int(hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), 16) % len(_WORDS)]
        for i in range(length)
    )


def _units() -> tuple[TextUnit, ...]:
    return tuple(
        TextUnit(
            identifier=f"{register}-w{work}/part-{part}",
            text=_text(work * 10 + part, 400),
            group=f"{register}-w{work}",
        )
        for register in ("prose", "science")
        for work in range(4)
        for part in range(2)
    )


_VOCABULARY = train_subword_vocabulary([u.text for u in _units()], target_size=300, min_frequency=2)


def _items(**overrides):  # type: ignore[no-untyped-def]
    settings = {
        "prefix_length": 16,
        "continuation_length": 6,
        "hard_gap": 40,
        "per_register": 5,
        "registers": ("prose", "science"),
        **overrides,
    }
    return build_items(_units(), _VOCABULARY, **settings)


def test_every_register_gets_the_requested_number_of_items() -> None:
    items = _items()

    assert len(items) == 10
    assert sorted({item.register for item in items}) == ["prose", "science"]
    assert sum(item.register == "prose" for item in items) == 5


def test_the_item_shapes_are_exactly_what_was_asked_for() -> None:
    for item in _items():
        assert len(item.prefix) == 16
        assert (
            len(item.true_continuation)
            == len(item.easy_distractor)
            == len(item.hard_distractor)
            == 6
        )


def test_the_true_continuation_follows_the_prefix_in_the_document() -> None:
    by_identifier = {unit.identifier: _VOCABULARY.encode(unit.text) for unit in _units()}

    for item in _items():
        document = by_identifier[item.document]
        window = document[item.offset : item.offset + 16 + 6]
        assert tuple(window[:16]) == item.prefix
        assert tuple(window[16:]) == item.true_continuation


def test_the_easy_distractor_comes_from_another_work_of_the_same_register() -> None:
    units = {unit.identifier: unit for unit in _units()}

    for item in _items():
        own, other = units[item.document], units[item.easy_source]
        assert other.group != own.group
        assert other.group.split("-")[0] == own.group.split("-")[0] == item.register


def test_the_hard_distractor_comes_from_the_same_document_far_enough_away() -> None:
    for item in _items():
        assert item.hard_offset is not None
        assert abs(item.hard_offset - (item.offset + 16)) >= 40
        document = _VOCABULARY.encode(
            {unit.identifier: unit for unit in _units()}[item.document].text
        )
        assert tuple(document[item.hard_offset : item.hard_offset + 6]) == item.hard_distractor


def test_no_distractor_is_the_true_continuation() -> None:
    for item in _items():
        assert item.easy_distractor != item.true_continuation
        assert item.hard_distractor != item.true_continuation


def test_building_twice_gives_identical_items_and_digest() -> None:
    assert _items() == _items()
    assert items_digest(_items()) == items_digest(_items())


def test_carrying_the_work_on_the_item_does_not_move_the_item_set_digest() -> None:
    """``ed21d50c…`` is what this fixture digested before ``work`` became a field, read off the committed code.

    E2623's 215-item set is cited by three records, so the hash must not move when an analysis-only field is added;
    the field is excluded because it follows from the hashed ``document``.
    """
    assert items_digest(_items()) == (
        "ed21d50cd322c41e626ad361550909c414b8860811a9bd76f0ab4bef520ea726"
    )


def test_every_item_names_the_work_its_passage_was_cut_from() -> None:
    for item in _items():
        assert item.work == item.document.split("/", 1)[0]
        assert item.easy_source != item.document


def test_the_digest_changes_when_anything_in_an_item_changes() -> None:
    items = _items()
    changed = (replace(items[0], true_continuation=(0,) * 6), *items[1:])

    assert items_digest(changed) != items_digest(items)


def test_a_register_without_enough_eligible_passages_is_refused_not_padded() -> None:
    with pytest.raises(CorpusError, match="prose"):
        _items(per_register=500)


def test_at_most_a_few_items_come_from_one_work() -> None:
    items = _items(per_register=6)
    works = [
        next(unit.group for unit in _units() if unit.identifier == item.document) for item in items
    ]

    assert max(works.count(work) for work in set(works)) <= 2


# ---------------------------------------------------------------- scoring


def _scorer_prefers(good_when_conditioned: bool):  # type: ignore[no-untyped-def]
    """A scorer for tests: with its prefix, true continuations score higher; without it, everything ties."""

    def score(prefix, continuation):  # type: ignore[no-untyped-def]
        if prefix and good_when_conditioned:
            return 0.0 if continuation in truth else -5.0
        return -1.0

    truth = {item.true_continuation for item in _items()}
    return score


def test_a_model_that_uses_its_prefix_is_perfect_and_one_that_ignores_it_sits_at_chance() -> None:
    items = _items()

    uses = score_items(items, _scorer_prefers(True))
    ignores = score_items(items, _scorer_prefers(False))

    assert uses.pmi_easy.accuracy == 1.0 and uses.pmi_hard.accuracy == 1.0
    assert ignores.pmi_easy.accuracy == 0.5 and ignores.pmi_hard.accuracy == 0.5
    assert ignores.pmi_easy.ties == len(items)


def test_the_prefix_blind_reading_is_reported_and_separate() -> None:
    items = _items()

    def blind_prefers_truth(_prefix, continuation):  # type: ignore[no-untyped-def]
        truth = {item.true_continuation for item in items}
        return 0.0 if continuation in truth else -3.0

    result = score_items(items, blind_prefers_truth)

    # The same preference with and without a prefix: all of the "accuracy" is the continuation itself, none of it the prefix.
    assert result.blind_easy.accuracy == 1.0
    assert result.pmi_easy.accuracy == 0.5


def test_results_are_split_by_register_with_intervals() -> None:
    result = score_items(_items(), _scorer_prefers(True))

    assert set(result.by_register) == {"prose", "science"}
    prose = result.by_register["prose"]["pmi_easy"]
    assert prose["accuracy"] == 1.0
    assert prose["interval_95"][0] > 0.5
    assert result.as_dict()["items"] == 10


def test_every_reading_keeps_its_per_item_outcomes() -> None:
    """Same reason as the pair set: two models on one item set are compared item by item, and the aggregate
    accuracy is not enough to run that test. ``as_dict`` has to carry it, because the record cites the JSON.
    """
    items = _items()
    result = score_items(items, _scorer_prefers(True))

    assert set(result.outcomes) == {
        "pmi_easy",
        "pmi_hard",
        "conditional_easy",
        "conditional_hard",
        "blind_easy",
        "blind_hard",
    }
    for name, pairs in result.outcomes.items():
        assert len(pairs) == len(items), name
        assert {identifier for identifier, _ in pairs} == {item.identifier for item in items}, name
    # The model that uses its prefix answers every item right, and the one that ignores it ties every item.
    assert all(value == 1.0 for _, value in result.outcomes["pmi_easy"])
    assert all(
        value == 0.5 for _, value in score_items(items, _scorer_prefers(False)).outcomes["pmi_easy"]
    )
    assert result.as_dict()["outcomes"]["pmi_hard"][0][0] == items[0].identifier
