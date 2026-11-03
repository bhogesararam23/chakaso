"""Tests for the same-region continuation construction: what its distractors are, and what chance means for each.

The construction has to prove three things: that a distractor is really the document's own text at the position
claimed, that the pointwise-mutual-information reading is what makes a *same-topic* comparison mean use of the
passage rather than a preference for orderly English, and that nothing about the item set moves when it is
rebuilt.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from chakaso.training.context_continuations import (
    CONTEXT_DISTRACTOR_KINDS,
    build_context_items,
    context_items_digest,
    score_context_items,
)
from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.subword import train_subword_vocabulary

_WORDS = ["river", "stone", "field", "bird", "horse", "door", "letter", "ship", "tree", "king"]


def _text(seed: int, length: int) -> str:
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
        "per_register": 5,
        "registers": ("prose", "science"),
        **overrides,
    }
    return build_context_items(_units(), _VOCABULARY, **settings)


def _encoded(unit: TextUnit) -> tuple[int, ...]:
    return _VOCABULARY.encode(unit.text)


# ------------------------------------------------------------------ construction


def test_every_register_gets_its_items_and_every_item_carries_every_kind() -> None:
    items = _items()

    assert len(items) == 10
    assert {item.register for item in items} == {"prose", "science"}
    for item in items:
        assert item.kinds() == CONTEXT_DISTRACTOR_KINDS


def test_the_distractors_are_the_documents_own_text_at_the_positions_claimed() -> None:
    items = _items()
    by_document = {unit.identifier: unit for unit in _units()}

    for item in items:
        ids = _encoded(by_document[item.document])
        prefix_end = item.offset + len(item.prefix)
        truth_end = prefix_end + len(item.true_continuation)
        assert item.prefix == ids[item.offset : prefix_end]
        assert item.true_continuation == ids[prefix_end:truth_end]
        for kind, position, span in item.distractors:
            assert len(span) == len(item.true_continuation), kind
            if kind == "prior_span":
                # Ends where the passage begins: same document, same style, none of it visible to the model.
                assert position + len(span) == item.offset
            elif kind == "next_span":
                # Begins where the continuation ends.
                assert position == truth_end
            else:
                assert position == prefix_end
            assert span != item.true_continuation, kind
            if kind in ("prior_span", "next_span"):
                assert span == ids[position : position + len(span)]


def test_the_form_controls_keep_the_identifiers_and_destroy_only_the_order() -> None:
    truth = next(iter(_items())).true_continuation

    kinds = {kind: span for kind, _, span in _items()[0].distractors}

    assert sorted(kinds["permuted"]) == sorted(truth)
    assert kinds["reversed"] == truth[::-1]
    assert kinds["permuted"] != truth


def test_rebuilding_the_set_reproduces_its_digest_and_a_changed_span_moves_it() -> None:
    items = _items()
    first = context_items_digest(items)

    assert first == context_items_digest(_items())
    changed = replace(items[0], distractors=(items[0].distractors[0],))
    assert first != context_items_digest([changed, *items[1:]])


def test_at_most_max_per_work_items_come_from_one_work() -> None:
    items = _items(per_register=6, max_per_work=2)

    counts: dict[str, int] = {}
    for item in items:
        counts[item.document.split("/")[0]] = counts.get(item.document.split("/")[0], 0) + 1
    assert max(counts.values()) <= 2


def test_a_register_without_enough_eligible_passages_is_a_refusal_not_a_short_set() -> None:
    with pytest.raises(CorpusError, match="yielded only"):
        _items(per_register=40)


@pytest.mark.parametrize(
    ("field", "value"),
    [("prefix_length", 0), ("continuation_length", -2), ("per_register", True)],
)
def test_a_length_that_cannot_mean_anything_is_refused(field: str, value: object) -> None:
    with pytest.raises(CorpusError, match="positive integer"):
        _items(**{field: value})


def test_an_unknown_distractor_kind_is_refused_by_name() -> None:
    with pytest.raises(CorpusError, match="unknown same-region distractor"):
        _items(kinds=("prior_span", "adjacent_whole_document"))


def test_a_short_document_yields_no_items_rather_than_a_short_distractor() -> None:
    short = (
        TextUnit(
            identifier="prose-w9/part-0",
            text=_text(900, 3),
            group="prose-w9",
        ),
    )

    with pytest.raises(CorpusError, match="yielded only 0"):
        build_context_items(
            short,
            _VOCABULARY,
            prefix_length=8,
            continuation_length=4,
            per_register=1,
            registers=("prose",),
        )


def test_an_item_never_borrows_a_distractor_span_it_cannot_fill() -> None:
    """Every kind must be a full ``continuation_length`` span; a partial one would be an easier neighbour."""
    items = _items(continuation_length=4)

    for item in items:
        for kind, _, span in item.distractors:
            assert len(span) == 4, kind


# ------------------------------------------------------------------ scoring


def test_a_model_that_ignores_its_passage_scores_exactly_chance_on_every_kind() -> None:
    items = _items()

    def ignores_prefix(_prefix, continuation):  # type: ignore[no-untyped-def]
        return float(len(continuation))

    scores = score_context_items(items, ignores_prefix)

    for kind in CONTEXT_DISTRACTOR_KINDS:
        assert scores.pmi[kind].accuracy == 0.5, kind
        assert scores.pmi[kind].ties == len(items), kind


def test_a_model_that_prefers_the_true_continuation_under_the_passage_is_perfect() -> None:
    items = _items()
    truth = {item.true_continuation for item in items}

    def uses_context(prefix, continuation):  # type: ignore[no-untyped-def]
        return 0.0 if prefix and continuation in truth else -5.0

    scores = score_context_items(items, uses_context)

    for kind in CONTEXT_DISTRACTOR_KINDS:
        assert scores.pmi[kind].accuracy == 1.0, kind


def test_the_subtraction_is_what_stops_an_order_sensor_from_passing() -> None:
    """A scorer that can tell arranged English from permuted English, and knows nothing about the passage.

    It answers every conditional reading perfectly, because it never looks at the prefix and the truth is the
    only contiguous span in the pair. Under pointwise mutual information it scores exactly one half: the same
    preference is present with the passage and without it, so it cannot be use of the passage. This is the
    construction's whole claim, tested against the counter-case it was built for.
    """
    items = _items()
    truth = {item.true_continuation for item in items}

    def sensor(_prefix, continuation):  # type: ignore[no-untyped-def]
        return 0.0 if continuation in truth else -1.0

    scores = score_context_items(items, sensor)

    assert scores.conditional["permuted"].accuracy == 1.0
    assert scores.conditional["reversed"].accuracy == 1.0
    assert scores.pmi["permuted"].accuracy == 0.5
    assert scores.pmi["reversed"].accuracy == 0.5
    # The adjacent spans are real text of the same document, so an order sensor is at chance on those too.
    assert scores.pmi["prior_span"].accuracy == 0.5
    assert scores.pmi["next_span"].accuracy == 0.5


def test_scoring_keeps_the_per_item_record_and_splits_it_by_register() -> None:
    items = _items()
    truth = {item.true_continuation for item in items}

    def uses_context(prefix, continuation):  # type: ignore[no-untyped-def]
        return 0.0 if prefix and continuation in truth else -5.0

    scores = score_context_items(items, uses_context)

    assert set(scores.outcomes) == {f"pmi_{kind}" for kind in CONTEXT_DISTRACTOR_KINDS}
    for kind, pairs in scores.outcomes.items():
        assert len(pairs) == len(items), kind
        assert {identifier for identifier, _ in pairs} == {item.identifier for item in items}, kind
    assert set(scores.by_register) == {"prose", "science"}
    assert scores.by_register["prose"]["pmi_next_span"]["accuracy"] == 1.0
    assert scores.as_dict()["items"] == len(items)


def test_zero_items_is_a_refusal() -> None:
    with pytest.raises(CorpusError, match="zero items"):
        score_context_items((), lambda _prefix, _continuation: 0.0)
