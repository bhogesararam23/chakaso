"""Tests for the answer store.

The store is where "history, not a mutable cache" becomes enforceable, so the tests hammer the
invariants: nothing is overwritten, a correction cannot dangle or reach across conversations, the
chain walks deterministically oldest-first, branching is allowed, and an unknown answer has no
history to fabricate. The in-memory store must also satisfy the ``AnswerStore`` protocol, because
the point of the boundary is that a future backend is a drop-in.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.answers import (
    AnswerId,
    AnswerRecord,
    AnswerStore,
    AnswerStoreError,
    InMemoryAnswerStore,
    new_answer_id,
)

NOW = datetime(2026, 10, 8, tzinfo=UTC)


def make(*, conversation_id: str = "conv_1", correction_of: AnswerId | None = None) -> AnswerRecord:
    return AnswerRecord(
        answer_id=new_answer_id(),
        conversation_id=conversation_id,
        text="An answer.",
        created_at=NOW,
        model_id="deterministic-double",
        model_is_development_double=True,
        correction_of=correction_of,
    )


def test_save_and_get_roundtrip_and_missing_is_none() -> None:
    store = InMemoryAnswerStore()
    record = make()

    store.save(record)

    assert store.get(record.answer_id) == record
    assert store.get(new_answer_id()) is None
    assert len(store) == 1


def test_a_saved_answer_is_never_overwritten() -> None:
    store = InMemoryAnswerStore()
    record = make()
    store.save(record)

    with pytest.raises(AnswerStoreError, match="append-only"):
        store.save(record)


def test_list_is_in_insertion_order_and_filters_by_conversation() -> None:
    store = InMemoryAnswerStore()
    first = make()
    second = make()
    other = make(conversation_id="conv_2")
    for record in (first, second, other):
        store.save(record)

    assert store.list() == (first, second, other)
    assert store.list(conversation_id="conv_1") == (first, second)


def test_history_walks_a_correction_chain_oldest_first() -> None:
    store = InMemoryAnswerStore()
    original = make()
    revised = make(correction_of=original.answer_id)
    rerevised = make(correction_of=revised.answer_id)
    for record in (original, revised, rerevised):
        store.save(record)

    assert store.history(original.answer_id) == (original,)
    assert store.history(rerevised.answer_id) == (original, revised, rerevised)


def test_branching_corrections_are_allowed() -> None:
    store = InMemoryAnswerStore()
    parent = make()
    left = make(correction_of=parent.answer_id)
    right = make(correction_of=parent.answer_id)
    for record in (parent, left, right):
        store.save(record)

    assert store.history(left.answer_id) == (parent, left)
    assert store.history(right.answer_id) == (parent, right)


def test_a_dangling_correction_cannot_be_saved() -> None:
    store = InMemoryAnswerStore()
    orphan = make(correction_of=new_answer_id())

    with pytest.raises(AnswerStoreError, match="not in the store"):
        store.save(orphan)


def test_a_correction_cannot_reach_into_another_conversation() -> None:
    store = InMemoryAnswerStore()
    prior = make(conversation_id="conv_1")
    store.save(prior)
    cross = make(conversation_id="conv_2", correction_of=prior.answer_id)

    with pytest.raises(AnswerStoreError, match="conversation"):
        store.save(cross)


def test_history_of_an_unknown_answer_is_refused_not_fabricated() -> None:
    store = InMemoryAnswerStore()

    with pytest.raises(AnswerStoreError, match="no history"):
        store.history(new_answer_id())


def test_in_memory_store_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryAnswerStore(), AnswerStore)
