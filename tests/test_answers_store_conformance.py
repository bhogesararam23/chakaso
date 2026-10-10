"""Conformance of every ``AnswerStore`` backend to the append-only contract.

ADR-0018 made persistence a protocol so a durable backend could be dropped in without changing
what the history guarantees; ADR-0021 added that backend. The claim that matters is not that the
in-memory store behaves correctly — it has always done that — but that *both* backends enforce the
same rules identically, so wiring the SQLite store into a turn cannot weaken a guarantee a caller
already relied on. Running the invariant suite over each backend (rather than once) is what makes
"drop-in" a tested fact rather than an assertion.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from chakaso.answers import (
    AnswerId,
    AnswerRecord,
    AnswerStore,
    AnswerStoreError,
    InMemoryAnswerStore,
    SQLiteAnswerStore,
    new_answer_id,
)

NOW = datetime(2026, 10, 10, tzinfo=UTC)


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


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[AnswerStore]:
    """Each backend in turn, so every assertion below runs against both."""
    if request.param == "memory":
        yield InMemoryAnswerStore()
        return
    durable = SQLiteAnswerStore(tmp_path / "answers.db")
    try:
        yield durable
    finally:
        durable.close()


def test_every_backend_satisfies_the_protocol(store: AnswerStore) -> None:
    assert isinstance(store, AnswerStore)


def test_save_and_get_roundtrip_and_missing_is_none(store: AnswerStore) -> None:
    record = make()

    store.save(record)

    assert store.get(record.answer_id) == record
    assert store.get(new_answer_id()) is None
    assert len(store) == 1


def test_a_saved_answer_is_never_overwritten(store: AnswerStore) -> None:
    record = make()
    store.save(record)

    with pytest.raises(AnswerStoreError, match="append-only"):
        store.save(record)


def test_list_is_in_insertion_order_and_filters_by_conversation(store: AnswerStore) -> None:
    first = make()
    second = make()
    other = make(conversation_id="conv_2")
    for record in (first, second, other):
        store.save(record)

    assert store.list() == (first, second, other)
    assert store.list(conversation_id="conv_1") == (first, second)


def test_history_walks_a_correction_chain_oldest_first(store: AnswerStore) -> None:
    original = make()
    revised = make(correction_of=original.answer_id)
    rerevised = make(correction_of=revised.answer_id)
    for record in (original, revised, rerevised):
        store.save(record)

    assert store.history(original.answer_id) == (original,)
    assert store.history(rerevised.answer_id) == (original, revised, rerevised)


def test_branching_corrections_are_allowed(store: AnswerStore) -> None:
    parent = make()
    left = make(correction_of=parent.answer_id)
    right = make(correction_of=parent.answer_id)
    for record in (parent, left, right):
        store.save(record)

    assert store.history(left.answer_id) == (parent, left)
    assert store.history(right.answer_id) == (parent, right)


def test_a_dangling_correction_cannot_be_saved(store: AnswerStore) -> None:
    orphan = make(correction_of=new_answer_id())

    with pytest.raises(AnswerStoreError, match="not in the store"):
        store.save(orphan)


def test_a_correction_cannot_reach_into_another_conversation(store: AnswerStore) -> None:
    prior = make(conversation_id="conv_1")
    store.save(prior)
    cross = make(conversation_id="conv_2", correction_of=prior.answer_id)

    with pytest.raises(AnswerStoreError, match="conversation"):
        store.save(cross)


def test_history_of_an_unknown_answer_is_refused_not_fabricated(store: AnswerStore) -> None:
    with pytest.raises(AnswerStoreError, match="no history"):
        store.history(new_answer_id())


def test_save_many_appends_a_whole_batch(store: AnswerStore) -> None:
    parent = make()
    child = make(correction_of=parent.answer_id)

    store.save_many((parent, child))

    assert store.history(child.answer_id) == (parent, child)
    assert len(store) == 2


def test_save_many_rejects_a_batch_that_breaks_an_invariant(store: AnswerStore) -> None:
    # A batch whose second record repeats a live identifier is refused by both backends, with
    # the durable backend leaving nothing of the first (valid) record behind — the same all-or-
    # nothing outcome the restart/rollback test checks in detail for SQLite.
    existing = make()
    store.save(existing)
    fresh = make()

    with pytest.raises(AnswerStoreError, match="append-only"):
        store.save_many((fresh, existing))

    assert store.get(fresh.answer_id) is None
    assert len(store) == 1
