"""The answer store: an append-only history of recorded answers.

Persistence is its own boundary (ADR-0018): the conversation manager *coordinates* saving and
reading answer records, but the storage mechanism lives here and behind a protocol, so a future
backend is an implementation change rather than a rewrite. The only implementation today is
in-memory — deliberately, because committing to SQLite, Postgres or a file format before there is
a workload that needs one is a guess the project's rules forbid (the same reasoning that keeps
conversation state in memory and the retrieval corpus un-persisted).

The store enforces the invariants a *history* depends on and that a mutable cache would not:

* an answer is never overwritten — saving a second answer under a live identifier is an error,
  not an update;
* a correction can only point at an answer that already exists (no dangling links);
* a correction cannot reach into another conversation;
* because every correction points back at something already saved, the graph is acyclic by
  construction, and ``history`` walks it deterministically from the original to the answer given.

Nothing here silently repairs a broken reference. An impossible state is a thing to see, not to
paper over — a corrupted research history that looks clean is worse than one that fails loudly.

These guarantees are a property of *history*, not of one backend, so the checks live in
:func:`_ensure_appendable` and are shared by every implementation. A durable store (ADR-0021)
must reject the same overwrite, dangling-correction and cross-conversation writes the in-memory
store does; routing both through one validator is what keeps that promise from drifting as a
second backend is added.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol, runtime_checkable

from chakaso.answers.errors import AnswerStoreError
from chakaso.answers.identity import AnswerId
from chakaso.answers.record import AnswerRecord

__all__ = ["AnswerStore", "InMemoryAnswerStore"]


@runtime_checkable
class AnswerStore(Protocol):
    """The persistence boundary for answer records."""

    def save(self, record: AnswerRecord) -> None:
        """Append ``record`` to the history.

        Raises:
            AnswerStoreError: the identifier is already present, or the record's
                ``correction_of`` references an answer that does not exist or belongs to
                another conversation.
        """
        ...

    def save_many(self, records: Iterable[AnswerRecord]) -> None:
        """Append ``records`` atomically: either all persist or none do.

        Raises:
            AnswerStoreError: any record repeats a live identifier or forms a correction
                that dangles or reaches across conversations. A rejected batch leaves the
                history exactly as it was.
        """
        ...

    def get(self, answer_id: AnswerId) -> AnswerRecord | None:
        """The recorded answer with ``answer_id``, or ``None`` if there is none."""
        ...

    def list(self, *, conversation_id: str | None = None) -> tuple[AnswerRecord, ...]:
        """Every recorded answer, in the order it was saved.

        A conversation filter narrows the result; ordering is by insertion, which is the order
        the history was written, so traversal is deterministic and does not depend on a random
        identifier's byte order.
        """
        ...

    def history(self, answer_id: AnswerId) -> tuple[AnswerRecord, ...]:
        """The correction chain ending at ``answer_id``, oldest first.

        Raises:
            AnswerStoreError: ``answer_id`` is not in the store. A history cannot be produced
                for an answer that was never recorded.
        """
        ...


class InMemoryAnswerStore:
    """An ``AnswerStore`` holding records in insertion-ordered memory."""

    def __init__(self) -> None:
        self._records: dict[AnswerId, AnswerRecord] = {}

    def save(self, record: AnswerRecord) -> None:
        _ensure_appendable(record, self._records)
        self._records[record.answer_id] = record

    def save_many(self, records: Iterable[AnswerRecord]) -> None:
        # Atomic in the observable sense: validate the whole batch against the live history
        # plus everything earlier in this batch, and only then merge. If any record breaks an
        # invariant the loop raises before a single mutation, so the store is left untouched —
        # mirroring a durable store's rollback rather than its partial-write hazard.
        pending: dict[AnswerId, AnswerRecord] = dict(self._records)
        batch: dict[AnswerId, AnswerRecord] = {}
        for record in records:
            _ensure_appendable(record, pending)
            pending[record.answer_id] = record
            batch[record.answer_id] = record
        self._records.update(batch)

    def get(self, answer_id: AnswerId) -> AnswerRecord | None:
        return self._records.get(answer_id)

    def list(self, *, conversation_id: str | None = None) -> tuple[AnswerRecord, ...]:
        records = tuple(self._records.values())
        if conversation_id is not None:
            records = tuple(
                record for record in records if record.conversation_id == conversation_id
            )
        return records

    def history(self, answer_id: AnswerId) -> tuple[AnswerRecord, ...]:
        record = self._records.get(answer_id)
        if record is None:
            message = f"no answer {answer_id} is recorded; there is no history to walk"
            raise AnswerStoreError(message)

        chain = [record]
        seen = {record.answer_id}
        current = record
        while current.correction_of is not None:
            parent = self._records.get(current.correction_of)
            if parent is None:
                message = (
                    f"answer {current.answer_id} references a missing predecessor "
                    f"{current.correction_of}; the stored history is inconsistent"
                )
                raise AnswerStoreError(message)
            if parent.answer_id in seen:
                break  # defensive: a cycle cannot be created through save(), but never loop forever
            chain.append(parent)
            seen.add(parent.answer_id)
            current = parent
        return tuple(reversed(chain))

    def __len__(self) -> int:
        """How many answers are recorded."""
        return len(self._records)


def _ensure_appendable(record: AnswerRecord, existing: Mapping[AnswerId, AnswerRecord]) -> None:
    """Reject a ``record`` that would break the append-only invariants against ``existing``.

    The three checks are the whole contract of a history — no overwrite, no dangling
    correction, no cross-conversation correction — and are shared by every ``AnswerStore``
    backend so a durable store cannot quietly weaken one of them. ``existing`` is the state
    the record would be added to; a batch validator passes the live history plus earlier
    items in the same batch so within-batch corrections resolve.

    The error wording comes from the shared message helpers below rather than being written
    inline, so a durable store that reaches the same conclusion through ``SELECT``s raises a
    record-indistinguishable error and the two backends cannot drift apart in what they say.

    Raises:
        AnswerStoreError: the record repeats a live identifier, corrects an answer that is
            not present, or corrects an answer in another conversation.
    """
    if record.answer_id in existing:
        raise _overwrite_error(record)

    if record.correction_of is not None:
        prior = existing.get(record.correction_of)
        if prior is None:
            raise _dangling_error(record)
        if prior.conversation_id != record.conversation_id:
            raise _cross_conversation_error(record, prior.conversation_id)


def _overwrite_error(record: AnswerRecord) -> AnswerStoreError:
    """The error for saving a second answer under a live identifier."""
    message = (
        f"answer {record.answer_id} is already recorded; the history is append-only "
        "and a saved answer is never overwritten"
    )
    return AnswerStoreError(message)


def _dangling_error(record: AnswerRecord) -> AnswerStoreError:
    """The error for a correction that references an answer not in the store."""
    message = (
        f"answer {record.answer_id} corrects {record.correction_of}, which is not "
        "in the store; a correction cannot reference an answer that does not exist"
    )
    return AnswerStoreError(message)


def _cross_conversation_error(record: AnswerRecord, prior_conversation_id: str) -> AnswerStoreError:
    """The error for a correction that reaches into another conversation."""
    message = (
        f"answer {record.answer_id} (conversation {record.conversation_id!r}) cannot "
        f"correct {record.correction_of} from conversation {prior_conversation_id!r}"
    )
    return AnswerStoreError(message)
