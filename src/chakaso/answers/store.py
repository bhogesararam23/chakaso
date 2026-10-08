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
"""

from __future__ import annotations

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
        if record.answer_id in self._records:
            message = (
                f"answer {record.answer_id} is already recorded; the history is append-only "
                "and a saved answer is never overwritten"
            )
            raise AnswerStoreError(message)

        if record.correction_of is not None:
            prior = self._records.get(record.correction_of)
            if prior is None:
                message = (
                    f"answer {record.answer_id} corrects {record.correction_of}, which is not "
                    "in the store; a correction cannot reference an answer that does not exist"
                )
                raise AnswerStoreError(message)
            if prior.conversation_id != record.conversation_id:
                message = (
                    f"answer {record.answer_id} (conversation {record.conversation_id!r}) cannot "
                    f"correct {record.correction_of} from conversation {prior.conversation_id!r}"
                )
                raise AnswerStoreError(message)

        self._records[record.answer_id] = record

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
