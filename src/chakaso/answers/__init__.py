"""Answers: the persistent, connective record of what the system said.

An ``AnswerRecord`` is the object that ties the layers together — the conversation turn it
came from, the model that produced it, the evidence it was given and the sources it cited,
the claims extracted from it, how the supplied evidence bears on those claims, and, when a
later answer supersedes it, the link that makes the correction history traceable. It
references the evidence, claim and evaluation layers by identifier rather than duplicating
their records.

Identity is deliberate: an answer is a runtime event, so its identifier is a typed random
token, not a content hash, so that a corrected answer is always distinguishable from the one
it replaces (ADR-0017). Persistence is an append-only history behind a store boundary
(ADR-0018). There are two implementations of that boundary: an in-memory store, and — since
ADR-0021 — a durable store backed by SQLite from the standard library alone, so a correction
history can survive the process that wrote it without the project adopting a third-party or
hosted database. Both satisfy the same ``AnswerStore`` protocol and enforce the same
append-only invariants; the default backend stays in-memory, and durable storage is opt-in.
"""

from __future__ import annotations

from chakaso.answers.errors import (
    AnswerError,
    AnswerSerializationError,
    AnswerStoreError,
    InvalidAnswerError,
    StorageSchemaError,
    UnknownAnswerError,
)
from chakaso.answers.identity import ANSWER_ID_PREFIX, AnswerId, new_answer_id
from chakaso.answers.record import AnswerRecord
from chakaso.answers.sqlite import SQLiteAnswerStore
from chakaso.answers.store import AnswerStore, InMemoryAnswerStore

__all__ = [
    "ANSWER_ID_PREFIX",
    "AnswerError",
    "AnswerId",
    "AnswerRecord",
    "AnswerSerializationError",
    "AnswerStore",
    "AnswerStoreError",
    "InMemoryAnswerStore",
    "InvalidAnswerError",
    "SQLiteAnswerStore",
    "StorageSchemaError",
    "UnknownAnswerError",
    "new_answer_id",
]
