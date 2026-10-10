"""Errors raised by the answer layer.

An answer can be rejected because it is internally inconsistent (no text, a citation
it was not given, a self-referential correction) or because a store operation would
break history (overwriting a saved answer, linking to one that does not exist, or
pointing at another conversation). These are different failures with different fixes,
so they are different types, and both derive from ``ValueError`` so that a caller
validating untrusted answer input can catch the built-in it already expects.
"""

from __future__ import annotations

from chakaso.core.errors import ChakasoError

__all__ = [
    "AnswerError",
    "AnswerSerializationError",
    "AnswerStoreError",
    "InvalidAnswerError",
    "StorageSchemaError",
    "UnknownAnswerError",
]


class AnswerError(ChakasoError, ValueError):
    """Base class for every failure the answer layer raises deliberately."""


class InvalidAnswerError(AnswerError):
    """An ``AnswerRecord`` cannot be constructed — it is internally inconsistent.

    The answer names no conversation, has no text, cites evidence it was not given, or
    claims a correction relationship that cannot hold. Because an answer record is the
    unit history and evaluation are built on, an inconsistent one is not a nuisance; it
    is a record that could not later be cited, corrected or superseded unambiguously.
    """


class AnswerStoreError(AnswerError):
    """A store operation would corrupt the append-only answer history.

    Saving a second answer under an identifier already present, correcting an answer
    that is not in the store, or forming a correction chain that references a different
    conversation — each is refused here rather than silently repaired, because rewriting
    research history to make an operation succeed destroys the evidence the history
    exists to preserve.
    """


class UnknownAnswerError(AnswerError):
    """An operation named an answer that is not recorded.

    Reassessing or linking to a missing answer is refused rather than treated as a
    no-op: silently succeeding would let a correction target that never existed pass for
    one that did, which is exactly the corrupted history the layer exists to prevent.
    """


class AnswerSerializationError(AnswerError):
    """A stored answer document is malformed or inconsistent and cannot be read back.

    A durable record that does not match its own schema — a missing field, a non-string
    identifier, unparseable JSON — is reported rather than guessed at or silently
    repaired, because a corrupted research history that loads "successfully" into a wrong
    record is worse than one that refuses to load.
    """


class StorageSchemaError(AnswerError):
    """A durable store's schema does not match what this code can safely use.

    A database from a newer version, or one older than the migrations here can bring
    forward, is refused rather than opened and mutated: silently writing an unknown or
    partially-migrated schema into research history is unrecoverable, so the store stops
    and says which version it found and which it expects.
    """
