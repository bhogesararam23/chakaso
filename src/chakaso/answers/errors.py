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

__all__ = ["AnswerError", "AnswerStoreError", "InvalidAnswerError"]


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
