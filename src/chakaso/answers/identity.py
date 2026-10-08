"""Answer identity.

An ``AnswerId`` uses the same validated, prefixed shape as every other identifier in
the system (``ans_`` plus a 64-bit hexadecimal digest) so it is recognised in a log line
and checked before any lookup — but it is derived *differently*, and that difference is
the decision recorded in ADR-0017.

Evidence identifiers are content-derived (ADR-0007): the same bytes must deduplicate,
and a changed page must produce a distinguishable record. An answer is not evidence.
It is a runtime event — a model's reply at a turn — and two identical replies produced at
different moments are two answers, exactly as two sessions opening with the same question
are two conversations (``new_conversation_id``). Deriving an answer's identifier from its
text would be actively wrong here: a corrected answer that happens to repeat the original
wording would collide with the answer it supersedes, and history that cannot tell "the
first saying" from "the second, after correction" is not history. So the identifier is a
random token, while remaining a typed, validated ``Identifier``.
"""

from __future__ import annotations

import secrets
from typing import ClassVar

from chakaso.core.identifiers import Identifier

__all__ = ["ANSWER_ID_PREFIX", "AnswerId", "new_answer_id"]

#: Prefix for answer identifiers, so one is distinguishable from the evidence and
#: conversation identifiers it references.
ANSWER_ID_PREFIX = "ans_"


class AnswerId(Identifier):
    """A stable identifier for one recorded answer: ``ans_`` plus 16 hex digits."""

    PREFIX: ClassVar[str] = ANSWER_ID_PREFIX
    TYPE_NAME: ClassVar[str] = "answer identifier"


def new_answer_id() -> AnswerId:
    """Return a fresh, random answer identifier.

    Deliberately *not* content-derived (ADR-0017). A revised answer must always be
    distinguishable from the answer it supersedes, even when their text is identical,
    and a random per-event identifier guarantees that where a content hash cannot.
    """
    return AnswerId(f"{ANSWER_ID_PREFIX}{secrets.token_hex(8)}")
