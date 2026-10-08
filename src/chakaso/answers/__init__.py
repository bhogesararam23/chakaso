"""Answers: the persistent, connective record of what the system said.

An ``AnswerRecord`` is the object that ties the layers together — the conversation turn it
came from, the model that produced it, the evidence it was given and the sources it cited,
the claims extracted from it, how the supplied evidence bears on those claims, and, when a
later answer supersedes it, the link that makes the correction history traceable. It
references the evidence, claim and evaluation layers by identifier rather than duplicating
their records.

Identity is deliberate: an answer is a runtime event, so its identifier is a typed random
token, not a content hash, so that a corrected answer is always distinguishable from the one
it replaces (ADR-0017). Persistence is an append-only history through a store boundary
(ADR-0018); the current implementation is in-memory, and the abstraction is what makes a
future backend possible without a database being introduced early.
"""

from __future__ import annotations

from chakaso.answers.errors import AnswerError, AnswerStoreError, InvalidAnswerError
from chakaso.answers.identity import ANSWER_ID_PREFIX, AnswerId, new_answer_id

__all__ = [
    "ANSWER_ID_PREFIX",
    "AnswerError",
    "AnswerId",
    "AnswerStoreError",
    "InvalidAnswerError",
    "new_answer_id",
]
