"""Detecting an explicit reassessment request, without executing one.

The correction path can already decide and record a reassessment *once it is handed* an
``AnswerId`` (:func:`chakaso.correction.service.reassess_stored_answer`). What was missing is
the front edge the runtime composes: recognising that a follow-up has *explicitly named a stored
answer* and asked for it to be reconsidered. This module is that edge and nothing more. It reads
no store, runs no evaluator, retrieves no evidence and mutates nothing, so intent detection stays
independently testable from reassessment execution (the two are deliberately separate).

The trigger is deterministic and inspectable by construction, which is what keeps P21 from being a
fake autonomous loop. It fires only when a follow-up carries a well-formed answer identifier — an
``ans_…`` reference the caller supplied, never one inferred from a vague phrase like "the previous
answer". It never guesses a target: two distinct identifiers are *ambiguous* and are reported as
such rather than resolved by a heuristic, and a follow-up that names no identifier is simply not a
reassessment. Nothing here is a model, and the classification of every message is reproducible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from chakaso.answers import AnswerId

__all__ = [
    "DeterministicReassessmentTrigger",
    "ReassessmentTriggerKind",
    "TriggerDecision",
    "detect_reassessment_request",
]

#: An explicit, bounded ``ans_`` reference: the identifier shape every answer carries
#: (ADR-0017), matched only when it is a standalone token rather than the leading
#: sixteen hex digits of a longer run. Case-sensitive, because identifiers are lowercase.
_ANSWER_REFERENCE = re.compile(r"(?<![0-9a-zA-Z_])ans_[0-9a-f]{16}(?![0-9a-f])")


class ReassessmentTriggerKind(Enum):
    """How a follow-up names a stored answer, if at all."""

    NO_TARGET = "no_target"
    """The follow-up carries no valid answer identifier: an ordinary turn, not a reassessment."""

    EXPLICIT = "explicit"
    """The follow-up names exactly one distinct answer: an unambiguous reassessment target."""

    AMBIGUOUS = "ambiguous"
    """The follow-up names two or more distinct answers: the runtime must not guess."""


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """What the trigger concluded about one follow-up, and every identifier it saw.

    ``answer_id`` is set only for :attr:`ReassessmentTriggerKind.EXPLICIT`; a decision is a value,
    so an ambiguous or empty outcome carries no half-truth about a target. ``references`` lists the
    distinct well-formed identifiers in first-seen order, so a caller can report *why* a turn was
    read the way it was rather than merely that it was.
    """

    kind: ReassessmentTriggerKind
    answer_id: AnswerId | None = None
    references: tuple[AnswerId, ...] = ()

    @property
    def triggers_reassessment(self) -> bool:
        """Whether this decision is an unambiguous request to reassess a stored answer."""
        return self.kind is ReassessmentTriggerKind.EXPLICIT

    @property
    def is_ambiguous(self) -> bool:
        """Whether the follow-up named more than one distinct answer."""
        return self.kind is ReassessmentTriggerKind.AMBIGUOUS


class DeterministicReassessmentTrigger:
    """A :class:`TriggerDecision` maker that fires only on an explicit ``ans_`` reference."""

    #: Versioned so a recorded decision names the rule that produced it, the way a query plan
    #: carries its planner version (ADR-0022) and a later trigger can be compared against it.
    VERSION: ClassVar[str] = "deterministic-reassessment-trigger-0.1"

    def detect(self, message: str) -> TriggerDecision:
        """Classify ``message`` as an explicit reassessment request.

        This never raises for ordinary content: an empty message, a message with no identifier, or
        a message with a malformed identifier are all simply :attr:`ReassessmentTriggerKind.NO_TARGET`,
        so an ambiguous-looking reference is left to be handled as an ordinary turn rather than
        guessed into a correction.
        """
        references = _distinct_references(message)
        if not references:
            return TriggerDecision(kind=ReassessmentTriggerKind.NO_TARGET)
        if len(references) == 1:
            only = references[0]
            return TriggerDecision(
                kind=ReassessmentTriggerKind.EXPLICIT, answer_id=only, references=references
            )
        return TriggerDecision(kind=ReassessmentTriggerKind.AMBIGUOUS, references=references)


def detect_reassessment_request(message: str) -> TriggerDecision:
    """Detect an explicit reassessment target in ``message`` using the deterministic rule.

    A module-level entry point over :class:`DeterministicReassessmentTrigger`, mirroring how the
    correction package exposes plain functions; a caller that wants to inject or version a different
    trigger uses the class directly.
    """
    return DeterministicReassessmentTrigger().detect(message)


def _distinct_references(message: str) -> tuple[AnswerId, ...]:
    """The distinct valid answer identifiers in ``message``, in first-seen order.

    A candidate token is only trusted once :meth:`AnswerId.parse` accepts it, so the identifier's
    own validation — not a looser local regex — is the authority on what counts as a reference
    (the same ownership rule ADR-0003 applies to evidence identifiers). Duplicate mentions of one
    identifier are not an ambiguity: naming the same answer twice is one target.
    """
    seen: dict[AnswerId, None] = {}
    for token in _ANSWER_REFERENCE.findall(message):
        try:
            answer_id = AnswerId.parse(token)
        except Exception:  # a lookalike token that is not a valid identifier names nothing
            continue
        seen.setdefault(answer_id, None)
    return tuple(seen)
