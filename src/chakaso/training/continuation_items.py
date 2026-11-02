"""A coherence probe built from held-out text alone: does the model use the passage it was given?

An item is a passage prefix and three candidate continuations of equal length in identifiers: the **true** next span, an **easy** distractor from a
different work of the same register, and a **hard** distractor from the same document at least ``hard_gap`` identifiers away. Everything is cut from
documents the caller passes - for a measurement, the *test side* of a whole-work split - so no sentence was authored and no contamination screen is
needed beyond the split's own.

Scoring is by pointwise mutual information with the prefix: ``log p(c | prefix) - log p(c)``, where ``p(c)`` is the same model's probability of the
continuation with no passage before it. A candidate's own predictability (frequent words, short words, a quotable phrase) cancels, so a model that
ignores its prefix scores every pair as a tie and sits at exactly one half; only use of the prefix can lift the number. The raw-conditional and the
prefix-blind accuracies are reported beside it so the cancellation can be seen to matter.

Dependency-free. The caller supplies the vocabulary (which tokenizes the documents) and a scorer ``(prefix, continuation) -> log-probability``;
:func:`chakaso.training.nn_evaluation.continuation_log_probabilities` is the binding to a checkpoint.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from chakaso.core.hashing import sha256_hex
from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.minimal_pairs import wilson_interval
from chakaso.training.subword import SubwordVocabulary

__all__ = [
    "ChoiceReading",
    "ContinuationItem",
    "ItemScores",
    "build_items",
    "items_digest",
    "score_items",
]

Scorer = Callable[[tuple[int, ...], tuple[int, ...]], float]


@dataclass(frozen=True, slots=True)
class ContinuationItem:
    """One passage prefix with its true continuation and two distractors, all in identifiers."""

    identifier: str
    register: str
    document: str
    offset: int
    prefix: tuple[int, ...]
    true_continuation: tuple[int, ...]
    easy_source: str
    easy_offset: int
    easy_distractor: tuple[int, ...]
    hard_offset: int
    hard_distractor: tuple[int, ...]


def _rank(*parts: object) -> str:
    return sha256_hex("|".join(str(part) for part in parts))


def items_digest(items: Sequence[ContinuationItem]) -> str:
    """SHA-256 over every item's content, in order: the identity a result cites."""
    canonical = json.dumps(
        [
            [
                item.identifier,
                item.register,
                item.document,
                item.offset,
                list(item.prefix),
                list(item.true_continuation),
                item.easy_source,
                item.easy_offset,
                list(item.easy_distractor),
                item.hard_offset,
                list(item.hard_distractor),
            ]
            for item in items
        ],
        separators=(",", ":"),
    )
    return sha256_hex(canonical)


def build_items(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    *,
    prefix_length: int,
    continuation_length: int,
    hard_gap: int,
    per_register: int,
    registers: Sequence[str],
    max_per_work: int = 2,
) -> tuple[ContinuationItem, ...]:
    """Build ``per_register`` items for each register prefix, deterministically, from ``units``.

    Documents are visited in SHA-256 order of their identifier, each contributes at most one item and a work at most ``max_per_work``; the offset of the
    passage, the hard distractor's position and the easy distractor's document and position are all chosen by hash, so no random generator is involved.
    Documents are tokenized lazily, only as far as they are visited.

    Raises:
        CorpusError: a register has fewer eligible passages than ``per_register``. The probe is never padded with ineligible ones.
    """
    for name, value in (
        ("prefix_length", prefix_length),
        ("continuation_length", continuation_length),
        ("hard_gap", hard_gap),
        ("per_register", per_register),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            message = f"{name} must be a positive integer, got {value!r}"
            raise CorpusError(message)
    cache: dict[str, tuple[int, ...]] = {}

    def encoded(unit: TextUnit) -> tuple[int, ...]:
        if unit.identifier not in cache:
            cache[unit.identifier] = vocabulary.encode(unit.text)
        return cache[unit.identifier]

    items: list[ContinuationItem] = []
    span = prefix_length + continuation_length
    for register in registers:
        members = sorted(
            (unit for unit in units if unit.group.startswith(register)),
            key=lambda unit: unit.identifier,
        )
        order = sorted(members, key=lambda unit: _rank("anchor", register, unit.identifier))
        taken_from_work: dict[str, int] = defaultdict(int)
        chosen = 0
        for unit in order:
            if chosen == per_register:
                break
            if taken_from_work[unit.group] >= max_per_work:
                continue
            ids = encoded(unit)
            if len(ids) < span + hard_gap + continuation_length:
                continue
            slots = list(range(0, len(ids) - span + 1, span))
            offset = min(slots, key=lambda slot: _rank("offset", unit.identifier, slot))
            anchor = offset + prefix_length
            hard_candidates = [
                position
                for position in range(0, len(ids) - continuation_length + 1, continuation_length)
                if abs(position - anchor) >= hard_gap
            ]
            if not hard_candidates:
                continue
            hard_offset = min(hard_candidates, key=lambda p: _rank("hard", unit.identifier, p))
            others = sorted(
                (
                    other
                    for other in members
                    if other.group != unit.group and len(encoded(other)) >= continuation_length
                ),
                key=lambda other: _rank("easy", unit.identifier, other.identifier),
            )
            if not others:
                continue
            source = others[0]
            source_ids = encoded(source)
            easy_slots = range(0, len(source_ids) - continuation_length + 1, continuation_length)
            easy_offset = min(easy_slots, key=lambda p: _rank("easy-offset", unit.identifier, p))
            truth = ids[anchor : anchor + continuation_length]
            easy = source_ids[easy_offset : easy_offset + continuation_length]
            hard = ids[hard_offset : hard_offset + continuation_length]
            if easy == truth or hard == truth:
                continue
            items.append(
                ContinuationItem(
                    identifier=f"{register}-{_rank('item', unit.identifier, offset)[:8]}",
                    register=register,
                    document=unit.identifier,
                    offset=offset,
                    prefix=ids[offset:anchor],
                    true_continuation=truth,
                    easy_source=source.identifier,
                    easy_offset=easy_offset,
                    easy_distractor=easy,
                    hard_offset=hard_offset,
                    hard_distractor=hard,
                )
            )
            taken_from_work[unit.group] += 1
            chosen += 1
        if chosen < per_register:
            message = (
                f"register {register!r} yielded only {chosen} eligible passage(s) of the {per_register} asked for; "
                "the probe is not padded with ineligible ones"
            )
            raise CorpusError(message)
    return tuple(items)


@dataclass(frozen=True, slots=True)
class ChoiceReading:
    """Accuracy of one way of choosing between the true continuation and one distractor."""

    items: int
    accuracy: float
    interval: tuple[float, float]
    ties: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "accuracy": self.accuracy,
            "interval_95": list(self.interval),
            "ties": self.ties,
        }


def _reading(outcomes: Sequence[float], ties: int) -> ChoiceReading:
    total = len(outcomes)
    successes = sum(outcomes)
    return ChoiceReading(
        items=total,
        accuracy=successes / total if total else 0.0,
        interval=wilson_interval(successes, total),
        ties=ties,
    )


@dataclass(frozen=True, slots=True)
class ItemScores:
    """Six readings of one scorer on one item set: PMI, raw-conditional and prefix-blind, against each distractor."""

    items: int
    pmi_easy: ChoiceReading
    pmi_hard: ChoiceReading
    conditional_easy: ChoiceReading
    conditional_hard: ChoiceReading
    blind_easy: ChoiceReading
    blind_hard: ChoiceReading
    by_register: dict[str, dict[str, dict[str, Any]]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "pmi_easy": self.pmi_easy.as_dict(),
            "pmi_hard": self.pmi_hard.as_dict(),
            "conditional_easy": self.conditional_easy.as_dict(),
            "conditional_hard": self.conditional_hard.as_dict(),
            "blind_easy": self.blind_easy.as_dict(),
            "blind_hard": self.blind_hard.as_dict(),
            "by_register": self.by_register,
        }


_READINGS = (
    "pmi_easy",
    "pmi_hard",
    "conditional_easy",
    "conditional_hard",
    "blind_easy",
    "blind_hard",
)


def score_items(items: Sequence[ContinuationItem], scorer: Scorer) -> ItemScores:
    """Score ``items`` with ``scorer(prefix, continuation)``; an empty prefix asks for the prefix-blind probability.

    A tie counts one half. Every reading is also given per register.
    """
    if not items:
        message = "cannot score zero items"
        raise CorpusError(message)
    outcomes: dict[str, dict[str, list[float]]] = {name: defaultdict(list) for name in _READINGS}
    ties: dict[str, dict[str, int]] = {name: defaultdict(int) for name in _READINGS}

    def record(name: str, register: str, good: float, bad: float) -> None:
        value = 0.5 if good == bad else (1.0 if good > bad else 0.0)
        for key in ("all", register):
            outcomes[name][key].append(value)
            ties[name][key] += good == bad

    for item in items:
        true_c = scorer(item.prefix, item.true_continuation)
        true_b = scorer((), item.true_continuation)
        for kind, distractor in (("easy", item.easy_distractor), ("hard", item.hard_distractor)):
            bad_c = scorer(item.prefix, distractor)
            bad_b = scorer((), distractor)
            record(f"pmi_{kind}", item.register, true_c - true_b, bad_c - bad_b)
            record(f"conditional_{kind}", item.register, true_c, bad_c)
            record(f"blind_{kind}", item.register, true_b, bad_b)
    readings = {name: _reading(outcomes[name]["all"], ties[name]["all"]) for name in _READINGS}
    registers = sorted({item.register for item in items})
    by_register = {
        register: {
            name: _reading(outcomes[name][register], ties[name][register]).as_dict()
            for name in _READINGS
        }
        for register in registers
    }
    return ItemScores(
        items=len(items),
        pmi_easy=readings["pmi_easy"],
        pmi_hard=readings["pmi_hard"],
        conditional_easy=readings["conditional_easy"],
        conditional_hard=readings["conditional_hard"],
        blind_easy=readings["blind_easy"],
        blind_hard=readings["blind_hard"],
        by_register=by_register,
    )
