"""Same-region continuation items: a coherence probe whose distractors share the passage's topic and style.

[E2623](../../../experiments/E2623/record.md) built the repository's second capability probe and found it
valid and **near ceiling** — six trained decoders between 0.958 and 0.986 against a span from another work, all
intervals overlapping, "it cannot rank the trained arms". Its own record names the reason and the fix: the easy
distractor differs in topic, and the hard one at least in style, so a model only has to notice a shift to pass.
This module is that fix. Its distractors are the same document's text immediately before the passage and
immediately after the continuation, plus the true continuation's own identifiers permuted and reversed. Topic,
register, hand and vocabulary are held fixed by construction; what is left to detect is that *this* span
follows *this* passage.

Scoring is the same pointwise mutual information as v1 — ``log p(c | passage) - log p(c)`` — and it matters
more here than it did there: a permuted span is far less predictable English than the real text it was built
from, so without the subtraction a reading could reward whichever model dislikes disordered text least rather
than any use of the passage.

Dependency-free like the rest of the instrument layer: the caller supplies the vocabulary and a scorer
``(prefix, continuation) -> log-probability``.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from chakaso.core.hashing import sha256_hex
from chakaso.training.continuation_items import ChoiceReading
from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.minimal_pairs import wilson_interval
from chakaso.training.subword import SubwordVocabulary

__all__ = [
    "CONTEXT_DISTRACTOR_KINDS",
    "ContextItem",
    "ContextScores",
    "build_context_items",
    "context_items_digest",
    "score_context_items",
]

Scorer = Callable[[tuple[int, ...], tuple[int, ...]], float]

#: The four ways a distractor can hold topic and style fixed and move only what the passage predicts.
CONTEXT_DISTRACTOR_KINDS = ("prior_span", "next_span", "permuted", "reversed")


def _rank(*parts: object) -> str:
    """The selection hash, defined here rather than borrowed from the v1 module's private namespace.

    One line, and the two probes are separate instruments with separate records: a change to how v1 picks its
    documents must not silently move this set's digest, and vice versa.
    """
    return sha256_hex("|".join(str(part) for part in parts))


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One passage prefix, its true continuation, and one distractor of each declared kind.

    Every distractor is exactly ``continuation_length`` identifiers long — the same as the truth — and carries
    the position in its own document it was cut from, so a reader can check that no span overlaps the passage or
    the truth.

    ``work`` is the split group the document was cut from — the unit the whole-work split held out, and therefore the
    unit a reader should treat as one observation when the verdicts are clustered (K-035). It is carried on the item
    rather than re-derived from ``document`` at analysis time, and it is deliberately *not* one of the fields
    :func:`context_items_digest` hashes: it is a function of the document, which is already in the digest, and a
    committed record's item identity has to keep recomputing.
    """

    identifier: str
    register: str
    document: str
    work: str
    offset: int
    prefix: tuple[int, ...]
    true_continuation: tuple[int, ...]
    distractors: tuple[tuple[str, int, tuple[int, ...]], ...]

    def kinds(self) -> tuple[str, ...]:
        """Which kinds this item carries, in the order they were declared."""
        return tuple(kind for kind, _, _ in self.distractors)


def context_items_digest(items: Sequence[ContextItem]) -> str:
    """SHA-256 over each item's identifier, register, document, offset and every distractor's kind, position
    and identifiers.

    The item's ``work`` is not among the hashed fields: it is a function of ``document``, which is hashed, and two
    committed records cite digests computed before the field existed. A reader who wants the clustering does not need
    the digest to carry it — the probe result's cluster block does, with its own digest.

    This is a separate digest from v1's :func:`~chakaso.training.continuation_items.items_digest` and the
    function that produced ``9557a327bfe61e2d747a826777b748ebe285b2dd352ba6a769aca0262cfacdcd`` is untouched:
    a committed record cites that identity, and an identity a record names has to keep recomputing.
    """
    canonical = json.dumps(
        [
            [
                item.identifier,
                item.register,
                item.document,
                item.offset,
                list(item.prefix),
                list(item.true_continuation),
                [[kind, position, list(span)] for kind, position, span in item.distractors],
            ]
            for item in items
        ],
        separators=(",", ":"),
    )
    return sha256_hex(canonical)


def _permutation(values: tuple[int, ...], seed: str) -> tuple[int, ...]:
    """A deterministic reordering of ``values`` that is never the identity.

    Seeded from the item's own identifier and offset, so rebuilding the set reproduces every permutation and
    therefore the digest. A shuffle that happened to return the input falls back to a rotation: a distractor
    equal to the truth is not a comparison.
    """
    order = list(range(len(values)))
    random.Random(seed).shuffle(order)
    shuffled = tuple(values[index] for index in order)
    if shuffled == values:
        shuffled = values[1:] + values[:1]
    return shuffled


def build_context_items(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    *,
    prefix_length: int,
    continuation_length: int,
    per_register: int,
    registers: Sequence[str],
    max_per_work: int = 2,
    kinds: Sequence[str] = CONTEXT_DISTRACTOR_KINDS,
) -> tuple[ContextItem, ...]:
    """Build ``per_register`` same-region items for each register, deterministically, from ``units``.

    Selection follows E2623's rule exactly — documents in hash order, at most one item per document and
    ``max_per_work`` per work, the anchor chosen by hash among spans stepping by the passage length — so the two
    item sets sample the corpus alike and a difference between their scores is a difference in the task rather
    than in the documents. An anchor is eligible only when the document also holds the span *before* the
    passage and the span *after* the continuation: every distractor must be real text of that document, not a
    short neighbour that would be easier for its own length.

    Raises:
        CorpusError: a length is not a positive integer, a kind is unknown, or a register has fewer eligible
            passages than asked for. The probe is never padded with ineligible ones.
    """
    for name, value in (
        ("prefix_length", prefix_length),
        ("continuation_length", continuation_length),
        ("per_register", per_register),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            message = f"{name} must be a positive integer, got {value!r}"
            raise CorpusError(message)
    unknown = sorted(set(kinds) - set(CONTEXT_DISTRACTOR_KINDS))
    if unknown:
        message = (
            f"unknown same-region distractor(s) {unknown}; known kinds are "
            f"{list(CONTEXT_DISTRACTOR_KINDS)}"
        )
        raise CorpusError(message)
    cache: dict[str, tuple[int, ...]] = {}

    def encoded(unit: TextUnit) -> tuple[int, ...]:
        if unit.identifier not in cache:
            cache[unit.identifier] = vocabulary.encode(unit.text)
        return cache[unit.identifier]

    span = prefix_length + continuation_length
    items: list[ContextItem] = []
    for register in registers:
        order = sorted(
            (unit for unit in units if unit.group.startswith(register)),
            key=lambda unit: _rank("anchor", register, unit.identifier),
        )
        taken_from_work: dict[str, int] = defaultdict(int)
        chosen = 0
        for unit in order:
            if chosen == per_register:
                break
            if taken_from_work[unit.group] >= max_per_work:
                continue
            ids = encoded(unit)
            last_anchor = len(ids) - span - continuation_length + 1
            slots = list(range(continuation_length, last_anchor, span))
            if not slots:
                continue
            offset = min(slots, key=lambda slot: _rank("offset", unit.identifier, slot))
            anchor = offset + prefix_length
            truth = ids[anchor : anchor + continuation_length]
            produced: list[tuple[str, int, tuple[int, ...]]] = []
            for kind in kinds:
                if kind == "prior_span":
                    position = offset - continuation_length
                    candidate = ids[position:offset]
                elif kind == "next_span":
                    position = anchor + continuation_length
                    candidate = ids[position : position + continuation_length]
                elif kind == "permuted":
                    position = anchor
                    candidate = _permutation(truth, f"{unit.identifier}:{offset}:permuted")
                else:
                    position = anchor
                    candidate = truth[::-1]
                if len(candidate) != continuation_length or candidate == truth:
                    continue
                produced.append((kind, position, tuple(candidate)))
            if len(produced) != len(kinds):
                # A document that cannot carry every kind is left out entirely: a set whose comparisons are
                # declared once for the whole set cannot be built from items with different neighbours.
                continue
            items.append(
                ContextItem(
                    identifier=f"ctx-{register}-{_rank('item', unit.identifier, offset)[:8]}",
                    register=register,
                    document=unit.identifier,
                    work=unit.group,
                    offset=offset,
                    prefix=ids[offset:anchor],
                    true_continuation=truth,
                    distractors=tuple(produced),
                )
            )
            taken_from_work[unit.group] += 1
            chosen += 1
        if chosen < per_register:
            message = (
                f"register {register!r} yielded only {chosen} eligible passage(s) of the {per_register} "
                "asked for; the probe is not padded with ineligible ones"
            )
            raise CorpusError(message)
    return tuple(items)


@dataclass(frozen=True, slots=True)
class ContextScores:
    """PMI and raw-conditional accuracy against each kind, the per-register split and the per-item record."""

    items: int
    pmi: dict[str, ChoiceReading]
    conditional: dict[str, ChoiceReading]
    by_register: dict[str, dict[str, dict[str, Any]]]
    outcomes: dict[str, tuple[tuple[str, float], ...]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "pmi": {kind: reading.as_dict() for kind, reading in self.pmi.items()},
            "conditional": {kind: reading.as_dict() for kind, reading in self.conditional.items()},
            "by_register": self.by_register,
            "outcomes": {
                name: [[identifier, value] for identifier, value in pairs]
                for name, pairs in self.outcomes.items()
            },
        }


def _reading(values: Sequence[float], ties: int) -> ChoiceReading:
    total = len(values)
    successes = sum(values)
    return ChoiceReading(
        items=total,
        accuracy=successes / total if total else 0.0,
        interval=wilson_interval(successes, total),
        ties=ties,
    )


def score_context_items(items: Sequence[ContextItem], scorer: Scorer) -> ContextScores:
    """Every kind's PMI and raw-conditional accuracy, per item as well as in aggregate.

    A model that ignores its passage answers every pair alike and lands on one half for every kind: the
    subtraction removes each candidate's own predictability, which here ranges from ordinary prose to a
    permutation of it. The per-item record is kept because the paired comparison two models on one item set
    need it (:mod:`chakaso.training.paired_stats`), and a mean would throw the discordant items away.
    """
    if not items:
        message = "cannot score zero items"
        raise CorpusError(message)
    kinds = [kind for kind in CONTEXT_DISTRACTOR_KINDS if kind in items[0].kinds()]
    pmi_values: dict[str, dict[str, list[float]]] = {kind: defaultdict(list) for kind in kinds}
    conditional_values: dict[str, dict[str, list[float]]] = {
        kind: defaultdict(list) for kind in kinds
    }
    pmi_ties: dict[str, dict[str, int]] = {kind: defaultdict(int) for kind in kinds}
    conditional_ties: dict[str, dict[str, int]] = {kind: defaultdict(int) for kind in kinds}
    per_item: dict[str, list[tuple[str, float]]] = {f"pmi_{kind}": [] for kind in kinds}

    def outcome(good: float, bad: float) -> float:
        return 0.5 if good == bad else (1.0 if good > bad else 0.0)

    for item in items:
        truth_with = scorer(item.prefix, item.true_continuation)
        truth_without = scorer((), item.true_continuation)
        for kind, _, candidate in item.distractors:
            bad_with = scorer(item.prefix, candidate)
            bad_without = scorer((), candidate)
            pmi = outcome(truth_with - truth_without, bad_with - bad_without)
            conditional = outcome(truth_with, bad_with)
            for key in ("all", item.register):
                pmi_values[kind][key].append(pmi)
                conditional_values[kind][key].append(conditional)
                pmi_ties[kind][key] += pmi == 0.5
                conditional_ties[kind][key] += conditional == 0.5
            per_item[f"pmi_{kind}"].append((item.identifier, pmi))
    registers = sorted({item.register for item in items})
    return ContextScores(
        items=len(items),
        pmi={kind: _reading(pmi_values[kind]["all"], pmi_ties[kind]["all"]) for kind in kinds},
        conditional={
            kind: _reading(conditional_values[kind]["all"], conditional_ties[kind]["all"])
            for kind in kinds
        },
        by_register={
            register: {
                f"pmi_{kind}": _reading(
                    pmi_values[kind][register], pmi_ties[kind][register]
                ).as_dict()
                for kind in kinds
            }
            for register in registers
        },
        outcomes={name: tuple(pairs) for name, pairs in per_item.items()},
    )
