"""Training-set composition by share: drawing a stated fraction of a token budget from one kind of work.

E2612 and E2613 changed what the training side contained by filtering on a group prefix and capping the
stream, so the share of prose was whatever the shuffle produced (about half) or all of it. A sweep over the
share needs the share to be a number the run is told, not a consequence of a cap. :func:`select_by_share` makes
it one: the works whose group starts with a prefix supply that fraction of the identifier budget, every other
training work supplies the rest, and the realised counts are returned so a record states what was packed
rather than what was intended.

Two properties are deliberate. Each class is drawn as a *prefix of its own seeded shuffle*, so a larger share
of one class contains every document a smaller share of it contained: the arms of a sweep are nested, and a
difference between two arms is the added text rather than a different sample of it. And a share of 0 or 1
returns exactly what filtering on the prefix and shuffling returns, so the endpoints of a sweep are the same
runs the filter-based arms of E2612 already were.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.subword import SubwordVocabulary

__all__ = ["ShareSelection", "identifiers_by_register", "select_by_share", "unit_identifiers"]


def unit_identifiers(unit: TextUnit, vocabulary: SubwordVocabulary) -> int:
    """Identifiers one document contributes to a packed stream: its tokens plus the document separator."""
    return vocabulary.token_count(unit.text) + 1


@dataclass(frozen=True, slots=True)
class ShareSelection:
    """The documents a share-composed run packs, in packing order, and what they add up to."""

    units: tuple[TextUnit, ...]
    #: Identifiers per work (``TextUnit.group``), the finest grain a record can aggregate from.
    identifiers_by_group: dict[str, int]
    #: Identifiers drawn from the works that match the prefix, and from all the others.
    matched_identifiers: int
    other_identifiers: int

    @property
    def identifiers(self) -> int:
        return self.matched_identifiers + self.other_identifiers

    @property
    def realised_share(self) -> float:
        """The matched fraction actually packed; within one document of the declared share."""
        return self.matched_identifiers / self.identifiers if self.identifiers else 0.0


def select_by_share(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    *,
    prefix: str,
    share: float,
    max_tokens: int,
    seed: int,
) -> ShareSelection:
    """Choose training documents so ``share`` of ``max_tokens`` identifiers come from works under ``prefix``.

    Each class is shuffled with ``seed`` and taken from the front until it reaches its quota, so the quota is
    met to within the last document taken. The two selections are then shuffled together with the same seed so
    the registers interleave in the stream instead of arriving as two blocks — except when one class is empty
    (a share of 0 or 1), where the single class keeps the order filtering and shuffling would give it.

    Raises:
        ValueError: ``prefix`` is empty, ``share`` is outside [0, 1] or not a number, or ``max_tokens`` is not
            a positive integer.
        CorpusError: a class cannot supply its quota, which would silently turn the declared share into a
            different one.
    """
    if not isinstance(prefix, str) or not prefix:
        message = f"prefix must be a non-empty string, got {prefix!r}"
        raise ValueError(message)
    if isinstance(share, bool) or not isinstance(share, int | float) or not 0.0 <= share <= 1.0:
        message = f"share must be a number from 0 to 1, got {share!r}"
        raise ValueError(message)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        message = f"max_tokens must be an integer >= 1, got {max_tokens!r}"
        raise ValueError(message)

    matched_quota = round(share * max_tokens)
    other_quota = max_tokens - matched_quota
    matched, matched_counts = _take(
        _shuffled([unit for unit in units if unit.group.startswith(prefix)], seed),
        vocabulary,
        matched_quota,
        f"works starting with {prefix!r}",
    )
    other, other_counts = _take(
        _shuffled([unit for unit in units if not unit.group.startswith(prefix)], seed),
        vocabulary,
        other_quota,
        f"works not starting with {prefix!r}",
    )

    combined = [*matched, *other]
    if matched and other:
        random.Random(seed).shuffle(combined)
    by_group: dict[str, int] = {}
    for counts in (matched_counts, other_counts):
        for group, identifiers in counts.items():
            by_group[group] = by_group.get(group, 0) + identifiers
    return ShareSelection(
        units=tuple(combined),
        identifiers_by_group=dict(sorted(by_group.items())),
        matched_identifiers=sum(matched_counts.values()),
        other_identifiers=sum(other_counts.values()),
    )


def identifiers_by_register(
    identifiers_by_group: dict[str, int], registers: Sequence[str]
) -> dict[str, int]:
    """Collapse per-work counts to per-register counts, a work belonging to the first prefix it starts with.

    A work matching none of ``registers`` is counted under ``"other"`` rather than dropped, so the totals of
    the returned mapping always equal the totals of the input.
    """
    totals: dict[str, int] = dict.fromkeys(registers, 0)
    for group, identifiers in identifiers_by_group.items():
        key = next((register for register in registers if group.startswith(register)), "other")
        totals[key] = totals.get(key, 0) + identifiers
    return totals


def _shuffled(units: list[TextUnit], seed: int) -> list[TextUnit]:
    random.Random(seed).shuffle(units)
    return units


def _take(
    ordered: Sequence[TextUnit], vocabulary: SubwordVocabulary, quota: int, label: str
) -> tuple[list[TextUnit], dict[str, int]]:
    """The front of ``ordered`` whose identifiers first reach ``quota``; nothing when the quota is zero."""
    taken: list[TextUnit] = []
    counts: dict[str, int] = {}
    total = 0
    for unit in ordered:
        if total >= quota:
            break
        identifiers = unit_identifiers(unit, vocabulary)
        taken.append(unit)
        counts[unit.group] = counts.get(unit.group, 0) + identifiers
        total += identifiers
    if total < quota:
        message = (
            f"the {label} hold {total:,} identifiers and the declared share needs {quota:,}; "
            "a smaller share or a smaller budget fits, but the run does not quietly train on another mixture"
        )
        raise CorpusError(message)
    return taken, counts
