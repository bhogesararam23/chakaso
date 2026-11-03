"""A likelihood-comparison probe of learned grammatical knowledge: does the model prefer the grammatical sentence?

A minimal pair is two sentences that differ in one grammatical respect, one acceptable and one not. A language model
"passes" a pair when it assigns the acceptable sentence the higher probability. That asks nothing of generation, needs no
prompt format and no judge, and has a chance level of exactly one half, which is what makes a number from it
interpretable at a scale where generated text is not (the BLiMP benchmark of Warstadt et al., arXiv:1912.00582, is the
established precedent; this module is a re-implementation of the idea over this project's own pairs, not of that data).

The module is deliberately small and takes the model as a function ``text -> log-probability``:

* it is dependency-free, so the scoring rule and its guards are tested in every CI tier;
* a pair set carries a name, version, licence, provenance and a digest of its pairs, and refuses to load when they disagree;
* the result reports accuracy per phenomenon with a Wilson interval, and **a length-only reading** (the accuracy of
  "prefer the shorter sentence"), because a likelihood comparison between sentences of different length can be won by
  length alone;
* :func:`contaminated_pairs` flags pairs whose acceptable sentence shares a word n-gram with a given training text.

Importing it needs no numerical library; binding it to a trained checkpoint is
:func:`chakaso.training.nn_evaluation.sequence_log_probabilities`.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chakaso.core.hashing import sha256_hex
from chakaso.training.errors import CorpusError

__all__ = [
    "PAIR_SET_SCHEMA",
    "MinimalPair",
    "PairResult",
    "PairSet",
    "contaminated_pairs",
    "load_pair_set",
    "score_pairs",
    "wilson_interval",
]

PAIR_SET_SCHEMA = "chakaso-minimal-pairs/1"
_Z95 = 1.959964
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class MinimalPair:
    """One acceptable sentence and one unacceptable sentence that differ in one grammatical respect."""

    identifier: str
    phenomenon: str
    good: str
    bad: str

    def __post_init__(self) -> None:
        for name in ("identifier", "phenomenon", "good", "bad"):
            if not getattr(self, name).strip():
                message = f"a minimal pair's {name} must not be empty"
                raise CorpusError(message)
        if self.good == self.bad:
            message = (
                f"minimal pair {self.identifier!r} has identical sentences, so it tests nothing"
            )
            raise CorpusError(message)

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.identifier,
            "phenomenon": self.phenomenon,
            "good": self.good,
            "bad": self.bad,
        }


@dataclass(frozen=True, slots=True)
class PairSet:
    """A named, versioned collection of pairs with its provenance and a digest that binds them."""

    name: str
    version: str
    licence: str
    provenance: str
    pairs: tuple[MinimalPair, ...]

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "licence", "provenance"):
            if not getattr(self, field_name).strip():
                message = f"a pair set's {field_name} must not be empty"
                raise CorpusError(message)
        if not self.pairs:
            message = f"pair set {self.name!r} has no pairs"
            raise CorpusError(message)
        identifiers = [pair.identifier for pair in self.pairs]
        if len(set(identifiers)) != len(identifiers):
            repeated = sorted({item for item in identifiers if identifiers.count(item) > 1})
            message = f"pair set {self.name!r} lists a pair identifier twice: {repeated}"
            raise CorpusError(message)

    @property
    def digest(self) -> str:
        """SHA-256 over the pairs in order: the identity a result cites."""
        canonical = json.dumps(
            [pair.as_dict() for pair in self.pairs],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return sha256_hex(canonical)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PAIR_SET_SCHEMA,
            "name": self.name,
            "version": self.version,
            "licence": self.licence,
            "provenance": self.provenance,
            "digest": self.digest,
            "pairs": [pair.as_dict() for pair in self.pairs],
        }


def load_pair_set(path: Path) -> PairSet:
    """Read a pair set, refusing a schema this code does not know or pairs that no longer match their digest.

    Raises:
        CorpusError: the file is not valid, names another schema, or its stored digest disagrees with its pairs.
    """
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("schema_version") != PAIR_SET_SCHEMA:
            message = f"{str(path)!r} declares schema {document.get('schema_version')!r}, not {PAIR_SET_SCHEMA!r}"
            raise CorpusError(message)
        pairs = tuple(
            MinimalPair(
                identifier=str(item["id"]),
                phenomenon=str(item["phenomenon"]),
                good=str(item["good"]),
                bad=str(item["bad"]),
            )
            for item in document["pairs"]
        )
        pair_set = PairSet(
            name=str(document["name"]),
            version=str(document["version"]),
            licence=str(document["licence"]),
            provenance=str(document["provenance"]),
            pairs=pairs,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        message = f"cannot read the pair set at {str(path)!r}: {error}"
        raise CorpusError(message) from error
    if document.get("digest") != pair_set.digest:
        message = (
            f"the digest stored in {str(path)!r} ({document.get('digest')!r}) does not match its pairs "
            f"({pair_set.digest!r}); the file was edited after it was written"
        )
        raise CorpusError(message)
    return pair_set


def wilson_interval(successes: float, trials: int) -> tuple[float, float]:
    """The 95 % Wilson score interval for a proportion; ``(0, 1)`` when there are no trials."""
    if trials <= 0:
        return 0.0, 1.0
    proportion = successes / trials
    z2 = _Z95**2
    denominator = 1.0 + z2 / trials
    centre = (proportion + z2 / (2 * trials)) / denominator
    half = (
        _Z95
        * math.sqrt(proportion * (1 - proportion) / trials + z2 / (4 * trials**2))
        / denominator
    )
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True, slots=True)
class PairResult:
    """Accuracy of one scorer on one pair set, with what is needed to read it honestly."""

    pairs: int
    accuracy: float
    interval: tuple[float, float]
    ties: int
    by_phenomenon: dict[str, float]
    intervals_by_phenomenon: dict[str, tuple[float, float]]
    pairs_by_phenomenon: dict[str, int]
    length_only_accuracy: float
    good_longer: int
    good_shorter: int
    #: The accuracy of "prefer the sentence with fewer identifiers", when a counter was given; ``None`` otherwise.
    token_count_only_accuracy: float | None = None
    good_fewer_tokens: int = 0
    good_more_tokens: int = 0
    #: ``(identifier, phenomenon, value)`` for every pair, in the order it was scored. The aggregate accuracy is
    #: the wrong object for comparing two models on the same pairs — what carries the difference is the items
    #: they answered *unlike*, which is what McNemar's test counts (see :mod:`chakaso.training.paired_stats`).
    #: E2621-E2623 could not run that test because the per-item record was dropped at scoring time.
    outcomes: tuple[tuple[str, str, float], ...] = ()

    @property
    def chance_inside_interval(self) -> bool:
        """Whether chance (one half) cannot be ruled out at 95 %."""
        return self.interval[0] <= 0.5 <= self.interval[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "pairs": self.pairs,
            "accuracy": self.accuracy,
            "interval_95": list(self.interval),
            "chance_inside_interval": self.chance_inside_interval,
            "ties": self.ties,
            "by_phenomenon": {
                name: {
                    "accuracy": self.by_phenomenon[name],
                    "interval_95": list(self.intervals_by_phenomenon[name]),
                    "pairs": self.pairs_by_phenomenon[name],
                }
                for name in sorted(self.by_phenomenon)
            },
            "length_only_accuracy": self.length_only_accuracy,
            "good_longer": self.good_longer,
            "good_shorter": self.good_shorter,
            "token_count_only_accuracy": self.token_count_only_accuracy,
            "good_fewer_tokens": self.good_fewer_tokens,
            "good_more_tokens": self.good_more_tokens,
            # Every pair, in score order, so a second model's result on the same set can be tested against this
            # one item by item rather than through two marginal intervals.
            "outcomes": [[name, phenomenon, value] for name, phenomenon, value in self.outcomes],
        }


def score_pairs(
    pairs: Sequence[MinimalPair],
    log_probability: Callable[[str], float],
    *,
    token_count: Callable[[str], int] | None = None,
) -> PairResult:
    """Score ``pairs`` with ``log_probability``: a pair is passed when the good sentence scores higher; a tie is half.

    ``log_probability`` is called once per sentence. The length-only reading is the accuracy of the rule "prefer the
    shorter sentence" (a tie is half), independent of the scorer; a scorer whose accuracy is close to it has not been
    shown to know more than lengths. With ``token_count`` the same is reported for "prefer the sentence with fewer
    identifiers": a summed log-probability falls with every identifier, so an untrained model can pass pairs whose
    bad sentence tokenizes into more of them (E2621 found this on 0.95 of one phenomenon), and a character count
    cannot see it.
    """
    if not pairs:
        message = "cannot score zero pairs"
        raise CorpusError(message)
    credit: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    ties = 0
    length_credit = 0.0
    longer = shorter = 0
    token_credit = 0.0
    fewer = more = 0
    seen: list[tuple[str, str, float]] = []
    for pair in pairs:
        good, bad = log_probability(pair.good), log_probability(pair.bad)
        if good == bad:
            ties += 1
            value = 0.5
        else:
            value = 1.0 if good > bad else 0.0
        credit[pair.phenomenon] += value
        counts[pair.phenomenon] += 1
        seen.append((pair.identifier, pair.phenomenon, value))
        if token_count is not None:
            good_tokens, bad_tokens = token_count(pair.good), token_count(pair.bad)
            if good_tokens < bad_tokens:
                token_credit += 1.0
                fewer += 1
            elif good_tokens == bad_tokens:
                token_credit += 0.5
            else:
                more += 1
        if len(pair.good) < len(pair.bad):
            length_credit += 1.0
            shorter += 1
        elif len(pair.good) == len(pair.bad):
            length_credit += 0.5
        else:
            longer += 1
    total = len(pairs)
    successes = sum(credit.values())
    return PairResult(
        pairs=total,
        accuracy=successes / total,
        interval=wilson_interval(successes, total),
        ties=ties,
        by_phenomenon={name: credit[name] / counts[name] for name in counts},
        intervals_by_phenomenon={
            name: wilson_interval(credit[name], counts[name]) for name in counts
        },
        pairs_by_phenomenon=dict(counts),
        length_only_accuracy=length_credit / total,
        good_longer=longer,
        good_shorter=shorter,
        token_count_only_accuracy=None if token_count is None else token_credit / total,
        good_fewer_tokens=fewer,
        good_more_tokens=more,
        outcomes=tuple(seen),
    )


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD.findall(text)]


def contaminated_pairs(
    pairs: Iterable[MinimalPair], training_texts: Iterable[str], *, n: int = 5
) -> tuple[str, ...]:
    """Identifiers of pairs whose good sentence shares a word ``n``-gram with the training text.

    Case and punctuation are ignored. A sentence with fewer than ``n`` words cannot be flagged: the test is for
    verbatim reuse of a stretch of text, and a shorter sentence has no stretch of that length. This is a screen for
    memorisable overlap, not a proof of independence: a pair that passes the screen may still be a near-paraphrase.
    """
    if n < 1:
        message = f"n must be a positive integer, got {n}"
        raise CorpusError(message)
    wanted: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for pair in pairs:
        words = _words(pair.good)
        for index in range(len(words) - n + 1):
            wanted[tuple(words[index : index + n])].append(pair.identifier)
    flagged: set[str] = set()
    for text in training_texts:
        words = _words(text)
        for index in range(len(words) - n + 1):
            hit = wanted.get(tuple(words[index : index + n]))
            if hit:
                flagged.update(hit)
    return tuple(sorted(flagged))
