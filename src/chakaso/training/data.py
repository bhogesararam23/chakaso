"""A deterministic train/validation split with leakage prevention.

The honesty of the first experiment rests on one property: the held-out set must be genuinely
held out. If a validation example were allowed to influence the vocabulary or overlap a training
example, a lower validation loss would prove nothing about learning — it would be an artefact of the
split. This module makes that separation a property of the code, not a convention the caller is asked
to remember.

The unit of separation is a **line of the corpus**, treated as a document-sized unit. A unit goes
entirely to one side of the split or the other, never both, so no complete example can be shared
across the boundary. Character windows (the causal-LM examples) are generated *within* one side's text
only, so a window cannot straddle the split.

The split is deterministic: for a given text, fraction and seed, two runs produce identical unit sets,
identical vocabulary, and an identical dataset fingerprint. The seed only decides *which* units land in
validation; it never touches the data. This is the reproducibility the experiment record depends on,
checked by test rather than asserted.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Protocol

from chakaso.training.tokenizer import UNK_ID, CharVocabulary

__all__ = [
    "Example",
    "Split",
    "ThreeWaySplit",
    "character_windows",
    "dataset_fingerprint",
    "fit_vocabulary",
    "split_corpus",
    "split_three_way",
]


class _HasTrainText(Protocol):
    """Anything that exposes the training text — both split shapes do, so the vocabulary is fit the
    same way whether the corpus is split two or three ways. Read-only, as the frozen dataclasses are."""

    @property
    def train_text(self) -> str: ...


@dataclass(frozen=True, slots=True)
class Split:
    """A train/validation separation of a corpus, with the units kept on each side.

    Attributes:
        train_text: The concatenation of the training units.
        validation_text: The concatenation of the held-out units.
        train_units: The individual units assigned to training, in a stable order.
        validation_units: The individual units assigned to validation, in a stable order.
    """

    train_text: str
    validation_text: str
    train_units: tuple[str, ...]
    validation_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ThreeWaySplit:
    """A train/validation/test separation, for protocols that must select without touching test.

    The validation set is for model selection (e.g. early stopping); the test set is scored only for
    reporting, so a selection rule that reads validation never contaminates the reported held-out
    number. Each unit lands wholly in exactly one side. ``train_text`` is present so :func:`fit_vocabulary`
    is identical across the two- and three-way shapes.
    """

    train_text: str
    validation_text: str
    test_text: str
    train_units: tuple[str, ...]
    validation_units: tuple[str, ...]
    test_units: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Example:
    """One causal next-character example: a fixed-size context of identifiers and the target id."""

    context: tuple[int, ...]
    target: int


def _units(text: str) -> tuple[str, ...]:
    """Split corpus text into non-empty line units, dropping only blank separators."""
    return tuple(line for line in text.splitlines() if line)


def split_corpus(text: str, *, validation_fraction: float = 0.2, seed: int = 0) -> Split:
    """Split ``text`` into training and validation line units, deterministically.

    ``validation_fraction`` is the share of units held out. The unit set is shuffled with ``seed`` and
    partitioned, so the same inputs always give the same split.

    Raises:
        ValueError: the fraction is not in the open interval ``(0, 1)`` where a meaningful split needs
            units on both sides, or the text has too few units to hold any out.
    """
    if not 0.0 < validation_fraction < 1.0:
        message = (
            f"validation_fraction must be strictly between 0 and 1, got {validation_fraction!r}"
        )
        raise ValueError(message)
    if not isinstance(seed, int) or isinstance(seed, bool):
        message = f"seed must be an integer, got {seed!r}"
        raise ValueError(message)

    units = _units(text)
    if len(units) < 2:
        message = "a train/validation split needs at least two line units"
        raise ValueError(message)

    # A deterministic shuffle of indices: random.Random is seeded, so the permutation is reproducible
    # and independent of the interpreter's hash randomization.
    order = list(range(len(units)))
    random.Random(seed).shuffle(order)

    holdout_count = max(1, round(len(units) * validation_fraction))
    if holdout_count >= len(units):
        message = "the validation fraction would leave no units for training"
        raise ValueError(message)

    validation_positions = set(order[:holdout_count])
    validation_units = tuple(units[i] for i in range(len(units)) if i in validation_positions)
    train_units = tuple(units[i] for i in range(len(units)) if i not in validation_positions)

    # Keep the separator in each side's text so windows generated within a side respect line bounds
    # exactly as they did in the original corpus.
    joiner = "\n"
    return Split(
        train_text=joiner + joiner.join(train_units) + joiner,
        validation_text=joiner + joiner.join(validation_units) + joiner,
        train_units=train_units,
        validation_units=validation_units,
    )


def split_three_way(
    text: str,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> ThreeWaySplit:
    """Split ``text`` into train/validation/test line units, deterministically and disjointly.

    The validation set is intended for model selection and the test set for reporting only, so the two
    are kept separate from each other and from training. Each unit goes wholly to one side. The
    partition is a deterministic shuffle of unit indices driven by ``seed``.

    Raises:
        ValueError: a fraction is not in ``(0, 1)``, the two held-out fractions leave nothing for
            training, or the corpus has too few units to put at least one on each side.
    """
    for name, value in (
        ("validation_fraction", validation_fraction),
        ("test_fraction", test_fraction),
    ):
        if not 0.0 < value < 1.0:
            message = f"{name} must be strictly between 0 and 1, got {value!r}"
            raise ValueError(message)
    if validation_fraction + test_fraction >= 1.0:
        message = (
            f"validation_fraction + test_fraction must be < 1, got "
            f"{validation_fraction + test_fraction}"
        )
        raise ValueError(message)
    if not isinstance(seed, int) or isinstance(seed, bool):
        message = f"seed must be an integer, got {seed!r}"
        raise ValueError(message)

    units = _units(text)
    if len(units) < 3:
        message = "a train/validation/test split needs at least three line units"
        raise ValueError(message)

    order = list(range(len(units)))
    random.Random(seed).shuffle(order)

    test_count = max(1, round(len(units) * test_fraction))
    validation_count = max(1, round(len(units) * validation_fraction))
    if test_count + validation_count >= len(units):
        message = "the validation and test fractions would leave no units for training"
        raise ValueError(message)

    test_positions = set(order[:test_count])
    validation_positions = set(order[test_count : test_count + validation_count])
    joiner = "\n"
    sides: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for index, unit in enumerate(units):
        if index in test_positions:
            sides["test"].append(unit)
        elif index in validation_positions:
            sides["validation"].append(unit)
        else:
            sides["train"].append(unit)

    def _text(unit_list: list[str]) -> str:
        return joiner + joiner.join(unit_list) + joiner

    return ThreeWaySplit(
        train_text=_text(sides["train"]),
        validation_text=_text(sides["validation"]),
        test_text=_text(sides["test"]),
        train_units=tuple(sides["train"]),
        validation_units=tuple(sides["validation"]),
        test_units=tuple(sides["test"]),
    )


def fit_vocabulary(split: _HasTrainText) -> CharVocabulary:
    """Build the vocabulary from **training text only**.

    Held-out characters therefore never widen the identifier space; a validation- or test-only character
    encodes to :data:`UNK_ID`, which is exactly the leakage-safe behaviour the experiment relies on.
    """
    return CharVocabulary((split.train_text,))


def character_windows(
    text: str, vocabulary: CharVocabulary, context_size: int
) -> tuple[Example, ...]:
    """Produce one causal example per character position of ``text``.

    At position ``i`` the context is the previous ``context_size`` identifiers and the target is the
    identifier of the character at ``i``. Positions near the start are left-padded with
    :data:`UNK_ID` so every character is predicted from a bounded history, identically in both splits.
    """
    if context_size < 1:
        message = f"context_size must be at least 1, got {context_size}"
        raise ValueError(message)

    ids = vocabulary.encode(text)
    if len(ids) < 2:
        return ()

    examples: list[Example] = []
    for i in range(1, len(ids)):
        start = max(0, i - context_size)
        window = ids[start:i]
        if len(window) < context_size:
            window = (UNK_ID,) * (context_size - len(window)) + window
        examples.append(Example(context=window, target=ids[i]))
    return tuple(examples)


def dataset_fingerprint(split: Split, *, context_size: int) -> str:
    """A stable digest identifying the exact data and windowing an experiment used.

    It covers the split contents and the context size, so a change to either moves the fingerprint —
    the same regression-guard role the pinned benchmark fingerprint plays elsewhere (ADR-0013).
    """
    payload = "\x1e".join(
        (
            split.train_text,
            split.validation_text,
            str(context_size),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
