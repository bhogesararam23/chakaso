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

from chakaso.training.tokenizer import UNK_ID, CharVocabulary

__all__ = [
    "Example",
    "Split",
    "character_windows",
    "dataset_fingerprint",
    "fit_vocabulary",
    "split_corpus",
]


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


def fit_vocabulary(split: Split) -> CharVocabulary:
    """Build the vocabulary from **training text only**.

    Held-out characters therefore never widen the identifier space; a validation-only character
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
