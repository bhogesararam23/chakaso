"""Packing a split's documents into fixed-length training sequences.

A causal language model learns from contiguous token streams: at every position the model predicts the
next token from the prefix it has seen. This module turns one side of a corpus split into those positions,
and it does so **per side**, which is where the leakage guarantee actually lives — a block that mixed a
training document with a held-out one would teach the model the answer and then score it as unseen.

The rules, and the reason each one is a rule:

* **A document separator terminates every document.** Concatenating raw text would let a block span two
  unrelated documents invisibly, so the model is given an explicit end-of-document token to learn the
  boundary rather than infer it from punctuation.
* **A block is ``context_length + 1`` tokens.** ``input_ids`` is the first ``context_length``, ``targets``
  is the sequence shifted by one. The last token of a block is a target, never an input, so no example is
  ever scored on a position it was not conditioned on.
* **Long documents are truncated by being cut into blocks**, not discarded: a document longer than the
  context still contributes every full block it contains.
* **A short final block is dropped, and the drop is reported.** Padding with a fake token would teach the
  model to predict the padding, and silently discarding the remainder would hide how much of the corpus a
  run never saw.

Standard library only: token streams and integer tuples need no numerical framework, so this layer runs in
the dependency-free CI matrix and its arithmetic can be checked by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.subword import DOCUMENT_END, SubwordVocabulary

__all__ = ["PackedBatch", "PackedExample", "count_tokens", "pack_units"]


@dataclass(frozen=True, slots=True)
class PackedExample:
    """One training position set: a context window and the next-token target for each slot."""

    input_ids: tuple[int, ...]
    targets: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.targets):
            message = (
                f"input and target lengths disagree: {len(self.input_ids)} vs {len(self.targets)}"
            )
            raise CorpusError(message)
        if not self.input_ids:
            message = "a packed example with no positions is not an example"
            raise CorpusError(message)


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """Packed examples for one side of a split, plus the accounting a record must state.

    Attributes:
        examples: The blocks, in stream order.
        tokens_available: How many tokens the side's text produced, including the separators.
        dropped_tail: Tokens left over because they did not fill a final block.
        document_count: How many documents contributed.
    """

    examples: tuple[PackedExample, ...]
    tokens_available: int
    dropped_tail: int
    document_count: int

    @property
    def positions(self) -> int:
        """Total predicted positions — the denominator of any loss computed over this batch."""
        return sum(len(example.targets) for example in self.examples)

    @property
    def context_length(self) -> int:
        """Window width every example in this batch shares."""
        if not self.examples:
            return 0
        return len(self.examples[0].input_ids)


def count_tokens(units: Sequence[TextUnit], vocabulary: SubwordVocabulary) -> int:
    """Total identifiers the units encode to, counting one separator per document."""
    return sum(vocabulary.token_count(unit.text) + 1 for unit in units)


def token_stream(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    *,
    separate_documents: bool = True,
) -> tuple[int, ...]:
    """Encode every unit in order, optionally terminating each with the document separator."""
    produced: list[int] = []
    for unit in units:
        produced.extend(vocabulary.encode(unit.text))
        if separate_documents:
            produced.append(DOCUMENT_END)
    return tuple(produced)


def pack_units(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    *,
    context_length: int,
    separate_documents: bool = True,
) -> PackedBatch:
    """Pack ``units`` into next-token examples of exactly ``context_length`` positions each.

    Raises:
        CorpusError: ``context_length`` is not a positive integer, or the units produce no usable token
            stream at all (which would otherwise hand a caller an empty training loop).
    """
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        message = f"context_length must be an integer >= 1, got {context_length!r}"
        raise CorpusError(message)

    stream = token_stream(units, vocabulary, separate_documents=separate_documents)
    width = context_length + 1
    blocks = len(stream) // width
    examples: tuple[PackedExample, ...] = tuple(
        PackedExample(
            input_ids=tuple(stream[index * width : index * width + context_length]),
            targets=tuple(stream[index * width + 1 : (index + 1) * width]),
        )
        for index in range(blocks)
    )
    if not examples:
        message = (
            f"packing produced no examples: {len(stream)} token(s) from {len(units)} document(s) fill "
            f"zero blocks of width {width}"
        )
        raise CorpusError(message)
    return PackedBatch(
        examples=examples,
        tokens_available=len(stream),
        dropped_tail=len(stream) - blocks * width,
        document_count=len(units),
    )
