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
* **Document boundaries can be carried, not inferred.** The separator marks where a document ends, but the
  attention mask still sees the whole packed block; ``track_documents`` records the document index of every
  position so a training run can mask attention across the boundary. It is opt-in because it doubles the
  per-token bookkeeping and every measured protocol so far ran without it.
* **A pack can say what it is made of.** A capped, seeded sample of a held-out side is not that side, and a loss
  printed over it is a number whose denominator has to be legible: :func:`sample_composition` counts the
  documents, works and registers the blocks actually scored, from the pack's own boundary identifiers rather
  than from the plan that asked for them.

Standard library only: token streams and integer tuples need no numerical framework, so this layer runs in
the dependency-free CI matrix and its arithmetic can be checked by hand.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError
from chakaso.training.subword import DOCUMENT_END, SubwordVocabulary

__all__ = [
    "PackedBatch",
    "PackedExample",
    "count_tokens",
    "pack_units",
    "sample_composition",
    "select_units_by_token_cap",
    "target_segments",
]


@dataclass(frozen=True, slots=True)
class PackedExample:
    """One training position set: a context window and the next-token target for each slot."""

    input_ids: tuple[int, ...]
    targets: tuple[int, ...]
    #: Which document each input position came from, when the packer was asked to track it. ``None`` means
    #: the example predates document tracking (or was packed without it); a training run that wants
    #: document-boundary attention masking must have this populated.
    segment_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.targets):
            message = (
                f"input and target lengths disagree: {len(self.input_ids)} vs {len(self.targets)}"
            )
            raise CorpusError(message)
        if not self.input_ids:
            message = "a packed example with no positions is not an example"
            raise CorpusError(message)
        if self.segment_ids is not None:
            if len(self.segment_ids) != len(self.input_ids):
                message = (
                    f"segment identifiers and input lengths disagree: {len(self.segment_ids)} vs "
                    f"{len(self.input_ids)}"
                )
                raise CorpusError(message)
            if any(value < 0 for value in self.segment_ids):
                message = "segment identifiers must be non-negative document indices"
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


def select_units_by_token_cap(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    max_tokens: int,
    *,
    separate_documents: bool = True,
) -> tuple[TextUnit, ...]:
    """The whole units :func:`pack_units` would use under ``max_tokens``, chosen by counting with ``vocabulary``.

    Packing under a cap counts identifiers with the vocabulary being packed, so two tokenizers cut the same
    ordered documents at different places and a model is then scored on different text. Selecting with one
    reference vocabulary and packing the chosen units, uncapped, with each tokenizer gives them the same
    documents - which is what a bits-per-character comparison across tokenizers needs (H17). The rule is
    :func:`pack_units`'s own: keep taking whole units while the stream is still under the cap.

    Raises:
        CorpusError: ``max_tokens`` is not a positive integer.
    """
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        message = f"max_tokens must be an integer >= 1, got {max_tokens!r}"
        raise CorpusError(message)
    chosen: list[TextUnit] = []
    length = 0
    for unit in units:
        if length >= max_tokens:
            break
        length += len(vocabulary.encode(unit.text)) + (1 if separate_documents else 0)
        chosen.append(unit)
    return tuple(chosen)


def pack_units(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    *,
    context_length: int,
    separate_documents: bool = True,
    track_documents: bool = False,
    max_tokens: int | None = None,
) -> PackedBatch:
    """Pack ``units`` into next-token examples of exactly ``context_length`` positions each.

    ``track_documents`` records, per input position, which document it came from. The default is off
    because it doubles the per-token bookkeeping for a feature only a document-masking run consumes; a run
    that asks for masking must ask for it here too, and the training engine refuses a masked run over
    examples that carry no document identifiers.

    ``max_tokens`` stops after the first whole unit that takes the stream to at least that many identifiers.
    A corpus can be far larger than the positions a run trains on, and packing all of it costs about 46
    bytes per token — a 100M-token corpus is several gigabytes of Python tuples for a run that will read
    20M of them. The cap is applied to whole units, so no document is ever half packed.

    Raises:
        CorpusError: ``context_length`` is not a positive integer, the units produce no usable token
            stream at all (which would otherwise hand a caller an empty training loop), or document
            tracking was requested but the units cannot be identified.
    """
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        message = f"context_length must be an integer >= 1, got {context_length!r}"
        raise CorpusError(message)

    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
    ):
        message = f"max_tokens must be an integer >= 1, got {max_tokens!r}"
        raise CorpusError(message)

    stream_parts: list[int] = []
    segment_parts: list[int] = []
    used_units = 0
    for document_index, unit in enumerate(units):
        if max_tokens is not None and len(stream_parts) >= max_tokens:
            break
        identifiers = vocabulary.encode(unit.text)
        stream_parts.extend(identifiers)
        if track_documents:
            segment_parts.extend([document_index] * len(identifiers))
        if separate_documents:
            stream_parts.append(DOCUMENT_END)
            if track_documents:
                segment_parts.append(document_index)
        used_units += 1
    stream: tuple[int, ...] = tuple(stream_parts)
    segments: tuple[int, ...] | None = tuple(segment_parts) if track_documents else None

    width = context_length + 1
    blocks = len(stream) // width
    examples: tuple[PackedExample, ...] = tuple(
        PackedExample(
            input_ids=tuple(stream[index * width : index * width + context_length]),
            targets=tuple(stream[index * width + 1 : (index + 1) * width]),
            segment_ids=(
                None
                if segments is None
                else tuple(segments[index * width : index * width + context_length])
            ),
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
        document_count=used_units,
    )


def target_segments(examples: Sequence[PackedExample]) -> list[int]:
    """Which document each **scored target** of a tracked pack came from, one entry per position.

    ``segment_ids`` covers the input positions, and a target is the prediction for the input one place later: the
    first token of the next block when the target is a block's last. The final scored token lies one past the last
    block's last input, so no segment covers it and it belongs to that block's final document — the stream was
    still inside that document when the block ended. This is the shift
    :func:`chakaso.training.nn_evaluation.document_losses` scores by, stated once, so a count of positions and a
    sum of losses cannot drift apart.

    Raises:
        CorpusError: an example carries no segment identifiers, so there is no boundary to read; or the blocks are
            not all the same width, in which case there is no single seam rule to apply.
    """
    tracked: list[tuple[int, ...]] = []
    for example in examples:
        if example.segment_ids is None:
            message = (
                "a pack's scored positions need document boundaries: pack it with track_documents=True, or "
                "count the whole pack through PackedBatch.positions"
            )
            raise CorpusError(message)
        tracked.append(example.segment_ids)
    widths = {len(example.input_ids) for example in examples}
    if len(widths) != 1:
        message = f"a pack has one block width; found {sorted(widths)}"
        raise CorpusError(message)
    width = len(examples[0].input_ids)
    segments: list[int] = []
    for block, identifiers in enumerate(tracked):
        for index in range(width):
            if index + 1 < width:
                segments.append(identifiers[index + 1])
            elif block + 1 < len(tracked):
                segments.append(tracked[block + 1][0])
            else:
                segments.append(identifiers[-1])
    return segments


def _register_of(group: str) -> str:
    """The register a work group belongs to — the part before the first hyphen, which is what a partition prefix
    selects on. A group with no hyphen is its own register; a document with no group is counted as ungrouped rather
    than dropped, because a sample that silently omits its odd documents is a sample nobody can audit.
    """
    if not group:
        return "(ungrouped)"
    prefix, separator, _ = group.partition("-")
    return prefix + "-" if separator else group


def sample_composition(batch: PackedBatch, units: Sequence[TextUnit]) -> dict[str, Any]:
    """What a packed side is made of, counted from the blocks that were packed.

    The whole-pack loss a report prints is pooled over a capped, seeded sample of a side, while a register table
    samples each side separately: E2633 read the pack behind E2612 arm T's published **2.0554 bits per character**
    and found no medicine document in it, though the equal-weight mean of five partitions gives medicine a fifth of
    its weight (K-033). Both figures are legitimate; neither is legible as its denominator unless the run says so.
    This counts positions per register from the pack's own boundary identifiers, so the statement describes the text
    that was scored rather than the intent behind the plan that selected it.

    Raises:
        CorpusError: the pack names a document the supplied unit list does not hold — a segment index is only
            meaningful beside the list that was packed — or the pack was built without document tracking.
    """
    segments = target_segments(batch.examples)
    if segments and max(segments) >= len(units):
        message = (
            f"the pack names document {max(segments)} but the caller supplied {len(units)} units; a segment "
            "index is only meaningful beside the unit list that was packed"
        )
        raise CorpusError(message)
    positions_by_segment = Counter(segments)
    registers: dict[str, set[int]] = {}
    groups: dict[str, set[str]] = {}
    for segment in positions_by_segment:
        register = _register_of(units[segment].group)
        registers.setdefault(register, set()).add(segment)
        groups.setdefault(register, set()).add(units[segment].group)
    total = len(segments)
    return {
        "documents": len(positions_by_segment),
        "positions_scored": total,
        "registers": {
            register: {
                "documents": len(members),
                "groups": len(groups[register]),
                "positions": sum(positions_by_segment[segment] for segment in members),
                "position_share": round(
                    sum(positions_by_segment[segment] for segment in members) / total, 4
                ),
            }
            for register, members in sorted(registers.items())
        },
        "works": len({units[segment].group for segment in positions_by_segment}),
    }
