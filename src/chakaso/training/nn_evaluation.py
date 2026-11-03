"""Held-out evaluation of a trained Transformer, in the units the record reports.

:mod:`chakaso.training.model_metrics` owns the arithmetic; this module owns the torch side of it: run the
model over every packed example under ``no_grad``, take the per-position cross-entropy and the argmax, and
hand the totals to the shared metric function. The split matters — the same arithmetic then scores the
character model, which is why a character model and a subword model can appear in one table.

Nothing here reads the test set: the caller supplies the examples it wants scored, and the experiment
driver decides which side of the split that is.

Requires the optional ``[nn]`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from chakaso.training.data import TextUnit
from chakaso.training.errors import TrainingError, TrainingScheduleError
from chakaso.training.model_metrics import ModelMetrics, metrics_from_scores
from chakaso.training.nn_checkpoint import ModelArtifact, TrainedModel
from chakaso.training.nn_training import _segment_tensor, build_model
from chakaso.training.packing import PackedExample
from chakaso.training.subword import DOCUMENT_END, SubwordVocabulary

__all__ = [
    "DocumentLoss",
    "continuation_log_probabilities",
    "document_losses",
    "evaluate_artifact",
    "evaluate_model",
    "position_losses",
    "score_model_on_examples",
    "sequence_log_probabilities",
    "untrained_model_like",
]


@dataclass(frozen=True, slots=True)
class _ScoredRow:
    """One example's cross-entropies, targets and argmaxes — the row every scorer here walks."""

    losses: list[float]
    targets: list[int]
    predictions: list[int]


@dataclass(frozen=True, slots=True)
class DocumentLoss:
    """What one held-out document contributed to a packed evaluation.

    ``bits_per_character`` is the same arithmetic as the whole-pack metric on that document's own slice, so a
    comparison between two arms can be paired document by document; ``group`` is the work the split held out,
    which is the unit a bootstrap resamples because a book contributes many of these rows and they are not
    independent observations.
    """

    segment: int
    group: str
    positions: int
    nats: float
    characters: int

    @property
    def bits_per_character(self) -> float:
        """This document's loss in the series' unit, or ``inf`` if it scored no characters at all."""
        if not self.characters:
            return math.inf
        return math.log2(math.e) * self.nats / self.characters


def score_model_on_examples(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    *,
    batch_size: int = 32,
    document_masking: bool = False,
) -> tuple[list[float], list[int], list[int]]:
    """Return per-position cross-entropies, target identifiers and predicted identifiers.

    Positions come out in block order, so the concatenation of the targets is the scored token stream —
    which is what makes the bits-per-character denominator the decoded length of *that* text. The walk is
    :func:`_scored_rows`; this is its flattened form.

    Scoring is batched because the alternative is a measurement artefact: at one forward pass per block,
    evaluating 3,908 test blocks of a 256-token model took longer than the 2,400 training steps did, which
    is how an experiment ends up spending most of its wall clock on the part that updates nothing. Rows of
    unequal width are scored one at a time, since a batch has to be rectangular, and a test asserts that
    both paths return the same numbers for the same model.

    ``document_masking`` must match how the model was trained: a model trained with attention confined to a
    document and scored with full causal attention is evaluated under a condition it never saw, and the
    held-out number would silently measure that mismatch rather than the model. The loop's own validation
    loss already honours the setting (:func:`chakaso.training.nn_training.evaluate_loss`); this is the same
    switch for the held-out scorer.

    Raises:
        TrainingScheduleError: there are no examples to score, ``batch_size`` is not a positive integer, or
            ``document_masking`` is set and an example carries no segment identifiers.
    """
    nats: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    for row in _scored_rows(
        model, examples, batch_size=batch_size, document_masking=document_masking
    ):
        nats.extend(row.losses)
        targets.extend(row.targets)
        predictions.extend(row.predictions)
    return nats, targets, predictions


def _scored_rows(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    *,
    batch_size: int = 32,
    document_masking: bool = False,
) -> list[_ScoredRow]:
    """One :class:`_ScoredRow` per example, in the order given — the single forward-pass path every scorer here shares.

    Kept in one place because two paths that compute the same held-out number differently is how an arm
    comparison becomes a statement about the evaluation code.
    """
    if not examples:
        message = "cannot score zero examples"
        raise TrainingScheduleError(message)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        message = f"batch_size must be an integer >= 1, got {batch_size!r}"
        raise TrainingScheduleError(message)
    model.eval()
    rows: list[_ScoredRow] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = list(examples[start : start + batch_size])
            widths = {len(row.input_ids) for row in chunk}
            if len(widths) != 1:
                # A ragged chunk is legal input and rare; scoring it row by row keeps one code path for the
                # numbers rather than two that could disagree.
                groups: list[Sequence[PackedExample]] = [[row] for row in chunk]
            else:
                groups = [chunk]
            for group in groups:
                inputs = torch.tensor([list(row.input_ids) for row in group], dtype=torch.int64)
                wanted = torch.tensor([list(row.targets) for row in group], dtype=torch.int64)
                if document_masking:
                    logits, _ = model(inputs, segment_ids=_segment_tensor(group))
                else:
                    logits, _ = model(inputs)
                log_probabilities = torch.log_softmax(logits, dim=-1)
                selected = log_probabilities.gather(-1, wanted.unsqueeze(-1)).squeeze(-1)
                best = logits.argmax(dim=-1)
                for index in range(len(group)):
                    rows.append(
                        _ScoredRow(
                            losses=[-float(value) for value in selected[index]],
                            targets=[int(value) for value in wanted[index]],
                            predictions=[int(value) for value in best[index]],
                        )
                    )
    return rows


def position_losses(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    *,
    batch_size: int = 32,
    document_masking: bool = False,
) -> list[list[float]]:
    """Per-position cross-entropies, one list per packed example, in block order."""
    return [
        row.losses
        for row in _scored_rows(
            model, examples, batch_size=batch_size, document_masking=document_masking
        )
    ]


def document_losses(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    vocabulary: SubwordVocabulary,
    units: Sequence[TextUnit] = (),
    *,
    batch_size: int = 32,
    document_masking: bool = False,
) -> list[DocumentLoss]:
    """Split a packed evaluation into the documents that produced it, with each one's loss and character count.

    A paired comparison between two arms needs a unit to pair on, and the packed block is not one: it is a
    window over a stream, and the same document usually spans several. ``segment_ids`` says which document each
    *input* position came from, and a target at position ``i`` is the prediction for the input token one place
    later — the first token of the next block when ``i`` is the block's last. That shift is the whole of this
    function's arithmetic, and it is why the totals below equal the shared metric exactly rather than
    approximately.

    The final scored token lies one past the last block's last input, so no segment covers it: exactly one
    position per pack, attributed to that last block's final document, which is the document the stream was
    still inside when the block ended.

    Raises:
        TrainingScheduleError: an example carries no segment identifiers — the pack was built without
            ``track_documents=True``, so there is no boundary to split on and the function refuses to guess
            one — or the pack names a document the caller's unit list does not contain.
    """
    segments: list[tuple[int, ...]] = []
    for example in examples:
        if example.segment_ids is None:
            message = (
                "per-document losses need document boundaries: pack the evaluation with "
                "track_documents=True, or score the whole pack through evaluate_model"
            )
            raise TrainingScheduleError(message)
        segments.append(example.segment_ids)
    rows = _scored_rows(model, examples, batch_size=batch_size, document_masking=document_masking)
    # ``segment`` answers for every scored token: the segments of each block cover its own input positions,
    # and a block's last target is the next block's first input token.
    width = len(examples[0].input_ids)
    grouped: dict[int, list[float]] = {}
    identifiers: dict[int, list[int]] = {}
    for block, row in enumerate(rows):
        for index, target in enumerate(row.targets):
            if index + 1 < width:
                segment = segments[block][index + 1]
            elif block + 1 < len(segments):
                segment = segments[block + 1][0]
            else:
                segment = segments[block][-1]
            grouped.setdefault(segment, []).append(row.losses[index])
            identifiers.setdefault(segment, []).append(target)

    if units and max(grouped) >= len(units):
        message = (
            f"the pack names document {max(grouped)} but the caller supplied {len(units)} units; a segment "
            "index is only meaningful beside the unit list that was packed"
        )
        raise TrainingScheduleError(message)
    out: list[DocumentLoss] = []
    for segment in sorted(grouped):
        characters = len(vocabulary.decode(identifiers[segment]))
        out.append(
            DocumentLoss(
                segment=segment,
                group=units[segment].group if units else "",
                positions=len(grouped[segment]),
                nats=math.fsum(grouped[segment]),
                characters=characters,
            )
        )
    return out


def evaluate_model(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    vocabulary: SubwordVocabulary,
    *,
    batch_size: int = 32,
    document_masking: bool = False,
) -> ModelMetrics:
    """Score ``model`` on ``examples`` using the shared metric arithmetic.

    Raises:
        TrainingError: ``examples`` is empty, or the vocabulary cannot decode what was scored. The message
            comes from :func:`chakaso.training.model_metrics.metrics_from_scores`, which refuses a
            measurement over nothing rather than reporting a loss of zero.
    """
    if not examples:
        # The refusal belongs to the metric, not to the loop: an evaluation over nothing has no loss, and
        # "at least one scored position" is the message every caller of this function has always been
        # documented to receive.
        message = (
            "cannot evaluate a model over an empty example set: there is no such thing as at least one "
            "scored position"
        )
        raise TrainingError(message)
    nats, targets, predictions = score_model_on_examples(
        model, examples, batch_size=batch_size, document_masking=document_masking
    )
    return metrics_from_scores(nats, targets, predictions, vocabulary)


def evaluate_artifact(
    artifact: ModelArtifact,
    examples: Sequence[PackedExample],
    *,
    batch_size: int = 32,
) -> ModelMetrics:
    """Score a saved artifact on examples, using the tokenizer bound inside it.

    The vocabulary is taken from the artifact rather than supplied, because scoring weights against a
    tokenizer they were not trained with is the error this artifact format exists to refuse.
    """
    return evaluate_model(artifact.to_model(), examples, artifact.vocabulary, batch_size=batch_size)


def sequence_log_probabilities(
    model: TrainedModel,
    vocabulary: SubwordVocabulary,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
) -> list[float]:
    """The natural-log probability the model gives each text, as a document that begins after a separator.

    Training streams separate documents with ``DOCUMENT_END``, so a text is scored as the continuation of one:
    ``log p(t1 | <doc-end>) + log p(t2 | <doc-end>, t1) + ...`` over its identifiers, with no end-of-text term.
    The sum, not the mean, is returned: a minimal-pair comparison is between whole sentences (see
    :mod:`chakaso.training.minimal_pairs`, which also reports a length-only and a token-count-only reading because a
    sum rewards short sentences).

    Texts of the same identifier count are scored together in one forward pass, which needs no padding and so
    computes exactly what scoring them one at a time would (a test holds the two to 1e-4 nats); the result is in
    the order of ``texts`` and a text's value does not depend on what it is scored beside.

    Raises:
        TrainingError: a text is empty, or has more identifiers than the model's context holds - it is refused
            rather than truncated, since a truncated sentence is a different sentence.
    """
    context = int(model.config.context_length)
    sequences: list[tuple[int, ...]] = []
    for text in texts:
        identifiers = vocabulary.encode(text)
        if not identifiers:
            message = "cannot score an empty text: it has no identifiers to give a probability to"
            raise TrainingError(message)
        if len(identifiers) > context:
            message = (
                f"a text of {len(identifiers)} identifiers does not fit the model's context of {context} "
                "once the document-start separator is added; it is not truncated"
            )
            raise TrainingError(message)
        sequences.append((DOCUMENT_END, *identifiers))
    by_length: dict[int, list[int]] = {}
    for index, sequence in enumerate(sequences):
        by_length.setdefault(len(sequence), []).append(index)
    totals = [0.0] * len(sequences)
    model.eval()
    with torch.no_grad():
        for members in by_length.values():
            for start in range(0, len(members), batch_size):
                chunk = members[start : start + batch_size]
                inputs = torch.tensor([list(sequences[i][:-1]) for i in chunk], dtype=torch.int64)
                wanted = torch.tensor([list(sequences[i][1:]) for i in chunk], dtype=torch.int64)
                logits, _ = model(inputs)
                selected = (
                    torch.log_softmax(logits, dim=-1).gather(-1, wanted.unsqueeze(-1)).squeeze(-1)
                )
                for row, index in enumerate(chunk):
                    totals[index] = float(selected[row].sum())
    return totals


def continuation_log_probabilities(
    model: TrainedModel,
    requests: Sequence[tuple[Sequence[int], Sequence[int]]],
    *,
    batch_size: int = 64,
) -> list[float]:
    """The natural-log probability of each ``continuation`` given its ``prefix``, both already in identifiers.

    A request ``(prefix, continuation)`` is scored as ``log p(continuation | <doc-end> prefix)``: only the continuation's own positions are
    summed, so the prefix conditions the model and contributes nothing to the value. An empty prefix asks for the continuation straight after the
    separator, which is the prefix-blind probability the pointwise-mutual-information reading subtracts. Requests of equal total length share a forward
    pass (no padding, so the values are what scoring them one at a time gives, to 1e-4 nats), and the result is in the order of ``requests``.

    Raises:
        TrainingError: a continuation is empty, or a passage and its continuation do not fit the model's context together.
    """
    context = int(model.config.context_length)
    sequences: list[tuple[int, ...]] = []
    lengths: list[int] = []
    for prefix, continuation in requests:
        if not continuation:
            message = (
                "cannot score an empty continuation: it has no identifiers to give a probability to"
            )
            raise TrainingError(message)
        sequence = (DOCUMENT_END, *prefix, *continuation)
        if len(sequence) - 1 > context:
            message = (
                f"a passage of {len(prefix)} identifiers and a continuation of {len(continuation)} do not fit "
                f"the model's context of {context} with the separator; nothing is truncated"
            )
            raise TrainingError(message)
        sequences.append(sequence)
        lengths.append(len(continuation))
    by_length: dict[tuple[int, int], list[int]] = {}
    for index, sequence in enumerate(sequences):
        by_length.setdefault((len(sequence), lengths[index]), []).append(index)
    totals = [0.0] * len(sequences)
    model.eval()
    with torch.no_grad():
        for (_, own), members in by_length.items():
            for start in range(0, len(members), batch_size):
                chunk = members[start : start + batch_size]
                inputs = torch.tensor([list(sequences[i][:-1]) for i in chunk], dtype=torch.int64)
                wanted = torch.tensor([list(sequences[i][1:]) for i in chunk], dtype=torch.int64)
                logits, _ = model(inputs)
                selected = (
                    torch.log_softmax(logits, dim=-1).gather(-1, wanted.unsqueeze(-1)).squeeze(-1)
                )
                for row, index in enumerate(chunk):
                    totals[index] = float(selected[row, -own:].sum())
    return totals


def untrained_model_like(artifact: ModelArtifact, seed: int) -> TrainedModel:
    """A randomly initialised model of the artifact's architecture, seeded so the control is reproducible.

    The floor of any likelihood probe: whatever an untrained model scores is what the probe measures without learning. The global random
    state is restored afterwards, so building a control does not move any stream a caller is drawing from.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model: TrainedModel = build_model(artifact.config)
    model.eval()
    return model
