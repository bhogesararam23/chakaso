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

from collections.abc import Sequence

import torch

from chakaso.training.errors import TrainingError, TrainingScheduleError
from chakaso.training.model_metrics import ModelMetrics, metrics_from_scores
from chakaso.training.nn_checkpoint import ModelArtifact, TrainedModel
from chakaso.training.nn_training import _segment_tensor
from chakaso.training.packing import PackedExample
from chakaso.training.subword import DOCUMENT_END, SubwordVocabulary

__all__ = [
    "evaluate_artifact",
    "evaluate_model",
    "score_model_on_examples",
    "sequence_log_probabilities",
]


def score_model_on_examples(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    *,
    batch_size: int = 32,
    document_masking: bool = False,
) -> tuple[list[float], list[int], list[int]]:
    """Return per-position cross-entropies, target identifiers and predicted identifiers.

    Positions come out in block order, so the concatenation of the targets is the scored token stream —
    which is what makes the bits-per-character denominator the decoded length of *that* text.

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
    if not examples:
        message = "cannot score zero examples"
        raise TrainingScheduleError(message)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        message = f"batch_size must be an integer >= 1, got {batch_size!r}"
        raise TrainingScheduleError(message)
    model.eval()
    nats: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = list(examples[start : start + batch_size])
            widths = {len(row.input_ids) for row in chunk}
            if len(widths) != 1:
                # A ragged chunk is legal input and rare; scoring it row by row keeps one code path for the
                # numbers rather than two that could disagree.
                rows: list[Sequence[PackedExample]] = [[row] for row in chunk]
            else:
                rows = [chunk]
            for group in rows:
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
                    nats.extend(-float(value) for value in selected[index])
                    targets.extend(int(value) for value in wanted[index])
                    predictions.extend(int(value) for value in best[index])
    return nats, targets, predictions


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
    model: TrainedModel, vocabulary: SubwordVocabulary, texts: Sequence[str]
) -> list[float]:
    """The natural-log probability the model gives each text, as a document that begins after a separator.

    Training streams separate documents with ``DOCUMENT_END``, so a text is scored as the continuation of one:
    ``log p(t1 | <doc-end>) + log p(t2 | <doc-end>, t1) + ...`` over its identifiers, with no end-of-text term. Each text
    is scored on its own, so the value does not depend on what it is scored beside. The sum, not the mean, is returned:
    a minimal-pair comparison is between whole sentences (see :mod:`chakaso.training.minimal_pairs`, which also reports
    a length-only reading because a sum rewards short sentences).

    Raises:
        TrainingError: a text is empty, or has more identifiers than the model's context holds - it is refused
            rather than truncated, since a truncated sentence is a different sentence.
    """
    context = int(model.config.context_length)
    model.eval()
    totals: list[float] = []
    with torch.no_grad():
        for text in texts:
            identifiers = vocabulary.encode(text)
            if not identifiers:
                message = (
                    "cannot score an empty text: it has no identifiers to give a probability to"
                )
                raise TrainingError(message)
            sequence = (DOCUMENT_END, *identifiers)
            if len(sequence) - 1 > context:
                message = (
                    f"a text of {len(identifiers)} identifiers does not fit the model's context of {context} "
                    "once the document-start separator is added; it is not truncated"
                )
                raise TrainingError(message)
            inputs = torch.tensor([list(sequence[:-1])], dtype=torch.int64)
            wanted = torch.tensor([list(sequence[1:])], dtype=torch.int64)
            logits, _ = model(inputs)
            selected = (
                torch.log_softmax(logits, dim=-1).gather(-1, wanted.unsqueeze(-1)).squeeze(-1)
            )
            totals.append(float(selected.sum()))
    return totals
