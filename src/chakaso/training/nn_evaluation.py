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
from chakaso.training.packing import PackedExample
from chakaso.training.subword import SubwordVocabulary

__all__ = ["evaluate_artifact", "evaluate_model", "score_model_on_examples"]


def score_model_on_examples(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    *,
    batch_size: int = 32,
) -> tuple[list[float], list[int], list[int]]:
    """Return per-position cross-entropies, target identifiers and predicted identifiers.

    Positions come out in block order, so the concatenation of the targets is the scored token stream —
    which is what makes the bits-per-character denominator the decoded length of *that* text.

    Scoring is batched because the alternative is a measurement artefact: at one forward pass per block,
    evaluating 3,908 test blocks of a 256-token model took longer than the 2,400 training steps did, which
    is how an experiment ends up spending most of its wall clock on the part that updates nothing. Rows of
    unequal width are scored one at a time, since a batch has to be rectangular, and a test asserts that
    both paths return the same numbers for the same model.

    Raises:
        TrainingScheduleError: there are no examples to score, or ``batch_size`` is not a positive integer.
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
    nats, targets, predictions = score_model_on_examples(model, examples, batch_size=batch_size)
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
