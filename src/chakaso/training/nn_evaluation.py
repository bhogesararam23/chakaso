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

from chakaso.training.model_metrics import ModelMetrics, metrics_from_scores
from chakaso.training.nn_checkpoint import ModelArtifact, TrainedModel
from chakaso.training.packing import PackedExample
from chakaso.training.subword import SubwordVocabulary

__all__ = ["evaluate_artifact", "evaluate_model", "score_model_on_examples"]


def score_model_on_examples(
    model: TrainedModel,
    examples: Sequence[PackedExample],
) -> tuple[list[float], list[int], list[int]]:
    """Return per-position cross-entropies, target identifiers and predicted identifiers.

    Positions come out in block order, so the concatenation of the targets is the scored token stream —
    which is what makes the bits-per-character denominator the decoded length of *that* text.

    The per-block arithmetic is vectorized (a gather over the target column and one argmax) rather than a
    Python loop over positions: on a corpus of millions of characters the loop was the evaluation, not the
    model. A test asserts the gathered loss and the per-position softmax values agree with each other.
    """
    model.eval()
    nats: list[float] = []
    targets: list[int] = []
    predictions: list[int] = []
    with torch.no_grad():
        for example in examples:
            inputs = torch.tensor([list(example.input_ids)], dtype=torch.int64)
            target_row = torch.tensor([list(example.targets)], dtype=torch.int64)
            logits, _ = model(inputs)
            log_probabilities = torch.log_softmax(logits, dim=-1)
            selected = log_probabilities.gather(-1, target_row.unsqueeze(-1)).squeeze(-1)
            nats.extend(-value for value in selected[0].tolist())
            target_values = target_row[0].tolist()
            targets.extend(int(value) for value in target_values)
            predictions.extend(int(value) for value in logits[0].argmax(dim=-1).tolist())
    return nats, targets, predictions


def evaluate_model(
    model: TrainedModel,
    examples: Sequence[PackedExample],
    vocabulary: SubwordVocabulary,
) -> ModelMetrics:
    """Score ``model`` on ``examples`` using the shared metric arithmetic.

    Raises:
        TrainingError: ``examples`` is empty, or the vocabulary cannot decode what was scored. The message
            comes from :func:`chakaso.training.model_metrics.metrics_from_scores`, which refuses a
            measurement over nothing rather than reporting a loss of zero.
    """
    nats, targets, predictions = score_model_on_examples(model, examples)
    return metrics_from_scores(nats, targets, predictions, vocabulary)


def evaluate_artifact(
    artifact: ModelArtifact,
    examples: Sequence[PackedExample],
) -> ModelMetrics:
    """Score a saved artifact on examples, using the tokenizer bound inside it.

    The vocabulary is taken from the artifact rather than supplied, because scoring weights against a
    tokenizer they were not trained with is the error this artifact format exists to refuse.
    """
    return evaluate_model(artifact.to_model(), examples, artifact.vocabulary)
