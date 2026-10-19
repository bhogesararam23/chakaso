"""The training loop: run the learning curve and evaluate it as it goes.

This is the "real optimization procedure" the experiment needs. It wires the pieces together —
fit the vocabulary on the training split, build causal windows, initialize the model from the seed,
then take full-batch Adam steps over the training examples while recording the training loss and, at
intervals, the held-out validation loss. The curve it produces is the measured object the experiment
report will cite; nothing here is a stand-in for training and nothing is a canned result.

Full-batch gradient descent is chosen deliberately for the first experiment: the training set is tiny,
the batch is the whole set, so the update is deterministic given the seed and the loss curve is smooth
and easy to read. That is a simplification, and it is recorded as one (ADR-0030) — a later, larger
corpus would need mini-batching, which is a different experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chakaso.training.config import TrainingConfig
from chakaso.training.data import Split, character_windows, dataset_fingerprint, fit_vocabulary
from chakaso.training.model import CharCausalMLP, Params, batches_from_examples
from chakaso.training.optimizer import Adam

__all__ = ["EvaluationPoint", "TrainingResult", "run_training"]


@dataclass(frozen=True, slots=True)
class EvaluationPoint:
    """A recorded moment in the run: a step number with its training and held-out losses."""

    step: int
    train_loss: float
    validation_loss: float


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The outcome of one training run, with everything the record must state.

    Attributes:
        config: The configuration that produced this run.
        vocab_fingerprint: The identifier space the model was trained in.
        dataset_digest: The fingerprint of the exact split and windowing used.
        parameter_count: Number of trainable scalars.
        token_counts: Number of training and validation examples actually evaluated.
        curve: The recorded loss trajectory, starting at step 0 (before any update).
        final_params: The trained parameter bundle.
    """

    config: TrainingConfig
    vocab_fingerprint: str
    dataset_digest: str
    parameter_count: int
    train_tokens: int
    validation_tokens: int
    curve: tuple[EvaluationPoint, ...]
    final_params: Params

    @property
    def initial_loss(self) -> float:
        """The loss before any update."""
        return self.curve[0].train_loss

    @property
    def final_train_loss(self) -> float:
        """The training loss at the last recorded step."""
        return self.curve[-1].train_loss

    @property
    def final_validation_loss(self) -> float:
        """The held-out loss at the last recorded step."""
        return self.curve[-1].validation_loss


def run_training(split: Split, config: TrainingConfig) -> TrainingResult:
    """Train the model on ``split`` and return the measured curve and the trained parameters.

    Raises:
        ValueError: either side of the split yields no causal windows, so there is nothing to train or
            evaluate. A run over empty data is refused rather than reported as a loss of zero.
    """
    vocabulary = fit_vocabulary(split)
    train_examples = character_windows(split.train_text, vocabulary, config.context_size)
    validation_examples = character_windows(split.validation_text, vocabulary, config.context_size)

    if not train_examples:
        message = "the training split produced no causal windows; cannot train"
        raise ValueError(message)
    if not validation_examples:
        message = "the validation split produced no causal windows; nothing to evaluate"
        raise ValueError(message)

    train_contexts, train_targets = batches_from_examples(train_examples, config.context_size)
    validation_contexts, validation_targets = batches_from_examples(
        validation_examples, config.context_size
    )

    model = CharCausalMLP(
        vocab_size=vocabulary.size,
        context_size=config.context_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
    )
    rng = np.random.default_rng(config.seed)
    params = model.initialize(rng)
    optimizer = Adam(learning_rate=config.learning_rate)

    initial_train = model.forward(params, train_contexts, train_targets)
    initial_validation = model.forward(params, validation_contexts, validation_targets)
    curve: list[EvaluationPoint] = [EvaluationPoint(0, initial_train, initial_validation)]

    for step in range(1, config.steps + 1):
        grads = model.gradient(params, train_contexts, train_targets)
        params = optimizer.step(params, grads)

        if step == config.steps or step % config.eval_every == 0:
            train_loss = model.forward(params, train_contexts, train_targets)
            validation_loss = model.forward(params, validation_contexts, validation_targets)
            curve.append(EvaluationPoint(step, train_loss, validation_loss))

    return TrainingResult(
        config=config,
        vocab_fingerprint=vocabulary.fingerprint,
        dataset_digest=dataset_fingerprint(split, context_size=config.context_size),
        parameter_count=model.parameter_count,
        train_tokens=len(train_examples),
        validation_tokens=len(validation_examples),
        curve=tuple(curve),
        final_params=params,
    )
