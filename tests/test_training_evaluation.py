"""Tests for held-out evaluation, and its agreement with the loop and the uniform baseline.

Numpy-gated (``[train]`` path). The important cross-checks: an all-zero model scores exactly the uniform
baseline (so the two code paths agree in nats), and the evaluator reproduces the training loop's own
held-out loss from the public interface.
"""

from __future__ import annotations

import math

import pytest

numpy = pytest.importorskip("numpy")

from chakaso.training.baselines import uniform_next_token_loss  # noqa: E402
from chakaso.training.config import TrainingConfig  # noqa: E402
from chakaso.training.corpus import CORPUS  # noqa: E402
from chakaso.training.data import character_windows, split_corpus  # noqa: E402
from chakaso.training.evaluation import evaluate  # noqa: E402
from chakaso.training.loop import run_training  # noqa: E402
from chakaso.training.model import (  # noqa: E402
    CharCausalMLP,
    Params,
    batches_from_examples,
)


def _zero_params(vocab_size: int, context_size: int, embed_dim: int, hidden_dim: int) -> Params:
    input_dim = context_size * embed_dim
    return Params(
        embedding=numpy.zeros((vocab_size, embed_dim)),
        hidden_weight=numpy.zeros((input_dim, hidden_dim)),
        hidden_bias=numpy.zeros(hidden_dim),
        output_weight=numpy.zeros((hidden_dim, vocab_size)),
        output_bias=numpy.zeros(vocab_size),
    )


def test_an_all_zero_model_scores_exactly_the_uniform_baseline() -> None:
    vocab_size = 5
    model = CharCausalMLP(vocab_size=vocab_size, context_size=2, embed_dim=3, hidden_dim=4)
    params = _zero_params(vocab_size, 2, 3, 4)
    contexts = numpy.array([[1, 2], [3, 4], [0, 1]], dtype=numpy.int64)
    targets = numpy.array([4, 0, 2], dtype=numpy.int64)

    metrics = evaluate(params, model, contexts, targets)

    assert metrics.loss == pytest.approx(uniform_next_token_loss(vocab_size))
    assert metrics.bits_per_char == pytest.approx(math.log2(vocab_size))
    assert metrics.perplexity == pytest.approx(vocab_size)
    assert metrics.token_count == 3


def test_accuracy_is_a_fraction_and_a_perfect_scorer_achieves_one() -> None:
    model = CharCausalMLP(vocab_size=3, context_size=1, embed_dim=2, hidden_dim=2)
    params = model.initialize(numpy.random.default_rng(3))
    # Use the model's own argmax as the "targets" so accuracy must be 1.0 — a real check that the
    # accuracy metric reads the predicted distribution, not a constant.
    contexts = numpy.array([[1], [2], [0], [1]], dtype=numpy.int64)
    probabilities = model.probabilities(params, contexts)
    targets = probabilities.argmax(axis=1)

    metrics = evaluate(params, model, contexts, targets)

    assert metrics.accuracy == pytest.approx(1.0)


def test_evaluation_refuses_an_empty_heldout_set() -> None:
    model = CharCausalMLP(vocab_size=3, context_size=2, embed_dim=2, hidden_dim=2)
    params = _zero_params(3, 2, 2, 2)
    empty_contexts = numpy.zeros((0, 2), dtype=numpy.int64)
    empty_targets = numpy.zeros(0, dtype=numpy.int64)

    with pytest.raises(ValueError, match="empty held-out"):
        evaluate(params, model, empty_contexts, empty_targets)


def test_the_public_evaluator_reproduces_the_loop_heldout_loss() -> None:
    config = TrainingConfig(
        context_size=4,
        embed_dim=8,
        hidden_dim=12,
        learning_rate=0.05,
        seed=5,
        steps=30,
        eval_every=10,
    )
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=5)
    result = run_training(split, config)

    # Rebuild the validation arrays through the public layer and score the final parameters directly.
    from chakaso.training.data import fit_vocabulary

    vocabulary = fit_vocabulary(split)
    validation_examples = character_windows(split.validation_text, vocabulary, config.context_size)
    contexts, targets = batches_from_examples(validation_examples, config.context_size)
    model = CharCausalMLP(
        vocab_size=vocabulary.size,
        context_size=config.context_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
    )
    metrics = evaluate(result.final_params, model, contexts, targets)

    assert metrics.loss == pytest.approx(result.final_validation_loss, rel=1e-9)
    assert metrics.token_count == len(validation_examples)


def test_per_position_entropies_aggregate_to_the_reported_mean_loss() -> None:
    """The character model must be scoreable by the same arithmetic a subword model is scored by.

    ``per_position_cross_entropies`` returns the series ``metrics_from_scores`` consumes; if its entries did
    not average back to the aggregate loss the evaluator already reports, E2605 would be comparing two
    different measurements and calling them one table.
    """
    from chakaso.training.data import fit_vocabulary
    from chakaso.training.evaluation import per_position_cross_entropies

    config = TrainingConfig(
        context_size=3,
        embed_dim=4,
        hidden_dim=5,
        learning_rate=0.05,
        seed=11,
        steps=2,
        eval_every=1,
    )
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=11)
    result = run_training(split, config)

    vocabulary = fit_vocabulary(split)
    examples = character_windows(split.validation_text, vocabulary, config.context_size)
    contexts, targets = batches_from_examples(examples, config.context_size)
    model = CharCausalMLP(
        vocab_size=vocabulary.size,
        context_size=config.context_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
    )

    nats = per_position_cross_entropies(result.final_params, model, contexts, targets)

    assert len(nats) == len(examples) == int(contexts.shape[0])
    assert all(value >= 0.0 for value in nats)
    assert sum(nats) / len(nats) == pytest.approx(
        evaluate(result.final_params, model, contexts, targets).loss, rel=1e-9
    )
