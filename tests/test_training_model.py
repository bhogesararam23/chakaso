"""Tests for the tiny character model's forward pass, shapes and validation.

The numerical layer needs numpy, so the whole module is gated: it is skipped in the dependency-free
core matrix and runs in the optional `[train]` job (ADR-0030). These cover structure and a loss value
computed by hand; the gradient itself is verified against finite differences in
``test_training_gradient.py``.
"""

from __future__ import annotations

import math

import pytest

numpy = pytest.importorskip("numpy")

from chakaso.training.model import CharCausalMLP, Params, batches_from_examples  # noqa: E402


def _zero_params(vocab_size: int, context_size: int, embed_dim: int, hidden_dim: int) -> Params:
    input_dim = context_size * embed_dim
    return Params(
        embedding=numpy.zeros((vocab_size, embed_dim)),
        hidden_weight=numpy.zeros((input_dim, hidden_dim)),
        hidden_bias=numpy.zeros(hidden_dim),
        output_weight=numpy.zeros((hidden_dim, vocab_size)),
        output_bias=numpy.zeros(vocab_size),
    )


def test_forward_loss_matches_a_hand_computed_value() -> None:
    # With every parameter zero except nothing, the logits are all zero, so the softmax is uniform over
    # the vocabulary and cross-entropy is exactly log(V). For V=2 that is ln(2) — known analytically.
    model = CharCausalMLP(vocab_size=2, context_size=1, embed_dim=1, hidden_dim=1)
    params = _zero_params(2, 1, 1, 1)
    contexts = numpy.array([[1]], dtype=numpy.int64)
    targets = numpy.array([1], dtype=numpy.int64)

    loss = model.forward(params, contexts, targets)
    assert loss == pytest.approx(math.log(2), abs=1e-9)


def test_forward_loss_matches_a_weighted_hand_computed_case() -> None:
    # Set the output bias so logits are [0, ln(3)]; softmax is [0.25, 0.75] and the loss for target 1
    # is -ln(0.75). Everything else zero keeps the hand computation exact.
    model = CharCausalMLP(vocab_size=2, context_size=1, embed_dim=1, hidden_dim=1)
    params = _zero_params(2, 1, 1, 1)
    params = Params(
        embedding=params.embedding,
        hidden_weight=params.hidden_weight,
        hidden_bias=params.hidden_bias,
        output_weight=params.output_weight,
        output_bias=numpy.array([0.0, math.log(3.0)]),
    )
    contexts = numpy.array([[0]], dtype=numpy.int64)
    targets = numpy.array([1], dtype=numpy.int64)

    loss = model.forward(params, contexts, targets)
    assert loss == pytest.approx(-math.log(0.75), abs=1e-9)


def test_gradient_shapes_match_the_parameters() -> None:
    model = CharCausalMLP(vocab_size=5, context_size=3, embed_dim=2, hidden_dim=4)
    params = model.initialize(numpy.random.default_rng(0))
    contexts = numpy.array([[1, 2, 3], [4, 0, 1]], dtype=numpy.int64)
    targets = numpy.array([2, 4], dtype=numpy.int64)

    grads = model.gradient(params, contexts, targets)

    assert grads.embedding.shape == params.embedding.shape
    assert grads.hidden_weight.shape == params.hidden_weight.shape
    assert grads.hidden_bias.shape == params.hidden_bias.shape
    assert grads.output_weight.shape == params.output_weight.shape
    assert grads.output_bias.shape == params.output_bias.shape


def test_gradients_are_finite_for_a_realistic_batch() -> None:
    model = CharCausalMLP(vocab_size=8, context_size=4, embed_dim=3, hidden_dim=6)
    params = model.initialize(numpy.random.default_rng(1))
    contexts = numpy.random.default_rng(2).integers(0, 8, size=(16, 4)).astype(numpy.int64)
    targets = numpy.random.default_rng(3).integers(0, 8, size=16).astype(numpy.int64)

    grads = model.gradient(params, contexts, targets)

    for _, array in grads.items():
        assert numpy.all(numpy.isfinite(array))


def test_initialization_is_seeded_and_reproducible() -> None:
    model = CharCausalMLP(vocab_size=6, context_size=2, embed_dim=3, hidden_dim=4)

    first = model.initialize(numpy.random.default_rng(42))
    second = model.initialize(numpy.random.default_rng(42))
    different = model.initialize(numpy.random.default_rng(43))

    for (_, a), (_, b) in zip(first.items(), second.items(), strict=True):
        assert numpy.array_equal(a, b)
    assert not numpy.array_equal(first.embedding, different.embedding)


def test_parameter_count_matches_the_sum_of_array_sizes() -> None:
    model = CharCausalMLP(vocab_size=6, context_size=2, embed_dim=3, hidden_dim=4)
    params = model.initialize(numpy.random.default_rng(0))

    total = sum(array.size for _, array in params.items())
    assert model.parameter_count == total


def test_batches_from_examples_preserves_order_and_width() -> None:
    from chakaso.training.data import Example

    examples = (Example(context=(1, 2), target=3), Example(context=(4, 0), target=1))
    contexts, targets = batches_from_examples(examples, context_size=2)

    assert contexts.tolist() == [[1, 2], [4, 0]]
    assert targets.tolist() == [3, 1]


def test_an_empty_batch_is_refused() -> None:
    model = CharCausalMLP(vocab_size=3, context_size=2, embed_dim=1, hidden_dim=1)
    params = _zero_params(3, 2, 1, 1)

    with pytest.raises(ValueError, match="at least one"):
        model.forward(
            params, numpy.zeros((0, 2), dtype=numpy.int64), numpy.zeros(0, dtype=numpy.int64)
        )


def test_an_out_of_range_identifier_is_refused_rather_than_indexing_silently() -> None:
    model = CharCausalMLP(vocab_size=3, context_size=2, embed_dim=1, hidden_dim=1)
    params = _zero_params(3, 2, 1, 1)

    with pytest.raises(ValueError, match="outside the vocabulary"):
        model.forward(
            params, numpy.array([[1, 99]], dtype=numpy.int64), numpy.array([0], dtype=numpy.int64)
        )


def test_a_wrong_context_width_is_refused() -> None:
    model = CharCausalMLP(vocab_size=3, context_size=2, embed_dim=1, hidden_dim=1)
    params = _zero_params(3, 2, 1, 1)

    with pytest.raises(ValueError, match="shape"):
        model.forward(
            params, numpy.array([[1, 2, 3]], dtype=numpy.int64), numpy.array([0], dtype=numpy.int64)
        )


def test_a_target_count_mismatch_is_refused() -> None:
    model = CharCausalMLP(vocab_size=3, context_size=2, embed_dim=1, hidden_dim=1)
    params = _zero_params(3, 2, 1, 1)

    with pytest.raises(ValueError, match="length"):
        model.forward(
            params,
            numpy.array([[1, 2], [0, 1]], dtype=numpy.int64),
            numpy.array([0], dtype=numpy.int64),
        )


def test_an_invalid_architecture_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        CharCausalMLP(vocab_size=0, context_size=2, embed_dim=1, hidden_dim=1)
    with pytest.raises(ValueError, match=">= 1"):
        CharCausalMLP(
            vocab_size=3, context_size=2, embed_dim=True, hidden_dim=1
        )  # bool is rejected
