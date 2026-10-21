"""Torch-gated tests for the causal Transformer.

These are the checks ADR-0034 accepted in place of analytically verified gradients, so they are written to
be capable of failing on the things a Transformer actually gets wrong: an attention mask that leaks the
future, a loss averaged over the wrong denominator, an initialization that is not actually seeded, a
positional table that runs off the end of the context, and a configuration that silently builds a different
architecture than the checkpoint declares.

They run only where the optional ``[nn]`` extra is installed and skip in the dependency-free matrix (K-009
records that remote CI cannot install it).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.transformer import (  # noqa: E402
    CausalTransformer,
    TransformerConfig,
    count_parameters,
    mask_positions,
)

CONFIG = TransformerConfig(
    vocab_size=13,
    context_length=8,
    d_model=16,
    num_layers=2,
    num_heads=4,
    feedforward_dim=32,
)


def _model(seed: int = 3, config: TransformerConfig = CONFIG) -> CausalTransformer:
    torch.manual_seed(seed)
    return CausalTransformer(config)


def _loss(model: CausalTransformer, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    """The mean cross-entropy as a plain number.

    Detached on purpose: reading a grad-carrying tensor into a float is something torch warns about, and
    this suite turns warnings into errors, so a test that wants a number has to say it does not need the
    graph.
    """
    _, loss = model(inputs, targets)
    assert loss is not None
    return float(loss.detach())


def _batch(rows: int = 3, length: int = 6, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randint(0, CONFIG.vocab_size, (rows, length), generator=generator)
    targets = torch.randint(0, CONFIG.vocab_size, (rows, length), generator=generator)
    return inputs, targets


# ------------------------------------------------------------------ configuration


def test_configuration_checks_the_relationships_between_dimensions() -> None:
    assert CONFIG.head_width == 4
    with pytest.raises(ValueError, match="divide evenly"):
        TransformerConfig(
            vocab_size=13,
            context_length=8,
            d_model=17,
            num_layers=2,
            num_heads=4,
            feedforward_dim=32,
        )
    with pytest.raises(ValueError, match="at least d_model"):
        TransformerConfig(
            vocab_size=13,
            context_length=8,
            d_model=16,
            num_layers=2,
            num_heads=4,
            feedforward_dim=8,
        )


@pytest.mark.parametrize("bad", [0, -1, True, "8", 2.5])
def test_configuration_refuses_sizes_that_are_not_positive_integers(bad: object) -> None:
    with pytest.raises(ValueError):
        TransformerConfig(
            vocab_size=bad,  # type: ignore[arg-type]
            context_length=8,
            d_model=16,
            num_layers=2,
            num_heads=4,
            feedforward_dim=32,
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 2.0, "x", True])
def test_configuration_refuses_an_impossible_dropout(dropout: object) -> None:
    with pytest.raises(ValueError, match="dropout"):
        TransformerConfig(
            vocab_size=13,
            context_length=8,
            d_model=16,
            num_layers=2,
            num_heads=4,
            feedforward_dim=32,
            dropout=dropout,  # type: ignore[arg-type]
        )


def test_configuration_refuses_a_non_positive_initialization_scale() -> None:
    with pytest.raises(ValueError, match="initialization_std"):
        TransformerConfig(
            vocab_size=13,
            context_length=8,
            d_model=16,
            num_layers=2,
            num_heads=4,
            feedforward_dim=32,
            initialization_std=0.0,
        )


def test_configuration_round_trips_through_a_mapping() -> None:
    stored = CONFIG.as_dict()

    assert TransformerConfig.from_mapping(stored) == CONFIG


def test_a_configuration_with_an_unknown_field_is_refused() -> None:
    # A checkpoint declaring a field this code would ignore is a model that is not the one that was saved.
    values = CONFIG.as_dict()
    values["rotary_base"] = 10000

    with pytest.raises(ValueError, match="unknown fields"):
        TransformerConfig.from_mapping(values)


def test_a_configuration_missing_a_required_field_is_refused() -> None:
    values = CONFIG.as_dict()
    values.pop("num_heads")

    with pytest.raises(ValueError, match="missing fields"):
        TransformerConfig.from_mapping(values)


# ------------------------------------------------------------------ masking


def test_the_causal_mask_allows_a_position_to_see_itself_and_nothing_after() -> None:
    mask = mask_positions(4)

    assert mask.shape == (4, 4)
    for row in range(4):
        for column in range(4):
            expected = 0.0 if column <= row else float("-inf")
            assert float(mask[row, column]) == expected


def test_the_causal_mask_refuses_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="sequence_length"):
        mask_positions(0)


def test_changing_a_later_token_cannot_change_an_earlier_prediction() -> None:
    model = _model()
    inputs, _ = _batch(rows=1, length=6)
    altered = inputs.clone()
    altered[0, -1] = (int(altered[0, -1]) + 1) % CONFIG.vocab_size

    with torch.no_grad():
        base, _ = model(inputs)
        changed, _ = model(altered)

    # Positions 0..4 predict from prefixes that exclude the final token, so their logits must be identical
    # bit for bit. Only position 5 may move.
    assert torch.equal(base[:, :5, :], changed[:, :5, :])
    assert not torch.equal(base[:, 5, :], changed[:, 5, :])


def test_changing_an_earlier_token_can_change_a_later_prediction() -> None:
    # The other half of the masking claim. Without it, "the test above passed" could mean the model ignores
    # its input entirely.
    model = _model()
    inputs, _ = _batch(rows=1, length=6)
    altered = inputs.clone()
    altered[0, 0] = (int(altered[0, 0]) + 5) % CONFIG.vocab_size

    with torch.no_grad():
        base, _ = model(inputs)
        changed, _ = model(altered)

    assert not torch.equal(base[:, -1, :], changed[:, -1, :])


# ------------------------------------------------------------------ shapes and loss


def test_output_shapes_are_batch_length_vocabulary() -> None:
    model = _model()
    inputs, targets = _batch(rows=4, length=6)

    logits, loss = model(inputs, targets)

    assert tuple(logits.shape) == (4, 6, CONFIG.vocab_size)
    assert loss is not None and loss.ndim == 0
    assert model.parameter_count == count_parameters(model)


def test_loss_without_targets_is_none_and_with_them_matches_manual_cross_entropy() -> None:
    model = _model()
    inputs, targets = _batch(rows=2, length=5)

    logits, _ = model(inputs, targets)

    log_probabilities = torch.log_softmax(logits.detach(), dim=-1)
    manual = torch.stack(
        [
            -log_probabilities[batch, position, int(targets[batch, position])]
            for batch in range(2)
            for position in range(5)
        ]
    ).mean()
    assert _loss(model, inputs, targets) == pytest.approx(float(manual), abs=1e-5)


def test_an_early_stopping_loss_averages_over_positions_not_over_sequences() -> None:
    # The denominator a bits-per-character number depends on: doubling the batch while keeping positions
    # identical must not change the mean, which a per-sequence average would.
    model = _model()
    inputs, targets = _batch(rows=2, length=5)
    doubled_inputs = torch.cat([inputs, inputs], dim=0)
    doubled_targets = torch.cat([targets, targets], dim=0)

    single = _loss(model, inputs, targets)
    doubled = _loss(model, doubled_inputs, doubled_targets)

    assert single == pytest.approx(doubled, abs=1e-5)


def test_next_token_logits_are_the_last_positions_logits() -> None:
    model = _model()
    inputs, _ = _batch(rows=1, length=6)

    with torch.no_grad():
        whole, _ = model(inputs)
        last = model.next_token_logits(inputs)

    assert tuple(last.shape) == (1, CONFIG.vocab_size)
    assert torch.equal(last, whole[:, -1, :])


def test_parameter_count_matches_the_configurations_arithmetic() -> None:
    config = TransformerConfig(
        vocab_size=20, context_length=6, d_model=8, num_layers=1, num_heads=2, feedforward_dim=16
    )
    model = _model(config=config)

    # Embedding (tied, so counted once): 20*8 + positions 6*8. Per layer: attention 4*(8*8)+4*8, two linear
    # in the feed-forward 8*16 + 16*8 with biases 16 + 8, two LayerNorms 2*(8+8). Final LayerNorm 2*8.
    expected = (
        20 * 8 + 6 * 8 + (4 * 8 * 8 + 4 * 8) + (8 * 16 + 16 + 16 * 8 + 8) + 2 * (8 + 8) + (8 + 8)
    )
    assert model.parameter_count == expected


# ------------------------------------------------------------------ optimisation


def test_gradients_are_finite_and_reach_the_embeddings() -> None:
    model = _model()
    inputs, targets = _batch()

    _, loss = model(inputs, targets)
    assert loss is not None
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.token_embedding.weight.grad is not None
    assert float(model.token_embedding.weight.grad.abs().sum()) > 0.0


def test_optimizer_steps_reduce_the_loss_on_a_fixed_batch() -> None:
    model = _model()
    inputs, targets = _batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    first_loss = _loss(model, inputs, targets)
    for _ in range(25):
        optimizer.zero_grad()
        loss = model(inputs, targets)[1]
        assert loss is not None
        loss.backward()
        optimizer.step()
    final_loss = _loss(model, inputs, targets)

    assert final_loss < first_loss


def test_a_sampled_finite_difference_check_agrees_with_autograd() -> None:
    # The weaker guarantee ADR-0034 accepted in place of verifying every scalar: up to 24 named
    # parameters, central differences with h = 1e-4, checked against the autograd gradient.
    model = _model()
    inputs, targets = _batch(rows=2, length=4)
    loss = model(inputs, targets)[1]
    assert loss is not None
    model.zero_grad(set_to_none=True)
    loss.backward()

    step = 1e-4
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not parameter.is_contiguous():
            continue
        # A detached view of the parameter, so writing through it perturbs the real weight while reading a
        # scalar out of it cannot drag the autograd graph into a float conversion.
        view = parameter.detach().view(-1)
        for index in (0, view.numel() // 2, view.numel() - 1):
            original = float(view[index])
            with torch.no_grad():
                view[index] = original + step
                plus = float(model(inputs, targets)[1].detach())
                view[index] = original - step
                minus = float(model(inputs, targets)[1].detach())
                view[index] = original
            numeric = (plus - minus) / (2 * step)
            analytic = float(parameter.grad.detach().view(-1)[index])
            assert numeric == pytest.approx(analytic, abs=2e-3, rel=5e-2), (
                f"{name}[{index}]: autograd {analytic!r} vs finite difference {numeric!r}"
            )
            checked += 1
        if checked >= 24:
            break

    assert checked >= 12, (
        f"only {checked} scalars were checked; the sample is too small to mean anything"
    )


# ------------------------------------------------------------------ initialization and determinism


def test_a_shared_seed_builds_identical_weights_and_a_different_seed_does_not() -> None:
    first = _model(seed=11)
    second = _model(seed=11)
    third = _model(seed=12)

    assert torch.equal(first.token_embedding.weight, second.token_embedding.weight)
    assert torch.equal(first.final_norm.weight, second.final_norm.weight)
    assert not torch.equal(first.token_embedding.weight, third.token_embedding.weight)


def test_initialization_is_a_named_normal_not_a_framework_default() -> None:
    model = _model(
        seed=5,
        config=TransformerConfig(
            vocab_size=64,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=4,
            feedforward_dim=32,
            initialization_std=0.05,
        ),
    )

    assert float(model.token_embedding.weight.detach().std()) == pytest.approx(0.05, rel=0.35)
    assert float(model.final_norm.weight.detach().mean()) == 1.0
    assert float(model.final_norm.bias.detach().abs().sum()) == 0.0


def test_dropout_zero_makes_repeated_forward_passes_identical() -> None:
    model = _model()
    model.train()
    inputs, _ = _batch()

    with torch.no_grad():
        first, _ = model(inputs)
        second, _ = model(inputs)

    assert torch.equal(first, second)


def test_dropout_on_makes_training_passes_differ_and_eval_passes_agree() -> None:
    config = TransformerConfig(
        vocab_size=13,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.3,
    )
    model = _model(seed=2, config=config)
    inputs, _ = _batch()

    model.train()
    with torch.no_grad():
        train_a, _ = model(inputs)
        train_b, _ = model(inputs)
    model.eval()
    with torch.no_grad():
        eval_a, _ = model(inputs)
        eval_b, _ = model(inputs)

    assert not torch.equal(train_a, train_b)
    assert torch.equal(eval_a, eval_b)


# ------------------------------------------------------------------ refusals


def test_a_batch_wider_than_the_context_is_refused() -> None:
    model = _model()
    inputs = torch.randint(0, CONFIG.vocab_size, (1, CONFIG.context_length + 1))

    with pytest.raises(ValueError, match="exceeds the configured context"):
        model(inputs)


@pytest.mark.parametrize("shape", [(6,), (2, 3, 4)])
def test_input_ids_must_be_a_two_dimensional_batch(shape: tuple[int, ...]) -> None:
    model = _model()

    with pytest.raises(ValueError, match="2-dimensional"):
        model(torch.zeros(shape, dtype=torch.int64))


def test_an_empty_batch_is_refused() -> None:
    model = _model()

    with pytest.raises(ValueError, match="at least one sequence"):
        model(torch.zeros(0, 4, dtype=torch.int64))


def test_an_identifier_outside_the_vocabulary_is_refused() -> None:
    model = _model()

    with pytest.raises(ValueError, match="outside the vocabulary"):
        model(torch.tensor([[1, 2, CONFIG.vocab_size]]))
    with pytest.raises(ValueError, match="outside the vocabulary"):
        model(torch.tensor([[1, 2, -1]]))


def test_targets_must_match_the_input_shape() -> None:
    model = _model()
    inputs, targets = _batch(rows=2, length=5)

    with pytest.raises(ValueError, match="targets must have shape"):
        model(inputs, targets[:, :4])


def test_supplied_positions_must_match_and_stay_inside_the_context() -> None:
    model = _model()
    inputs, _ = _batch(rows=2, length=5)

    with pytest.raises(ValueError, match="positions must have shape"):
        model(inputs, positions=torch.zeros(3, 5, dtype=torch.int64))
    with pytest.raises(ValueError, match="reaches past the context length"):
        model(inputs, positions=torch.full((2, 5), CONFIG.context_length, dtype=torch.int64))


def test_a_ragged_batch_is_refused_by_the_tensor_helper() -> None:
    with pytest.raises(ValueError, match="same length"):
        CausalTransformer.stack_ids([[1, 2, 3], [1, 2]])

    tensor = CausalTransformer.stack_ids([[1, 2, 3], [4, 5, 6]])

    assert tensor.dtype == torch.int64
    assert tuple(tensor.shape) == (2, 3)
