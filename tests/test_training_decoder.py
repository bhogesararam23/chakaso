"""Tests for the decoder architecture: causality, the cache, padding and initialization.

The properties that matter are the ones a measured run silently depends on. A key/value cache that disagrees
with the full forward pass would let a fast generation path produce different text from a slow one; a causal
mask off by one position would let a model train on its own future and report a beautiful loss; a padded batch
that leaks a pad token into attention would corrupt exactly the rows the loss claims to be scoring. Each of
those is tested as a property, at the shape where it would appear, rather than as a golden number.

Everything here is gated on the optional ``[nn]`` extra.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.decoder import (  # noqa: E402
    IGNORE_INDEX,
    ChakasoDecoder,
    DecoderConfig,
    KeyValueCache,
    apply_rotary,
    mask_positions,
    rope_angles,
)
from chakaso.training.errors import TrainingError  # noqa: E402

CONFIG = {
    "vocab_size": 61,
    "context_length": 16,
    "d_model": 8,
    "num_layers": 2,
    "num_heads": 2,
    "feedforward_dim": 16,
}


def _config(**overrides: object) -> DecoderConfig:
    return DecoderConfig(**{**CONFIG, **overrides})  # type: ignore[arg-type]


def _model(seed: int = 0, **overrides: object) -> ChakasoDecoder:
    torch.manual_seed(seed)
    return ChakasoDecoder(_config(**overrides))


def _identifiers(rows: int, length: int, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, CONFIG["vocab_size"], (rows, length), generator=generator)


def test_a_decoder_configuration_refuses_a_head_width_that_is_not_an_integer() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        _config(num_heads=3)


def test_a_decoder_configuration_refuses_an_odd_head_width() -> None:
    with pytest.raises(ValueError, match="must be even"):
        _config(d_model=9, num_heads=3)


def test_a_decoder_configuration_refuses_a_feedforward_that_does_not_expand() -> None:
    with pytest.raises(ValueError, match="feed-forward expansion"):
        _config(feedforward_dim=4)


@pytest.mark.parametrize(
    "field", ["vocab_size", "context_length", "d_model", "num_layers", "num_heads"]
)
def test_a_decoder_configuration_refuses_a_non_positive_structural_field(field: str) -> None:
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        _config(**{field: 0})


def test_a_decoder_configuration_refuses_a_dropout_that_is_not_a_probability() -> None:
    with pytest.raises(ValueError, match="dropout must be in"):
        _config(dropout=1.0)


def test_a_decoder_configuration_refuses_a_rotary_base_that_cannot_scale() -> None:
    with pytest.raises(ValueError, match="rope_theta must exceed"):
        _config(rope_theta=1.0)


def test_a_stored_configuration_round_trips_and_refuses_unknown_fields() -> None:
    config = _config()

    assert DecoderConfig.from_mapping(config.as_dict()) == config
    with pytest.raises(ValueError, match="unknown fields"):
        DecoderConfig.from_mapping({**config.as_dict(), "mixture_of_experts": 8})
    with pytest.raises(ValueError, match="missing fields"):
        DecoderConfig.from_mapping({"vocab_size": 61})


def test_the_arithmetic_parameter_count_is_the_built_model() -> None:
    config = _config()

    model = ChakasoDecoder(config)

    assert model.parameter_count == config.parameter_count
    assert model.parameter_count > 0


def test_untying_the_head_adds_exactly_a_second_embedding() -> None:
    tied = _config()
    untied = _config(tie_embeddings=False)

    assert untied.parameter_count - tied.parameter_count == tied.vocab_size * tied.d_model
    assert ChakasoDecoder(untied).parameter_count == untied.parameter_count


def test_rotary_angles_depend_only_on_the_difference_between_positions() -> None:
    pairs = 2
    left = rope_angles(torch.tensor([[4, 5]]), pairs)
    right = rope_angles(torch.tensor([[9, 10]]), pairs)
    query = torch.ones(1, 1, 1, pairs * 2)
    key = torch.arange(1, pairs * 2 + 1, dtype=torch.float32).unsqueeze(0).reshape(1, 1, 1, -1)

    # A dot product between two rotated vectors at the same relative offset is the same number at offset
    # (4,5) and at (9,10) — the property that makes a rotary model able to use a longer context.
    first = (apply_rotary(query, left[:, :1]) * apply_rotary(key, left[:, 1:])).sum()
    second = (apply_rotary(query, right[:, :1]) * apply_rotary(key, right[:, 1:])).sum()

    assert torch.allclose(first, second, atol=1e-5)


def test_rotary_angles_refuse_an_odd_head_and_a_mismatched_length() -> None:
    with pytest.raises(ValueError, match="odd"):
        apply_rotary(torch.zeros(1, 1, 2, 3), rope_angles(torch.arange(2).unsqueeze(0), 1))
    with pytest.raises(ValueError, match="position"):
        apply_rotary(torch.zeros(1, 1, 4, 4), rope_angles(torch.arange(2).unsqueeze(0), 2))
    with pytest.raises(ValueError, match="negative"):
        rope_angles(torch.tensor([[-1, 0]]), 2)


def test_the_causal_mask_offsets_a_stepped_query_against_a_longer_cached_past() -> None:
    mask = mask_positions(1, 4, offset=3)

    assert torch.isfinite(mask).all()
    square = mask_positions(4, 4)
    assert torch.isneginf(square[0, 1:]).all()
    assert torch.isfinite(square[3]).all()
    with pytest.raises(ValueError, match="negative"):
        mask_positions(2, 2, offset=-1)


def test_a_later_token_can_never_change_an_earlier_logit() -> None:
    model = _model().eval()
    identifiers = _identifiers(1, 8)
    changed = identifiers.clone()
    changed[0, 6] = (changed[0, 6] + 7) % CONFIG["vocab_size"]

    with torch.no_grad():
        before, _ = model(identifiers)
        after, _ = model(changed)

    assert torch.allclose(before[0, :6], after[0, :6], atol=1e-6)
    assert not torch.allclose(before[0, 6], after[0, 6], atol=1e-6)


def test_stepped_decoding_with_a_cache_equals_the_full_forward_pass() -> None:
    config = _config()
    model = _model().eval()
    identifiers = _identifiers(3, 6)
    cache = KeyValueCache(config.num_layers)

    with torch.no_grad():
        full, _ = model(identifiers)
        stepped = []
        for index in range(identifiers.shape[1]):
            logits, _ = model(identifiers[:, index : index + 1], cache=cache)
            stepped.append(logits[:, 0, :])

    assert cache.length == identifiers.shape[1]
    assert torch.allclose(full, torch.stack(stepped, dim=1), atol=1e-6)


def test_a_cache_of_the_wrong_depth_is_refused() -> None:
    model = _model()

    with pytest.raises(ValueError, match="holds 1 layer"):
        model(_identifiers(1, 2), cache=KeyValueCache(1))


def test_a_cache_needs_at_least_one_layer() -> None:
    with pytest.raises(TrainingError, match="at least one layer"):
        KeyValueCache(0)


def test_a_left_padded_row_scores_exactly_as_it_did_unpadded() -> None:
    """Padding on the *past* side is the case causality cannot hide, and the mask is what fixes it.

    Two rows of different lengths share one batch only if the shorter one is padded. Its pad keys sit
    before its own real tokens, so the causal triangle still lets them attend to the padding — which is why
    ``valid`` exists, and why the control below asserts that masking is the difference, not a coincidence.
    """
    model = _model().eval()
    short = _identifiers(1, 4)
    targets = torch.randint(
        1, CONFIG["vocab_size"], (1, 4), generator=torch.Generator().manual_seed(1)
    )
    # The same four tokens, positioned as if they began at 2, and the same four with two pad identifiers
    # (identifier 0) in front of them, ignored by the loss and masked out of attention.
    padded_ids = torch.cat((torch.zeros((1, 2), dtype=torch.long), short), dim=1)
    padded_targets = torch.cat((torch.full((1, 2), IGNORE_INDEX, dtype=torch.long), targets), dim=1)
    padded_positions = torch.arange(6).unsqueeze(0)

    with torch.no_grad():
        _, own = model(short, targets, positions=torch.arange(2, 6).unsqueeze(0))
        _, masked = model(
            padded_ids,
            padded_targets,
            positions=padded_positions,
            valid=torch.tensor([[False, False, True, True, True, True]]),
        )
        _, unmasked = model(
            padded_ids,
            padded_targets,
            positions=padded_positions,
            valid=torch.tensor([[True, True, True, True, True, True]]),
        )

    assert own is not None and masked is not None and unmasked is not None
    assert torch.allclose(masked, own, atol=1e-6)
    assert not torch.allclose(unmasked, own, atol=1e-6)


def test_two_rows_of_different_lengths_share_a_batch_without_disturbing_each_other() -> None:
    model = _model().eval()
    long = _identifiers(1, 6, seed=4)
    short = _identifiers(1, 4, seed=5)
    targets_long = torch.randint(
        1, CONFIG["vocab_size"], (1, 6), generator=torch.Generator().manual_seed(6)
    )
    targets_short = torch.randint(
        1, CONFIG["vocab_size"], (1, 4), generator=torch.Generator().manual_seed(7)
    )

    batched_ids = torch.cat((torch.zeros((1, 2), dtype=torch.long), short), dim=1)
    batched = torch.cat((long, batched_ids), dim=0)
    batched_targets = torch.cat(
        (
            targets_long,
            torch.cat((torch.full((1, 2), IGNORE_INDEX, dtype=torch.long), targets_short), dim=1),
        ),
        dim=0,
    )
    valid = torch.tensor([[True] * 6, [False, False] + [True] * 4])
    positions = torch.arange(6).unsqueeze(0).expand(2, 6)

    with torch.no_grad():
        _, together = model(batched, batched_targets, positions=positions, valid=valid)
        _, alone_long = model(long, targets_long)
        _, alone_short = model(short, targets_short, positions=torch.arange(2, 6).unsqueeze(0))

    # Each row's own mean over its own unignored positions; the batch mean is their position-weighted
    # combination, so equality here means neither row saw the other's padding.
    combined = (float(alone_long) * 6 + float(alone_short) * 4) / 10
    assert together is not None
    assert abs(float(together) - combined) < 1e-5


def test_a_batch_with_nothing_to_predict_is_refused_rather_than_returning_a_nan() -> None:
    model = _model()
    identifiers = _identifiers(2, 3)
    all_padding = torch.full((2, 3), IGNORE_INDEX, dtype=torch.long)

    with torch.no_grad(), pytest.raises(ValueError, match="mean over zero positions"):
        model(identifiers, all_padding)


def test_a_key_validity_mask_of_the_wrong_shape_or_dtype_is_refused() -> None:
    model = _model()
    identifiers = _identifiers(1, 3)

    with pytest.raises(ValueError, match="key-validity mask must have shape"):
        model(identifiers, valid=torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="must be boolean"):
        model(identifiers, valid=torch.ones(1, 3, dtype=torch.float32))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vocab_size": 10}, "outside the vocabulary"),
        ({"context_length": 2}, "past the configured context"),
    ],
)
def test_input_outside_the_declared_envelope_is_refused(
    overrides: dict[str, int], message: str
) -> None:
    model = _model(**overrides).eval()
    identifiers = _identifiers(1, 4)

    with torch.no_grad(), pytest.raises(ValueError, match=message):
        model(identifiers)


def test_ragged_shapes_and_empty_batches_are_refused() -> None:
    model = _model()

    with pytest.raises(ValueError, match="2-dimensional"):
        model(torch.zeros(4, dtype=torch.long))
    with pytest.raises(ValueError, match="at least one sequence"):
        model(torch.zeros(1, 0, dtype=torch.long))
    with pytest.raises(ValueError, match="positions must have shape"):
        model(_identifiers(1, 3), positions=torch.zeros((1, 2), dtype=torch.long))
    with pytest.raises(ValueError, match="targets must have shape"):
        model(_identifiers(1, 3), targets=torch.zeros((1, 4), dtype=torch.long))


def test_negative_positions_are_refused_before_any_arithmetic() -> None:
    model = _model()

    with pytest.raises(ValueError, match="position is negative"):
        model(_identifiers(1, 3), positions=torch.tensor([[-1, 0, 1]]))


def test_initialization_is_seeded_and_every_parameter_is_finite() -> None:
    first = _model(seed=5)
    second = _model(seed=5)
    third = _model(seed=6)

    assert all(
        torch.equal(a, b)
        for a, b in zip(first.state_dict().values(), second.state_dict().values(), strict=True)
    )
    assert any(
        not torch.equal(a, b)
        for a, b in zip(first.state_dict().values(), third.state_dict().values(), strict=True)
    )
    assert torch.isfinite(torch.cat([p.flatten() for p in first.parameters()])).all()


def test_the_residual_branches_are_scaled_down_and_the_norms_start_at_one() -> None:
    config = _config()
    model = _model()
    expected = config.initialization_std / math.sqrt(2.0 * config.num_layers)

    block = model.blocks[0]
    attention_output_std = float(block.attention.output.weight.data.std())
    feedforward_down_std = float(block.feedforward.down.weight.data.std())
    query_std = float(block.attention.query.weight.data.std())

    assert attention_output_std < query_std
    assert feedforward_down_std < query_std
    assert abs(attention_output_std - expected) < expected
    assert torch.equal(block.attention_norm.weight, torch.ones(config.d_model))
    assert torch.equal(model.final_norm.weight, torch.ones(config.d_model))
    assert block.feedforward.up.bias is not None and torch.equal(
        block.feedforward.up.bias, torch.zeros(config.feedforward_dim)
    )
    assert block.attention.query.bias is None


def test_the_head_shares_the_embedding_matrix_when_tied() -> None:
    model = _model()

    assert model.head.weight is model.token_embedding.weight


def test_a_fresh_decoder_scores_close_to_uniform_and_reports_the_loss_in_nats() -> None:
    config = _config()
    model = _model().eval()
    identifiers = _identifiers(2, 5)
    targets = torch.randint(
        1, config.vocab_size, (2, 5), generator=torch.Generator().manual_seed(7)
    )

    with torch.no_grad():
        _, loss = model(identifiers, targets)

    # At initialization the head is near zero, so the cross-entropy should sit close to ln(vocab_size) —
    # the same arithmetic the uniform baseline in every experiment record is defined by.
    assert loss is not None
    assert abs(float(loss) - math.log(config.vocab_size)) < 1.0


def test_dropout_makes_training_mode_differ_and_evaluation_mode_deterministic() -> None:
    model = _model(dropout=0.4)
    identifiers = _identifiers(1, 6)

    model.train()
    torch.manual_seed(11)
    first, _ = model(identifiers)
    torch.manual_seed(12)
    second, _ = model(identifiers)
    model.eval()
    with torch.no_grad():
        stable_a, _ = model(identifiers)
        stable_b, _ = model(identifiers)

    assert not torch.allclose(first, second, atol=1e-6)
    assert torch.equal(stable_a, stable_b)


def test_next_token_logits_is_the_last_row_of_the_full_pass() -> None:
    model = _model().eval()
    identifiers = _identifiers(1, 4)

    with torch.no_grad():
        full, _ = model(identifiers)
        stepped = model.next_token_logits(identifiers)

    assert torch.equal(stepped, full[:, -1, :])


def test_a_row_of_identifier_rows_must_be_rectangular() -> None:
    with pytest.raises(ValueError, match="same length"):
        ChakasoDecoder.stack_ids([[1, 2, 3], [1, 2]])
    assert ChakasoDecoder.stack_ids([[1, 2], [3, 4]]).dtype == torch.int64


def test_parameters_are_float32_on_cpu_because_that_is_what_this_project_verifies() -> None:
    model = _model()

    assert model.device.type == "cpu"
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_document_masking_stops_attention_into_a_previous_document() -> None:
    model = _model(seed=0).eval()
    identifiers = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.int64)
    segments = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int64)
    changed = identifiers.clone()
    changed[0, 0] = 7

    with torch.no_grad():
        masked, _ = model(identifiers, segment_ids=segments)
        masked_changed, _ = model(changed, segment_ids=segments)
        unmasked, _ = model(identifiers)
        unmasked_changed, _ = model(changed)

    # Positions 3-5 belong to the second document; nothing before position 3 may influence them.
    assert torch.equal(masked[:, 3:], masked_changed[:, 3:])
    # Without the mask the same change propagates through attention, which is what the mask prevents.
    assert not torch.allclose(unmasked[:, 3:], unmasked_changed[:, 3:])


def test_document_masking_of_one_document_is_the_causal_mask() -> None:
    model = _model(seed=0).eval()
    identifiers = _identifiers(2, 6)
    segments = torch.zeros_like(identifiers)

    with torch.no_grad():
        masked, _ = model(identifiers, segment_ids=segments)
        plain, _ = model(identifiers)

    assert torch.equal(masked, plain)


def test_document_masking_is_refused_with_a_cache_or_a_malformed_segment_tensor() -> None:
    model = _model().eval()
    identifiers = _identifiers(1, 4)
    cache = KeyValueCache(len(model.blocks))

    with pytest.raises(ValueError, match="key/value cache"):
        model(identifiers, cache=cache, segment_ids=torch.zeros_like(identifiers))
    with pytest.raises(ValueError, match="shape"):
        model(identifiers, segment_ids=torch.zeros((1, 3), dtype=torch.int64))
    with pytest.raises(ValueError, match="integers"):
        model(identifiers, segment_ids=torch.zeros((1, 4), dtype=torch.float32))
    with pytest.raises(ValueError, match="non-negative"):
        model(identifiers, segment_ids=torch.tensor([[-1, 0, 0, 0]], dtype=torch.int64))
