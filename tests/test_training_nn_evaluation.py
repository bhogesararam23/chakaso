"""Torch-gated tests for held-out Transformer evaluation.

The measurement side of a result is where a number quietly becomes wrong, so these tests check the series
that feeds the arithmetic: one entry per scored position, in stream order, with the separator scored but
not counted as text — and the same value the loop's own evaluation produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from chakaso.training.data import TextUnit  # noqa: E402
from chakaso.training.errors import (  # noqa: E402
    TrainingError,
    TrainingScheduleError,
)
from chakaso.training.nn_evaluation import (  # noqa: E402
    evaluate_artifact,
    evaluate_model,
    score_model_on_examples,
)
from chakaso.training.nn_training import evaluate_loss  # noqa: E402
from chakaso.training.packing import pack_units  # noqa: E402
from chakaso.training.subword import DOCUMENT_END, train_subword_vocabulary  # noqa: E402
from chakaso.training.transformer import CausalTransformer, TransformerConfig  # noqa: E402

_VOCABULARY = train_subword_vocabulary(
    ["the quick red fox ran over the dry sand and the quiet owl slept"],
    target_size=280,
    min_frequency=2,
)
_CONFIG = TransformerConfig(
    vocab_size=_VOCABULARY.size,
    context_length=6,
    d_model=8,
    num_layers=1,
    num_heads=2,
    feedforward_dim=16,
)


def _model(seed: int = 4) -> CausalTransformer:
    torch.manual_seed(seed)
    model = CausalTransformer(_CONFIG)
    model.eval()
    return model


def _examples(text: str) -> tuple:
    return pack_units(
        (TextUnit(identifier="unit", text=text),), _VOCABULARY, context_length=6
    ).examples


def test_scores_are_one_entry_per_scored_position_in_stream_order() -> None:
    examples = _examples("the quick red fox ran over the dry sand")

    nats, targets, predictions = score_model_on_examples(_model(), examples)

    assert len(nats) == len(targets) == len(predictions) == len(examples) * 6
    assert all(value >= 0.0 for value in nats)
    assert all(0 <= token < _VOCABULARY.size for token in targets + predictions)
    # Every target is a token of the packed stream, in order: the scored series is the stream shifted by
    # one, which is what makes its decoded total the text these predictions describe.
    stream = [token for example in examples for token in (*example.input_ids, example.targets[-1])]
    assert targets == [token for example in examples for token in example.targets]
    assert targets[0] == stream[1]


def test_the_reported_loss_matches_the_loops_own_position_weighted_evaluation() -> None:
    examples = _examples("the quick red fox ran over the dry sand again")
    model = _model()

    nats, targets, _predictions = score_model_on_examples(model, examples)
    metric = evaluate_model(model, examples, _VOCABULARY)
    loop_loss, positions = evaluate_loss(model, examples, batch_size=5)

    assert metric.positions_scored == positions == len(targets)
    assert metric.loss_nats == pytest.approx(loop_loss, abs=1e-5)
    assert metric.loss_nats == pytest.approx(math.fsum(nats) / len(nats), rel=1e-12)
    assert metric.perplexity == pytest.approx(math.exp(metric.loss_nats))


def test_bits_per_character_uses_the_decoded_targets_as_the_denominator() -> None:
    examples = _examples("the quick red fox ran over the dry sand")
    model = _model()

    metric = evaluate_model(model, examples, _VOCABULARY)
    _nats, targets, _predictions = score_model_on_examples(model, examples)

    assert metric.characters_scored == len(_VOCABULARY.decode(targets))
    assert metric.separator_targets == sum(1 for token in targets if token == DOCUMENT_END)
    assert metric.bits_per_character == pytest.approx(
        (metric.loss_nats * metric.positions_scored / math.log(2.0)) / metric.characters_scored
    )
    assert metric.bits_per_character > 0.0


def test_accuracy_is_the_argmax_match_rate_over_scored_positions() -> None:
    examples = _examples("the quick red fox ran")
    model = _model()

    _nats, targets, predictions = score_model_on_examples(model, examples)
    metric = evaluate_model(model, examples, _VOCABULARY)

    expected = sum(
        1 for predicted, target in zip(predictions, targets, strict=True) if predicted == target
    )
    assert metric.accuracy == pytest.approx(expected / len(targets))


def test_evaluating_no_examples_is_refused_rather_than_reported_as_zero() -> None:
    with pytest.raises(TrainingError, match="at least one scored position"):
        evaluate_model(_model(), [], _VOCABULARY)


def test_a_trained_model_scores_lower_on_text_it_has_learned_than_at_random() -> None:
    # The measurement must be able to distinguish a fitted model from an untrained one; otherwise every
    # number in the experiment record is unfalsifiable.
    text = "the quick red fox ran over the dry sand and the quick red fox ran"
    examples = _examples(text)
    untrained = evaluate_model(_model(seed=9), examples, _VOCABULARY)

    fitted = _model(seed=9)
    optimizer = torch.optim.AdamW(fitted.parameters(), lr=5e-2)
    inputs = torch.tensor([list(example.input_ids) for example in examples], dtype=torch.int64)
    targets = torch.tensor([list(example.targets) for example in examples], dtype=torch.int64)
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = fitted(inputs, targets)[1]
        assert loss is not None
        loss.backward()
        optimizer.step()
    trained = evaluate_model(fitted, examples, _VOCABULARY)

    assert trained.loss_nats < untrained.loss_nats
    assert trained.bits_per_character < untrained.bits_per_character
    assert trained.positions_scored == untrained.positions_scored


def test_an_artifact_is_scored_with_the_tokenizer_bound_inside_it(tmp_path: Path) -> None:
    from chakaso.training.nn_checkpoint import ModelArtifact, save_artifact

    model = _model()
    artifact = ModelArtifact(
        config=_CONFIG,
        state={name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
        vocabulary=_VOCABULARY,
        corpus_fingerprint="eval-fixture",
    )
    target = tmp_path / "artifact.pt"
    save_artifact(artifact, target)
    examples = _examples("the quiet owl slept over the dry sand")

    from chakaso.training.nn_checkpoint import load_artifact

    reloaded = load_artifact(target)

    assert evaluate_artifact(reloaded, examples).loss_nats == pytest.approx(
        evaluate_model(model, examples, _VOCABULARY).loss_nats, abs=1e-9
    )


@dataclass(frozen=True, slots=True)
class _Row:
    """A packed example of a chosen width, for the ragged-batch case packing never produces."""

    input_ids: tuple[int, ...]
    targets: tuple[int, ...]


def _float32_ulp(value: float) -> float:
    """The spacing of the format the model computes in, near ``value``.

    A float32 mantissa carries 24 bits, so one unit in the last place at an exponent of ``e`` is
    ``2 ** (e - 23)``. This is the only honest way to state how far two arithmetic paths through the same
    model may sit apart: the numbers themselves are float64 by the time they reach the assertion, and their
    precision was decided inside torch.
    """
    return math.ldexp(1.0, math.floor(math.log2(value)) - 23)


def test_batched_scoring_returns_the_same_numbers_row_by_row() -> None:
    """Batching is a speed change: the same model over the same examples agrees to float32 rounding.

    Not bit-for-bit, and that was the bug in this assertion rather than the bug in the code. On the machine
    that wrote it the two paths agreed exactly, so the test rounded to nine decimals and passed; on a GitHub
    runner it failed, because a different reduction order in the kernel moves the last bits of a 24-bit
    mantissa and nine decimals is finer than the format the computation is performed in. The bound is
    therefore stated in units of that format — 32 ulps, generously more than reassociation of a 256-wide dot
    product can cost — and anything wider than that is a real disagreement about what the model predicted,
    which is the thing worth catching.
    """
    model = _model()
    examples = _examples("the quick red fox ran over the dry sand and the quiet owl slept")

    one_at_a_time = score_model_on_examples(model, examples, batch_size=1)
    batched = score_model_on_examples(model, examples, batch_size=len(examples))

    tolerance = 32 * _float32_ulp(max([*one_at_a_time[0], *batched[0]]))
    assert [round(value, 9) for value in batched[0]] == pytest.approx(
        [round(value, 9) for value in one_at_a_time[0]], abs=tolerance
    )
    assert batched[1] == one_at_a_time[1]
    assert batched[2] == one_at_a_time[2]


def test_a_ragged_set_of_examples_is_scored_without_padding_any_row() -> None:
    model = _model()
    rows = _examples("the quick red fox ran over the dry sand and the quiet owl slept")
    wide = rows[0]
    narrow = _Row(input_ids=wide.input_ids[:4], targets=wide.targets[:4])
    narrow_nats, narrow_targets, _ = score_model_on_examples(model, [narrow], batch_size=8)

    together, targets, predictions = score_model_on_examples(
        model, [wide, narrow, rows[1]], batch_size=8
    )

    assert len(narrow_nats) == 4
    assert targets[: len(wide.targets)] == list(wide.targets)
    assert targets[len(wide.targets) :][:4] == list(narrow_targets)
    assert len(together) == len(wide.targets) + 4 + len(rows[1].targets)
    assert len(predictions) == len(together)


def test_an_unusable_batch_size_is_refused() -> None:
    model = _model()
    examples = _examples("the quick red fox ran over the dry sand")

    with pytest.raises(TrainingScheduleError, match="batch_size"):
        score_model_on_examples(model, examples, batch_size=0)
    with pytest.raises(TrainingScheduleError, match="zero examples"):
        score_model_on_examples(model, [], batch_size=8)
