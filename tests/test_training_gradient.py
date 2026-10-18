"""Verify the hand-derived gradients against finite differences, and the optimizer's behavior.

This is the test that makes the first experiment's learning loop believable rather than merely
plausible: it checks that the analytic gradient the model returns equals a central-difference estimate
of the loss for *every* scalar parameter, on a small but non-trivial model. If the backward pass were
wrong anywhere — a transposed term, a mis-scattered embedding update, a forgotten ``1/batch`` — this
fails. It runs only where numpy is installed (the optional ``[train]`` path).
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from chakaso.training.model import CharCausalMLP, Params  # noqa: E402
from chakaso.training.optimizer import Adam  # noqa: E402

_H = 1e-5


def _model_and_batch() -> tuple[CharCausalMLP, Params, numpy.ndarray, numpy.ndarray]:
    model = CharCausalMLP(vocab_size=4, context_size=2, embed_dim=3, hidden_dim=5)
    params = model.initialize(numpy.random.default_rng(11))
    contexts = numpy.random.default_rng(12).integers(0, 4, size=(9, 2)).astype(numpy.int64)
    targets = numpy.random.default_rng(13).integers(0, 4, size=9).astype(numpy.int64)
    return model, params, contexts, targets


def _replace_entry(params: Params, name: str, index: tuple[int, ...], delta: float) -> Params:
    arrays: dict[str, numpy.ndarray] = {}
    for key, array in params.items():
        if key == name:
            changed = array.copy()
            changed[index] += delta
            arrays[key] = changed
        else:
            arrays[key] = array
    return Params(**arrays)


@pytest.mark.parametrize(
    "name", ["embedding", "hidden_weight", "hidden_bias", "output_weight", "output_bias"]
)
def test_analytic_gradient_matches_central_differences(name: str) -> None:
    model, params, contexts, targets = _model_and_batch()
    analytic = model.gradient(params, contexts, targets)
    analytic_array = dict(analytic.items())[name]
    base_array = dict(params.items())[name]

    for index, _ in numpy.ndenumerate(base_array):
        forward = model.forward(_replace_entry(params, name, index, _H), contexts, targets)
        backward = model.forward(_replace_entry(params, name, index, -_H), contexts, targets)
        numeric = (forward - backward) / (2 * _H)

        assert analytic_array[index] == pytest.approx(numeric, abs=1e-6), (
            f"{name}{index}: analytic {analytic_array[index]!r} != finite-difference {numeric!r}"
        )


def test_a_single_optimizer_step_changes_every_parameter() -> None:
    model, params, contexts, targets = _model_and_batch()
    grads = model.gradient(params, contexts, targets)

    updated = Adam(learning_rate=0.1).step(params, grads)

    for (name, before), (_, after) in zip(params.items(), updated.items(), strict=True):
        assert not numpy.array_equal(before, after), f"{name} did not move"


def test_training_reduces_loss_on_a_fixed_batch() -> None:
    # A handful of Adam steps on the same batch must lower its loss: this proves the gradient is
    # connected to the objective the optimizer is minimizing, end to end.
    model, params, contexts, targets = _model_and_batch()
    optimizer = Adam(learning_rate=0.2)

    first = model.forward(params, contexts, targets)
    for _ in range(40):
        params = optimizer.step(params, model.gradient(params, contexts, targets))
    last = model.forward(params, contexts, targets)

    assert last < first


def test_optimizer_does_not_mutate_its_inputs() -> None:
    model, params, contexts, targets = _model_and_batch()
    grads = model.gradient(params, contexts, targets)
    snapshot = {name: array.copy() for name, array in params.items()}

    Adam().step(params, grads)

    for name, array in params.items():
        assert numpy.array_equal(array, snapshot[name]), f"{name} was mutated in place"


def test_an_invalid_learning_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        Adam(learning_rate=0.0)
    with pytest.raises(ValueError, match="beta1"):
        Adam(beta1=1.0)
