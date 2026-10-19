"""Tests for the training loop: it learns, it reproduces, and it refuses degenerate data.

Runs the real loop over the experiment corpus on a tiny budget. The central assertion is honest
learning: the held-out validation loss must be lower at the end than at the start, measured on the
split's held-out examples — not on the training examples. Numpy-gated: skipped in the dependency-free
core matrix, executed where ``numpy`` is installed.
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from chakaso.training.config import TrainingConfig  # noqa: E402
from chakaso.training.corpus import CORPUS  # noqa: E402
from chakaso.training.data import Split, split_corpus  # noqa: E402
from chakaso.training.loop import run_training  # noqa: E402


def _config(**overrides: object) -> TrainingConfig:
    base = {
        "context_size": 4,
        "embed_dim": 8,
        "hidden_dim": 16,
        "learning_rate": 0.05,
        "seed": 13,
        "steps": 120,
        "eval_every": 20,
    }
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


def _split() -> Split:
    return split_corpus(CORPUS.text, validation_fraction=0.2, seed=13)


def test_training_reduces_both_losses_below_their_initial_values() -> None:
    result = run_training(_split(), _config())

    assert result.final_train_loss < result.initial_loss
    assert result.final_validation_loss < result.curve[0].validation_loss


def test_the_curve_records_every_evaluation_point_and_the_final_step() -> None:
    config = _config(steps=120, eval_every=20)
    result = run_training(_split(), config)

    steps = [point.step for point in result.curve]
    # step 0 plus one point per eval boundary plus the final step.
    assert steps[0] == 0
    assert steps[-1] == config.steps
    assert set(steps) >= {20, 40, 60, 80, 100, 120}


def test_a_run_is_reproducible_from_the_seed() -> None:
    split = _split()
    config = _config()

    first = run_training(split, config)
    second = run_training(split, config)

    assert [p.train_loss for p in first.curve] == [p.train_loss for p in second.curve]
    assert first.final_validation_loss == second.final_validation_loss
    assert numpy.array_equal(first.final_params.embedding, second.final_params.embedding)


def test_the_result_reports_model_and_data_sizes() -> None:
    result = run_training(_split(), _config())

    assert result.parameter_count > 0
    assert result.train_tokens > 0
    assert result.validation_tokens > 0
    assert result.vocab_fingerprint
    assert result.dataset_digest


def test_a_split_with_no_windows_is_refused_not_reported_as_zero_loss() -> None:
    # A training text too short to form any window (a single character) yields no examples.
    empty_train = Split(
        train_text="a", validation_text="a b c", train_units=("a",), validation_units=("b c",)
    )
    with pytest.raises(ValueError, match="no causal windows"):
        run_training(empty_train, _config())
