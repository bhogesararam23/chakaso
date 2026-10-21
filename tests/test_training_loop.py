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
from chakaso.training.data import (  # noqa: E402
    Split,
    character_windows,
    fit_vocabulary,
    split_corpus,
)
from chakaso.training.loop import run_minibatch_training, run_training  # noqa: E402


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


# ---------------------------------------------------------------------------
# Mini-batch training: the shape a real corpus needs.
# ---------------------------------------------------------------------------

def _mini_config(**settings: object) -> TrainingConfig:
    values: dict[str, object] = {
        "context_size": 4,
        "embed_dim": 8,
        "hidden_dim": 12,
        "learning_rate": 0.02,
        "seed": 7,
        "steps": 20,
        "eval_every": 10,
    }
    values.update(settings)
    return TrainingConfig(**values)  # type: ignore[arg-type]


def test_minibatch_training_records_a_curve_on_both_losses() -> None:
    config = _mini_config()
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=config.seed)

    result = run_minibatch_training(
        split, config, batch_size=32, max_steps=60, eval_every=20, seed=3
    )

    assert [point.step for point in result.curve] == [0, 20, 40, 60]
    assert result.train_tokens == len(
        character_windows(split.train_text, fit_vocabulary(split), config.context_size)
    )
    assert result.validation_tokens > 0
    assert result.parameter_count > 0
    assert result.initial_loss > 0.0


def test_minibatch_training_lowers_heldout_loss() -> None:
    config = _mini_config()
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=config.seed)

    result = run_minibatch_training(
        split, config, batch_size=32, max_steps=300, eval_every=50, seed=3
    )

    assert result.final_validation_loss < result.initial_loss, (
        f"held-out loss did not fall: {result.initial_loss:.4f} -> {result.final_validation_loss:.4f}"
    )


def test_the_schedule_is_reproducible_and_the_shuffle_seed_changes_it() -> None:
    config = _mini_config()
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=config.seed)

    first = run_minibatch_training(split, config, batch_size=16, max_steps=40, eval_every=20, seed=11)
    second = run_minibatch_training(split, config, batch_size=16, max_steps=40, eval_every=20, seed=11)
    other = run_minibatch_training(split, config, batch_size=16, max_steps=40, eval_every=20, seed=12)

    assert [point.validation_loss for point in first.curve] == [
        point.validation_loss for point in second.curve
    ]
    assert [point.validation_loss for point in first.curve] != [
        point.validation_loss for point in other.curve
    ], "the shuffle seed is doing no work, so 'seeded' and 'shuffled' are one claim"


def test_a_full_batch_mini_run_reproduces_the_original_loop() -> None:
    # With a batch as large as the training set, mini-batching degenerates to full-batch descent: same
    # initialization, same update, so the two loops must agree. Divergence here would mean one of them has
    # a step-ordering or seeding bug.
    config = _mini_config(steps=40, eval_every=10)
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=config.seed)
    train_windows = len(character_windows(split.train_text, fit_vocabulary(split), config.context_size))

    full = run_training(split, config)
    as_mini = run_minibatch_training(
        split, config, batch_size=train_windows, max_steps=40, eval_every=10, seed=5
    )

    assert as_mini.train_tokens == full.train_tokens
    for mine, theirs in zip(as_mini.curve, full.curve, strict=True):
        assert mine.validation_loss == pytest.approx(theirs.validation_loss, rel=1e-9)


@pytest.mark.parametrize("batch_size", [0, -1, True, 10_000])
def test_an_impossible_batch_is_refused(batch_size: int) -> None:
    config = _mini_config()
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=config.seed)

    with pytest.raises(ValueError, match="batch_size"):
        run_minibatch_training(split, config, batch_size=batch_size, max_steps=5, eval_every=1, seed=1)


def test_an_impossible_schedule_or_seed_is_refused() -> None:
    config = _mini_config()
    split = split_corpus(CORPUS.text, validation_fraction=0.2, seed=config.seed)

    with pytest.raises(ValueError, match="max_steps"):
        run_minibatch_training(split, config, batch_size=8, max_steps=0, eval_every=1, seed=1)
    with pytest.raises(ValueError, match="eval_every"):
        run_minibatch_training(split, config, batch_size=8, max_steps=4, eval_every=0, seed=1)
    with pytest.raises(ValueError, match="seed"):
        run_minibatch_training(split, config, batch_size=8, max_steps=4, eval_every=2, seed=False)


def test_a_split_with_no_windows_on_a_side_is_refused() -> None:
    config = _mini_config()
    empty_validation = Split(
        train_text="alpha beta gamma delta\n",
        validation_text="x",
        train_units=("alpha beta gamma delta",),
        validation_units=("x",),
    )

    with pytest.raises(ValueError, match="windows on both sides"):
        run_minibatch_training(
            empty_validation, config, batch_size=2, max_steps=4, eval_every=2, seed=1
        )
