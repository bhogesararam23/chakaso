"""Tests for the controlled multi-seed protocol (E2602 driver).

Numpy-gated. They pin that the protocol is deterministic and reproducible, that the selection rule reads
validation only, and that the recorded aggregate — mean held-out (test) loss below the unigram baseline —
holds for the fixed seeds, rather than asserting a general property that a single run could not guarantee.
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from chakaso.training.config import TrainingConfig  # noqa: E402
from chakaso.training.controlled import (  # noqa: E402
    CONTROLLED_CONFIG,
    CONTROLLED_SEEDS,
    run_controlled_experiment,
)


def _small_config() -> TrainingConfig:
    return TrainingConfig(
        context_size=4,
        embed_dim=8,
        hidden_dim=12,
        learning_rate=0.02,
        seed=0,
        steps=40,
        eval_every=10,
    )


def test_the_protocol_is_reproducible_from_the_seeds() -> None:
    config = _small_config()

    first = run_controlled_experiment(config, seeds=(3, 11))
    second = run_controlled_experiment(config, seeds=(3, 11))

    assert [r.test_loss_at_best for r in first.results] == [
        r.test_loss_at_best for r in second.results
    ]
    assert first.mean_test_loss_at_best == second.mean_test_loss_at_best


def test_one_result_is_produced_per_seed_with_sane_fields() -> None:
    report = run_controlled_experiment(_small_config(), seeds=(1, 2, 3))

    assert report.seeds == (1, 2, 3)
    for result in report.results:
        assert result.vocab_size > 1
        assert result.test_tokens > 0 and result.train_tokens > 0 and result.validation_tokens > 0
        assert 0 <= result.best_step <= _small_config().steps
        assert (
            result.uniform_baseline > result.unigram_baseline
        )  # skewed toy data: unigram is better


def test_the_spread_is_reported_across_seeds_and_absent_for_one() -> None:
    many = run_controlled_experiment(_small_config(), seeds=(1, 2, 3))
    one = run_controlled_experiment(_small_config(), seeds=(1,))

    assert many.stdev_test_loss_at_best is not None
    assert many.stdev_test_loss_at_best >= 0.0
    assert one.stdev_test_loss_at_best is None


def test_the_recorded_claim_holds_for_the_fixed_seeds() -> None:
    # E2602's cited result: under early stopping selected on validation, the mean held-out (test) loss is
    # below the mean unigram baseline for the pre-declared seeds. Fixed and deterministic, so this is a
    # regression guard for the recorded number, not a claimed law.
    report = run_controlled_experiment(CONTROLLED_CONFIG, seeds=CONTROLLED_SEEDS)

    assert report.mean_test_loss_at_best < report.mean_unigram_baseline
    # Early stopping helps: the val-selected test loss beats the un-early-stopped final test loss.
    assert report.mean_test_loss_at_best < report.mean_test_loss_final


def test_no_seeds_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_controlled_experiment(_small_config(), seeds=())
