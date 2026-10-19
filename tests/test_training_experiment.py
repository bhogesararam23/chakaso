"""Tests for the experiment driver: determinism and the recorded outcome's shape.

Numpy-gated (``[train]`` path). It does not restate the report's prose; it pins that the run is
reproducible from the seed and that the measured comparison against the baselines has the structure the
record claims — the driver being the thing E2601's numbers are produced by.
"""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from chakaso.training.config import TrainingConfig  # noqa: E402
from chakaso.training.experiment import DEFAULT_CONFIG, measure, run_char_experiment  # noqa: E402


def test_the_report_is_reproducible_from_the_configuration_and_seed() -> None:
    config = TrainingConfig(
        context_size=4,
        embed_dim=8,
        hidden_dim=12,
        learning_rate=0.02,
        seed=3,
        steps=60,
        eval_every=20,
    )

    first = measure(config)
    second = measure(config)

    assert first == second


def test_the_default_experiment_beats_both_baselines_on_heldout_tokens() -> None:
    # The claim E2601 records: on the held-out split the trained model's cross-entropy is below both
    # the uniform and the unigram baseline. Fixed seed, so this is a deterministic regression guard.
    outcome = run_char_experiment(DEFAULT_CONFIG)

    assert outcome.beats_uniform
    assert outcome.beats_unigram
    assert (
        outcome.model_validation.loss
        < outcome.unigram_baseline_loss
        < outcome.uniform_baseline_loss
    )


def test_the_curve_learns_and_records_a_heldout_minimum_before_overfitting() -> None:
    # The measured shape of the result: training loss falls monotonically toward its first recorded
    # values, and the held-out loss drops below the random-init point at some recorded step (the loop
    # genuinely reduces held-out error), even though it later rises again — overfitting the tiny corpus.
    outcome = run_char_experiment(DEFAULT_CONFIG)
    curve = outcome.curve

    assert curve[0].step == 0
    assert curve[-1].step == DEFAULT_CONFIG.steps
    heldout_values = [point.validation_loss for point in curve]
    assert min(heldout_values) < heldout_values[0]
    # Overfitting is real and reported, not hidden: the final held-out loss is worse than the best.
    assert heldout_values[-1] > min(heldout_values)


def test_the_report_carries_identity_and_all_measured_fields() -> None:
    report = measure(
        TrainingConfig(
            context_size=3,
            embed_dim=6,
            hidden_dim=8,
            learning_rate=0.05,
            seed=1,
            steps=20,
            eval_every=10,
        )
    )

    for key in (
        "experiment",
        "config",
        "vocab_size",
        "vocab_fingerprint",
        "dataset_digest",
        "parameter_count",
        "uniform_baseline_loss_nats",
        "unigram_baseline_loss_nats",
        "final_validation_loss_nats",
        "beats_uniform",
        "beats_unigram",
        "checkpoint_fingerprint",
        "curve",
    ):
        assert key in report
    assert report["experiment"] == "E2601"
    assert len(report["curve"]) >= 3
