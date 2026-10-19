"""Tests for the baselines: their arithmetic is exact and their refusals are specific.

Pure standard library, so these run in the dependency-free core matrix.
"""

from __future__ import annotations

import math

import pytest

from chakaso.training.baselines import UnigramBaseline, uniform_next_token_loss


def test_the_uniform_loss_is_exactly_log_of_the_vocabulary_size() -> None:
    assert uniform_next_token_loss(2) == pytest.approx(math.log(2))
    assert uniform_next_token_loss(1) == pytest.approx(0.0)
    assert uniform_next_token_loss(5) == pytest.approx(math.log(5))


def test_uniform_loss_rejects_an_invalid_vocabulary_size() -> None:
    for bad in (0, -1, True):
        with pytest.raises(ValueError, match="vocab_size"):
            uniform_next_token_loss(bad)  # type: ignore[arg-type]


def test_the_unigram_cross_entropy_matches_a_hand_computed_value() -> None:
    # Train targets [1,1,1] over vocab 3 with alpha=1: probabilities are (1/6, 4/6, 1/6).
    # Cross-entropy of the all-1s evaluation set is -ln(4/6) = ln(1.5).
    baseline = UnigramBaseline.fit([1, 1, 1], vocab_size=3, alpha=1.0)

    assert baseline.cross_entropy([1, 1, 1]) == pytest.approx(math.log(1.5))


def test_a_skewed_unigram_beats_the_uniform_predictor() -> None:
    # A training distribution concentrated on one symbol should give lower held-out loss than uniform.
    train = [2] * 90 + [0, 1]
    baseline = UnigramBaseline.fit(train, vocab_size=3, alpha=1.0)
    held_out = [2] * 10

    assert baseline.cross_entropy(held_out) < uniform_next_token_loss(3)


def test_an_unseen_heldout_identifier_still_gets_a_finite_baseline_loss() -> None:
    # Only id 0 seen in training; id 2 held out. Laplace smoothing keeps the loss finite, not infinite.
    baseline = UnigramBaseline.fit([0, 0, 0], vocab_size=3, alpha=1.0)

    loss = baseline.cross_entropy([2, 2])
    assert math.isfinite(loss)
    assert loss > 0.0


def test_an_empty_evaluation_set_is_refused() -> None:
    baseline = UnigramBaseline.fit([0, 1], vocab_size=2, alpha=1.0)
    with pytest.raises(ValueError, match="zero targets"):
        baseline.cross_entropy([])


def test_an_out_of_range_identifier_is_refused_on_fit_and_evaluate() -> None:
    with pytest.raises(ValueError, match="outside the vocabulary"):
        UnigramBaseline.fit([5], vocab_size=3)
    baseline = UnigramBaseline.fit([0, 1], vocab_size=3)
    with pytest.raises(ValueError, match="outside the vocabulary"):
        baseline.cross_entropy([9])


def test_a_non_positive_smoothing_or_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError, match="alpha"):
        UnigramBaseline.fit([0], vocab_size=2, alpha=0.0)
    with pytest.raises(ValueError, match="vocab_size"):
        UnigramBaseline.fit([0], vocab_size=0)
