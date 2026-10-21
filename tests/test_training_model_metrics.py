"""Tests for the metric arithmetic that makes two tokenizations comparable.

Bits per character is the only number in this project that lets a character model and a subword model be
put in the same table, so its arithmetic is checked against hand-computable cases rather than against
another implementation of itself.
"""

from __future__ import annotations

import math

import pytest

from chakaso.training.errors import TrainingError
from chakaso.training.model_metrics import bits_per_character, metrics_from_scores
from chakaso.training.subword import DOCUMENT_END, train_subword_vocabulary

_VOCABULARY = train_subword_vocabulary(["ab ab ab cd cd cd"], target_size=260, min_frequency=2)


def _uniform(nats: float, count: int) -> tuple[float, ...]:
    return (nats,) * count


# ------------------------------------------------------------------ the conversion


def test_bits_per_character_converts_a_total_description_length() -> None:
    # 1 nat of description over 2 characters is 1/ln2 bits per character.
    expected = 1.0 / math.log(2.0) / 2

    assert bits_per_character(1.0, 2) == pytest.approx(expected)


def test_a_uniform_distribution_over_two_symbols_costs_exactly_one_bit_per_symbol() -> None:
    nats = math.log(2.0) * 4

    assert bits_per_character(nats, 4) == pytest.approx(1.0)


def test_bits_per_character_refuses_zero_characters() -> None:
    with pytest.raises(TrainingError, match="zero characters"):
        bits_per_character(1.0, 0)


# ------------------------------------------------------------------ assembled metrics


def test_metrics_combine_nats_targets_and_predictions() -> None:
    targets = list(_VOCABULARY.encode("ab"))
    nats = [0.5] * len(targets)
    predictions = list(targets)

    metrics = metrics_from_scores(nats, targets, predictions, _VOCABULARY)

    assert metrics.positions_scored == len(targets)
    assert metrics.loss_nats == pytest.approx(0.5)
    assert metrics.perplexity == pytest.approx(math.exp(0.5))
    assert metrics.accuracy == pytest.approx(1.0)
    assert metrics.characters_scored == len(_VOCABULARY.decode(targets))


def test_the_bpc_denominator_is_the_decoded_whole_sequence_not_a_per_token_count() -> None:
    # A multi-byte character split across byte tokens must count once. Scored as per-token character
    # counts, the same text would appear longer than it is and the model would look better than it is.
    text = "é"  # two UTF-8 bytes, no learned merge
    targets = list(_VOCABULARY.encode(text))
    nats = [1.0] * len(targets)

    metrics = metrics_from_scores(nats, targets, targets, _VOCABULARY)

    assert len(targets) == 2
    assert metrics.characters_scored == 1
    assert metrics.positions_scored == 2
    assert metrics.bits_per_character == pytest.approx(2.0 / math.log(2.0))


def test_a_separator_target_costs_its_prediction_and_no_characters() -> None:
    targets = [ord("a"), DOCUMENT_END, ord("b")]
    nats = [0.2, 0.3, 0.4]

    metrics = metrics_from_scores(nats, targets, targets, _VOCABULARY)

    assert metrics.separator_targets == 1
    assert metrics.characters_scored == 2
    assert metrics.positions_scored == 3
    assert metrics.bits_per_character == pytest.approx((0.9 / math.log(2.0)) / 2)
    assert metrics.loss_nats == pytest.approx(0.3)


def test_accuracy_counts_argmax_matches_over_every_scored_position() -> None:
    targets = [ord("a"), ord("b"), ord("c"), ord("d")]
    predictions = [ord("a"), ord("b"), ord("x"), ord("y")]

    metrics = metrics_from_scores([0.1] * 4, targets, predictions, _VOCABULARY)

    assert metrics.accuracy == pytest.approx(0.5)


def test_metrics_use_a_summation_that_does_not_accumulate_rounding() -> None:
    # A naive left-to-right loop loses precision once the running total dwarfs each term; the reported
    # mean must not depend on how many positions were summed.
    many = [1e-9] * 200_000
    targets = [ord("a")] * len(many)

    metrics = metrics_from_scores(many, targets, targets, _VOCABULARY)

    # math.fsum over two million tiny equal terms is exactly 2e-3; a naive loop drifts. The reported mean
    # must not depend on how many positions were summed.
    assert metrics.loss_nats == pytest.approx(1e-9, rel=1e-12)


def test_metrics_refuse_inconsistent_and_empty_inputs() -> None:
    with pytest.raises(TrainingError, match="at least one scored position"):
        metrics_from_scores([], [], [], _VOCABULARY)
    with pytest.raises(TrainingError, match="equal length"):
        metrics_from_scores([0.5, 0.5], [1, 2], [1], _VOCABULARY)
    with pytest.raises(TrainingError, match="cannot be negative"):
        metrics_from_scores([-0.1], [1], [1], _VOCABULARY)


def test_the_dict_view_carries_every_dimension_and_merges_none_of_them() -> None:
    targets = [ord("a"), ord("b")]

    values = metrics_from_scores([0.4, 0.6], targets, targets, _VOCABULARY).as_dict()

    assert set(values) == {
        "loss_nats_per_token",
        "perplexity",
        "bits_per_character",
        "positions_scored",
        "characters_scored",
        "token_accuracy",
        "separator_targets",
    }
    # The point of separate dimensions: there is no combined quality number to read off.
    assert not [key for key in values if "combined" in key or "quality" in key]
    assert values["positions_scored"] == 2 and values["characters_scored"] == 2
