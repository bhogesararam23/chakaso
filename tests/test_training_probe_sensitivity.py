"""Tests for the sensitivity comparison E2626's verdict is computed by.

The vectors here are small enough to count by hand: with twenty items and a model that passes all of them
against a model that passes fifteen, the discordant items are 0 and 5, and McNemar's exact two-sided p-value is
2/2**5 = 0.0625. Every expectation below is derived that way in the comment beside it.
"""

from __future__ import annotations

import pytest

from chakaso.training.errors import TrainingError
from chakaso.training.probe_sensitivity import (
    SensitivityError,
    compare_instruments,
    ordered_models,
)


def _vector(passes: int, total: int = 20) -> dict[str, float]:
    """A model that passes the first ``passes`` items and fails the rest, over ``total`` items."""
    return {f"item{index}": 1.0 if index < passes else 0.0 for index in range(total)}


def test_an_instrument_that_orders_the_models_as_the_criterion_does_is_measured_as_tracking() -> (
    None
):
    criterion = {"best": 1.0, "middle": 1.5, "worst": 3.0}
    instruments = {"probe": {"best": _vector(20), "middle": _vector(15), "worst": _vector(5)}}

    result = compare_instruments(criterion, instruments, materiality=0.4)

    verdict = result["instruments"]["probe"]
    # Accuracy 1.0, 0.75, 0.25 against quality 1.0, 1.5, 3.0: the same order read two ways.
    assert verdict["rho"] == pytest.approx(1.0)
    assert verdict["material_pairs"] == 3
    # best-vs-middle: discordant (0, 5), p = 2/32 = 0.0625 -> Holm's third step meets 0.05 and fails.
    # middle-vs-worst: (0, 10), p = 2/1024. best-vs-worst: (0, 15), p = 2/32768.
    # Step-down from the smallest: 6.1e-5 <= 0.05/3, 0.00195 <= 0.05/2, 0.0625 > 0.05 -> two of three.
    assert verdict["material_pairs_separated"] == 2
    assert verdict["material_coverage"] == pytest.approx(2 / 3)
    assert verdict["verdict"] == "S-b"
    assert verdict["smallest_separated_gap"] == pytest.approx(1.5)
    assert verdict["largest_unseparated_material_gap"] == pytest.approx(0.5)


def test_an_instrument_that_resolves_every_material_pair_earns_the_strong_verdict() -> None:
    criterion = {"best": 1.0, "middle": 1.5, "worst": 3.0}
    instruments = {"probe": {"best": _vector(20), "middle": _vector(10), "worst": _vector(0)}}

    result = compare_instruments(criterion, instruments, materiality=0.4)

    verdict = result["instruments"]["probe"]
    # best-vs-middle (0, 10) p = 2/1024; middle-vs-worst (10, 0) same; best-vs-worst (20, 0) p = 2/1048576.
    # All three clear the step-down thresholds 0.0167, 0.025, 0.05.
    assert verdict["material_pairs_separated"] == 3
    assert verdict["verdict"] == "S-a"
    assert verdict["smallest_separated_gap"] == pytest.approx(0.5)
    assert verdict["largest_unseparated_material_gap"] is None


def test_a_separation_the_wrong_way_round_is_reported_and_costs_the_verdict() -> None:
    criterion = {"better": 1.0, "worse": 2.0}
    # The instrument prefers the model the criterion says is worse, on 15 items to 0.
    instruments = {"probe": {"better": _vector(5), "worse": _vector(20)}}

    result = compare_instruments(criterion, instruments, materiality=0.4)

    verdict = result["instruments"]["probe"]
    row = result["pairs"][0]
    assert row["material"] is True
    assert row["probe"]["separated"] is True
    assert row["probe"]["direction"] == "contradicts"
    assert verdict["contradicted_pairs"] == [["better", "worse"]]
    assert verdict["verdict"] == "S-c"


def test_pairs_below_the_materiality_are_tested_but_do_not_count_toward_coverage() -> None:
    criterion = {"a": 1.0, "b": 1.02, "c": 3.0}
    instruments = {"probe": {"a": _vector(20), "b": _vector(19), "c": _vector(1)}}

    result = compare_instruments(criterion, instruments, materiality=0.05)

    # a-b differs by 0.02, below the threshold; a-c and b-c are material.
    assert [row["material"] for row in result["pairs"]] == [False, True, True]
    verdict = result["instruments"]["probe"]
    assert verdict["material_pairs"] == 2
    # a(20) vs b(19): discordant (1, 0), p = 2 * 1/2 = 1.0 -> not separated, and it does not count: it is not
    # a material pair. a vs c: (19, 0) p tiny -> separated. b vs c: (18, 0) -> separated.
    assert verdict["material_pairs_separated"] == 2
    assert verdict["material_coverage"] == pytest.approx(1.0)


def test_two_instruments_are_judged_separately_and_one_cannot_rescue_the_other() -> None:
    criterion = {"a": 1.0, "b": 1.5, "c": 3.0}
    instruments = {
        "good": {"a": _vector(20), "b": _vector(10), "c": _vector(0)},
        "flat": {"a": _vector(12), "b": _vector(12), "c": _vector(12)},
    }

    result = compare_instruments(criterion, instruments, materiality=0.4)

    assert result["instruments"]["good"]["verdict"] == "S-a"
    # A constant instrument has no discordant item anywhere: p = 1.0, and rho is 0 because it never moves.
    assert result["instruments"]["flat"]["verdict"] == "S-c"
    assert result["instruments"]["flat"]["rho"] == 0.0
    assert all(row["flat"]["n"] == 0 for row in result["pairs"])


def test_a_model_missing_from_one_instrument_is_refused_rather_than_dropped() -> None:
    criterion = {"a": 1.0, "b": 2.0}
    instruments = {"probe": {"a": _vector(20)}}

    with pytest.raises(SensitivityError, match="the criterion names"):
        compare_instruments(criterion, instruments, materiality=0.1)


def test_a_single_model_is_not_a_sensitivity_comparison() -> None:
    criterion = {"a": 1.0}
    instruments = {"probe": {"a": _vector(20)}}

    with pytest.raises(SensitivityError, match="at least two models"):
        compare_instruments(criterion, instruments, materiality=0.1)


def test_an_empty_outcome_vector_is_refused_before_it_becomes_a_zero_division() -> None:
    criterion = {"a": 1.0, "b": 2.0}
    instruments = {"probe": {"a": {}, "b": _vector(20)}}

    with pytest.raises(SensitivityError, match="no per-item outcomes"):
        compare_instruments(criterion, instruments, materiality=0.1)


def test_a_non_positive_materiality_is_refused() -> None:
    with pytest.raises(SensitivityError, match="materiality"):
        compare_instruments(
            {"a": 1.0, "b": 2.0}, {"p": {"a": _vector(3), "b": _vector(1)}}, materiality=0.0
        )


def test_ordered_models_sorts_by_the_criterion_and_not_by_name() -> None:
    # "aa" is the worse model, so it comes last even though it sorts first as a string.
    names = ordered_models({"zz": 1.0, "aa": 2.0}, {"p": {"zz": _vector(2), "aa": _vector(1)}})

    assert names == ("zz", "aa")


def test_the_error_type_is_a_training_error() -> None:
    assert issubclass(SensitivityError, TrainingError)
