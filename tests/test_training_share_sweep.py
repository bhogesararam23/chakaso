"""Tests for reading a share sweep: no optimum is claimed that the noise margin does not support."""

from __future__ import annotations

import pytest

from chakaso.training.share_sweep import (
    Margins,
    classify_difference,
    classify_share_sweep,
    dominance,
    equal_weight_mean,
    margins_from_stdev,
)

_MARGINS = Margins(central=0.006, upper=0.02)


def test_margins_are_the_pair_band_from_the_spread_and_from_its_upper_end() -> None:
    margins = margins_from_stdev(0.00226, 0.00648)

    assert margins.central == pytest.approx(2.77 * 0.00226)
    assert margins.upper == pytest.approx(2.77 * 0.00648)


def test_a_kept_band_can_only_raise_the_separable_margin() -> None:
    kept = margins_from_stdev(0.00226, 0.00648, declared_band=0.02)

    assert kept.upper == pytest.approx(0.02)
    assert kept.central == pytest.approx(2.77 * 0.00226)
    # A band below the interval's upper end does not lower it.
    assert margins_from_stdev(0.003, 0.0097, declared_band=0.01).upper == pytest.approx(
        2.77 * 0.0097
    )


def test_a_replacement_band_also_stands_in_for_the_central_margin() -> None:
    margins = margins_from_stdev(0.01232, 0.03541, declared_band=0.035, band_replaces_central=True)

    assert margins.central == pytest.approx(0.035)
    assert margins.upper == pytest.approx(2.77 * 0.03541)


@pytest.mark.parametrize(
    ("difference", "expected"),
    [
        (0.0, "tie"),
        (0.006, "tie"),
        (-0.006, "tie"),
        (0.0061, "indeterminate"),
        (-0.02, "indeterminate"),
        (0.0201, "separable"),
        (-0.5, "separable"),
    ],
)
def test_a_difference_is_a_tie_indeterminate_or_separable(difference: float, expected: str) -> None:
    assert classify_difference(difference, _MARGINS) == expected


def test_the_equal_weight_mean_counts_every_register_once() -> None:
    values = {"prose-": 2.0, "science-": 4.0, "medicine-": 6.0}

    assert equal_weight_mean(values, ("prose-", "science-")) == 3.0
    assert equal_weight_mean(values, ("prose-", "science-", "medicine-")) == 4.0
    with pytest.raises(ValueError, match="no value for partitions"):
        equal_weight_mean(values, ("prose-", "poetry-"))
    with pytest.raises(ValueError, match="at least one partition"):
        equal_weight_mean(values, ())


def test_an_interior_arm_that_clears_both_endpoints_is_an_interior_optimum() -> None:
    result = classify_share_sweep({0.0: 2.10, 0.25: 2.00, 0.5: 2.02, 0.75: 2.06, 1.0: 2.15}, 0.02)

    assert result["outcome"] == "interior_optimum"
    assert result["best_share"] == 0.25
    assert result["within_margin_of_best"] == [0.25, 0.5]


def test_an_interior_arm_that_clears_only_one_endpoint_is_not_an_optimum() -> None:
    result = classify_share_sweep({0.0: 2.01, 0.25: 2.00, 0.5: 2.05, 1.0: 2.20}, 0.02)

    assert result["outcome"] == "unresolved"
    assert result["within_margin_of_best"] == [0.0, 0.25]


def test_an_endpoint_that_beats_every_other_arm_is_a_boundary_optimum() -> None:
    result = classify_share_sweep({0.0: 2.00, 0.25: 2.05, 0.5: 2.10, 1.0: 2.30}, 0.02)

    assert result["outcome"] == "boundary_optimum"
    assert result["best_share"] == 0.0


def test_an_endpoint_that_an_interior_arm_nearly_matches_is_unresolved_not_a_boundary() -> None:
    result = classify_share_sweep({0.0: 2.00, 0.25: 2.01, 0.5: 2.10, 1.0: 2.30}, 0.02)

    assert result["outcome"] == "unresolved"


def test_a_flat_sweep_names_every_arm_it_cannot_separate() -> None:
    result = classify_share_sweep({0.0: 2.000, 0.25: 2.004, 0.5: 2.002, 1.0: 2.006}, 0.02)

    assert result["outcome"] == "unresolved"
    assert result["within_margin_of_best"] == [0.0, 0.25, 0.5, 1.0]


def test_a_sweep_needs_an_interior_and_a_sensible_margin() -> None:
    with pytest.raises(ValueError, match="at least three shares"):
        classify_share_sweep({0.0: 2.0, 1.0: 2.1}, 0.02)
    with pytest.raises(ValueError, match="margin must be >= 0"):
        classify_share_sweep({0.0: 2.0, 0.5: 2.1, 1.0: 2.2}, -0.1)


def test_a_trade_off_between_registers_produces_no_dominance_pair() -> None:
    bits = {
        0.0: {"prose-": 2.30, "science-": 1.90},
        1.0: {"prose-": 1.80, "science-": 2.10},
    }

    assert dominance(bits, {"prose-": 0.02, "science-": 0.027}) == []


def test_an_arm_better_everywhere_by_more_than_the_margins_dominates() -> None:
    bits = {
        0.25: {"prose-": 1.80, "science-": 1.90},
        0.5: {"prose-": 1.90, "science-": 2.00},
        0.75: {"prose-": 1.805, "science-": 1.91},
    }

    assert dominance(bits, {"prose-": 0.02, "science-": 0.027}) == [(0.25, 0.5), (0.75, 0.5)]
