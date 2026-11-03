"""Tests for the paired statistics used by the probe-sensitivity analysis.

Every expected value here is computed by hand in the comment beside it, because these functions decide whether
an instrument is said to detect something, and a statistic nobody can re-derive is not evidence.
"""

from __future__ import annotations

import math

import pytest

from chakaso.training.errors import TrainingError
from chakaso.training.paired_stats import (
    PairedStatsError,
    holm_rejections,
    mcnemar_exact,
    mean_confidence_interval,
    sign_agreement,
    sign_test_p_value,
    spearman_rho,
    student_t_quantile,
)


def test_mcnemar_counts_only_the_items_where_the_two_models_disagree() -> None:
    # b=5, c=3 over 8 discordant items (the 2 both tied are not directions). Lesser tail:
    # (C(8,0)+C(8,1)+C(8,2)+C(8,3)) / 2^8 = (1+8+28+56)/256 = 93/256. Two-sided: 186/256.
    first = [1.0] * 5 + [0.0] * 3 + [0.5] * 2
    second = [0.0] * 5 + [1.0] * 3 + [0.5] * 2

    b, c, total, p_value = mcnemar_exact(first, second)

    assert (b, c, total) == (5, 3, 8)
    assert p_value == pytest.approx(186 / 256)


def test_mcnemar_exact_p_value_matches_the_binomial_tail_by_hand() -> None:
    # Eight items the first model passed alone, one the second passed alone, one both passed. Lesser tail:
    # (C(9,0) + C(9,1)) / 2^9 = 10/512. Two-sided: 20/512.
    first = [1.0] * 8 + [0.0] + [1.0]
    second = [0.0] * 8 + [1.0] + [1.0]

    b, c, total, p_value = mcnemar_exact(first, second)

    assert (b, c, total) == (8, 1, 9)
    assert p_value == pytest.approx(20 / 512)


def test_mcnemar_reports_no_difference_when_the_models_answer_alike() -> None:
    first = [1.0, 0.0, 1.0]
    second = [1.0, 0.0, 1.0]

    assert mcnemar_exact(first, second) == (0, 0, 0, 1.0)


def test_a_tie_on_either_side_is_not_a_direction() -> None:
    # One item each way and a tie: b=c=1, n=2. Lesser tail (C(2,0)+C(2,1))/4 = 3/4, two-sided capped at 1.
    b, c, total, p_value = mcnemar_exact([1.0, 0.0, 0.5], [0.0, 1.0, 0.5])

    assert (b, c, total) == (1, 1, 2)
    assert p_value == 1.0


def test_mcnemar_refuses_vectors_that_are_not_the_same_items() -> None:
    with pytest.raises(PairedStatsError, match="equal-length"):
        mcnemar_exact([1.0, 0.0], [1.0])
    with pytest.raises(PairedStatsError, match="zero items"):
        mcnemar_exact([], [])


def test_a_perfectly_monotone_nonlinear_pair_scores_one() -> None:
    # Spearman reads order, not distance: 1, 4, 9 is as good a monotone image of 1, 2, 3 as 1, 2, 3 is.
    assert spearman_rho([1.0, 2.0, 3.0], [1.0, 4.0, 9.0]) == pytest.approx(1.0)
    assert spearman_rho([1.0, 2.0, 3.0], [9.0, 4.0, 1.0]) == pytest.approx(-1.0)


def test_spearman_resolves_a_tied_rank_by_averaging() -> None:
    # y ranks [1.5, 1.5, 3]; x ranks [1, 2, 3]. Covariance 1.5, spreads 2 and 1.5: rho = 1.5 / sqrt(3).
    assert spearman_rho([1.0, 2.0, 3.0], [5.0, 5.0, 6.0]) == pytest.approx(1.5 / 3.0**0.5)


def test_spearman_is_zero_when_one_vector_never_moves() -> None:
    assert spearman_rho([1.0, 2.0, 3.0], [0.8, 0.8, 0.8]) == 0.0
    assert spearman_rho([1.0], [1.0]) == 0.0


def test_holm_is_a_step_down_that_stops_at_the_first_failure() -> None:
    # m=3, alpha=0.05: thresholds are alpha/3, alpha/2, alpha applied to the ascending p-values.
    # 0.001 < 0.01667, 0.02 < 0.025, 0.03 < 0.05: the family is rejected whole.
    assert holm_rejections([0.001, 0.02, 0.03]) == [True, True, True]
    # 0.04 already fails alpha/2, so the step stops before 0.045 is reached even though it would clear alpha
    # on its own. A per-test Bonferroni-style threshold would have rejected nothing here either, but for the
    # wrong reason - and would also have rejected nothing in the line above.
    assert holm_rejections([0.04, 0.045]) == [False, False]
    # The first clears alpha/3, the second fails alpha/2, so the step stops there: 0.045 is never compared
    # against alpha, even though on its own it would clear it.
    assert holm_rejections([0.001, 0.03, 0.045]) == [True, False, False]


def test_holm_resolves_tied_p_values_by_the_same_threshold() -> None:
    # Both meet alpha/2 = 0.025 as the first step; neither clears it, so the family stops immediately.
    assert holm_rejections([0.05, 0.05]) == [False, False]
    assert holm_rejections([0.02, 0.02]) == [True, True]


def test_holm_keeps_the_input_order_in_its_answer() -> None:
    # p[2] is the smallest, so the rejection lands at index 2, not at the front.
    assert holm_rejections([0.4, 0.3, 0.001]) == [False, False, True]


def test_holm_refuses_an_empty_family_and_a_broken_alpha() -> None:
    with pytest.raises(PairedStatsError, match="at least one"):
        holm_rejections([])
    with pytest.raises(PairedStatsError, match="alpha"):
        holm_rejections([0.5], alpha=0.0)


def test_sign_agreement_counts_only_items_both_models_answered() -> None:
    # Both answered 4 items and agree on 3: 0.75. The two ties are not answers and are not counted.
    first = [1.0, 1.0, 1.0, 0.0, 0.5, 0.5]
    second = [1.0, 1.0, 0.0, 0.0, 0.5, 1.0]

    assert sign_agreement(first, second) == pytest.approx(3 / 4)
    assert sign_agreement([0.5, 0.5], [1.0, 0.0]) == 1.0


@pytest.mark.parametrize(
    ("degrees", "critical"),
    [
        (1, 12.706),
        (2, 4.303),
        (3, 3.182),
        (4, 2.776),
        (5, 2.571),
        (7, 2.365),
        (10, 2.228),
        (20, 2.086),
        (30, 2.042),
    ],
)
def test_the_student_t_quantile_matches_the_published_two_sided_ninety_five_table(
    degrees: int, critical: float
) -> None:
    assert student_t_quantile(0.975, degrees) == pytest.approx(critical, abs=5e-4)


def test_the_student_t_quantile_is_monotone_in_probability_and_shrinks_toward_the_normal() -> None:
    # The table the record reads is the 0.975 column; a reader checking a different confidence wants the same
    # shape: larger probability, larger critical value, and df -> infinity approaching 1.96.
    assert student_t_quantile(0.9, 4) < student_t_quantile(0.975, 4) < student_t_quantile(0.995, 4)
    assert student_t_quantile(0.975, 1000) == pytest.approx(1.96, abs=0.01)


def test_the_student_t_quantile_refuses_a_probability_outside_the_unit_interval() -> None:
    with pytest.raises(PairedStatsError, match="probability"):
        student_t_quantile(1.0, 4)
    with pytest.raises(PairedStatsError, match="degrees"):
        student_t_quantile(0.975, 0)


def test_mean_confidence_interval_is_the_students_t_interval_of_the_sample_mean() -> None:
    # The worked example every statistics text uses: 2 4 4 4 5 5 7 9. Mean 5, deviations -3 -1 -1 -1 0 0 2 4,
    # sum of squares 32, s^2 = 32/7, s = 2.1380899, SE = s/sqrt(8) = 0.7559289, t(0.975, 7) = 2.3646243,
    # half-width 0.7559289 x 2.3646243 = 1.7874879 -> [3.2125121, 6.7874879].
    mean, low, high, degrees, critical = mean_confidence_interval([2, 4, 4, 4, 5, 5, 7, 9])

    assert mean == pytest.approx(5.0)
    assert (low, high) == pytest.approx((3.2125121, 6.7874879), abs=1e-5)
    assert (degrees, critical) == (7, pytest.approx(2.3646243, abs=5e-5))


def test_mean_confidence_interval_widens_as_the_sample_shrinks() -> None:
    # Three and four draws of the same spread: the t critical value rises faster than the SE falls.
    wide = mean_confidence_interval([1.0, 2.0, 3.0])
    narrower = mean_confidence_interval([1.0, 2.0, 3.0, 2.0])
    assert wide[2] - wide[1] > narrower[2] - narrower[1]
    assert wide[0] == pytest.approx(2.0) == narrower[0]


def test_mean_confidence_interval_needs_two_draws_and_finite_ones() -> None:
    with pytest.raises(PairedStatsError, match="at least two"):
        mean_confidence_interval([1.0])
    with pytest.raises(PairedStatsError, match="finite"):
        mean_confidence_interval([1.0, math.nan])


@pytest.mark.parametrize(
    ("successes", "total", "tail"),
    [
        (5, 5, 1 / 32),  # all five: C(5,5)/2^5
        (4, 5, 6 / 32),  # four or five: (C(5,4)+C(5,5))/2^5
        (3, 5, 16 / 32),  # three or more: (10+5+1)/32
        (1, 4, 5 / 16),
        (0, 3, 1 / 8),
    ],
)
def test_the_sign_test_tail_is_the_exact_binomial_under_a_fair_coin(
    successes: int, total: int, tail: float
) -> None:
    assert sign_test_p_value(successes, total=total) == pytest.approx(tail)


def test_the_sign_test_reads_the_rarer_sign_when_the_coin_lands_mostly_the_other_way() -> None:
    # One seed against three: the test is on the minority sign, so this is the same question as 3 of 4.
    assert sign_test_p_value(1, total=4) == pytest.approx(sign_test_p_value(3, total=4))


def test_the_sign_test_refuses_a_count_outside_its_sample() -> None:
    with pytest.raises(PairedStatsError, match="successes"):
        sign_test_p_value(5, total=4)
    with pytest.raises(PairedStatsError, match="total"):
        sign_test_p_value(1, total=0)


def test_the_paired_error_is_the_package_error_type() -> None:
    assert issubclass(PairedStatsError, TrainingError)
