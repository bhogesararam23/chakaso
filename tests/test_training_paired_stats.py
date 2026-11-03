"""Tests for the paired statistics used by the probe-sensitivity analysis.

Every expected value here is computed by hand in the comment beside it, because these functions decide whether
an instrument is said to detect something, and a statistic nobody can re-derive is not evidence.
"""

from __future__ import annotations

import pytest

from chakaso.training.errors import TrainingError
from chakaso.training.paired_stats import (
    PairedStatsError,
    holm_rejections,
    mcnemar_exact,
    sign_agreement,
    spearman_rho,
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


def test_the_paired_error_is_the_package_error_type() -> None:
    assert issubclass(PairedStatsError, TrainingError)
