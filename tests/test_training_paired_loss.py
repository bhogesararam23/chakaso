"""Tests for the clustered paired test on held-out loss.

Every expectation is hand-derived. A cluster bootstrap that nobody can reproduce by hand is not evidence that an arm
is better; it is a picture of one. The seeded draws are checked for determinism, the point estimate is checked against
the pooled ratio it is defined to equal, and the small cases are computed with a pen.
"""

from __future__ import annotations

import pytest

from chakaso.training.errors import TrainingError
from chakaso.training.paired_loss import (
    PairedLossError,
    jackknife,
    paired_document_test,
)


def test_the_point_estimate_is_the_pooled_difference_not_a_mean_of_ratios() -> None:
    # Two documents, equal weight, differences +0.10 and -0.02: mean 0.04 either way. Unequal weights are the
    # case that distinguishes the definitions: with 3 and 1 characters of weight, the pooled difference is
    # (3*0.10 + 1*(-0.02)) / 4 = 0.28/4 = 0.07, while an unweighted mean of the per-document ratios would say 0.04.
    result = paired_document_test([0.10, -0.02], [3.0, 1.0], [0, 1], seed=7, resamples=200)
    assert result["mean_difference"] == pytest.approx(0.07, abs=1e-12)
    assert result["unit"] == "bits per character, first minus second"


def test_a_one_sided_run_of_documents_gives_an_interval_clear_of_zero() -> None:
    # Eight documents, all +0.05 with identical weights: every bootstrap draw resamples the same value, so the
    # interval is a point at 0.05 and it excludes zero. A degenerate case is worth pinning because it is what a
    # "the interval excludes zero" claim looks like when there is no spread at all.
    result = paired_document_test([0.05] * 8, [1.0] * 8, list(range(8)), seed=11, resamples=500)
    assert result["mean_difference"] == pytest.approx(0.05)
    assert result["interval"] == pytest.approx((0.05, 0.05), abs=1e-12)
    assert result["excludes_zero"] is True
    assert result["positive_documents"] == 8
    # Eight draws all one way: P(X >= 8) for a fair coin is 1/256.
    assert result["sign_test_p_value"] == pytest.approx(1 / 256)


def test_documents_that_disagree_in_sign_put_zero_inside_the_interval() -> None:
    # Four +0.05 against four -0.05: the pooled difference is exactly zero, and no resample can convince a
    # reader that one arm is better.
    result = paired_document_test(
        [0.05, 0.05, 0.05, 0.05, -0.05, -0.05, -0.05, -0.05],
        [1.0] * 8,
        list(range(8)),
        seed=3,
        resamples=400,
    )
    assert result["mean_difference"] == pytest.approx(0.0, abs=1e-12)
    assert result["excludes_zero"] is False
    assert result["positive_documents"] == 4
    # Four of eight the same way is the balanced case: P(X >= 4) = (70+56+28+8+1)/256.
    assert result["sign_test_p_value"] == pytest.approx(163 / 256)


def test_the_bootstrap_is_a_function_of_the_seed_and_of_the_clusters_it_resamples() -> None:
    differences = [0.04, -0.01, 0.09, 0.02, -0.03, 0.05, 0.01, -0.06, 0.03, -0.02, 0.07, 0.00]
    weights = [2.0] * 12
    groups = list(range(12))
    first = paired_document_test(differences, weights, groups, seed=123, resamples=300)
    again = paired_document_test(differences, weights, groups, seed=123, resamples=300)
    assert first["interval"] == again["interval"]
    other = paired_document_test(differences, weights, groups, seed=124, resamples=300)
    # The point estimate is not a function of the seed at all; only the interval is.
    assert other["mean_difference"] == pytest.approx(first["mean_difference"])
    assert other["interval"] != first["interval"]


def test_resampling_clusters_not_rows_leaves_a_document_inside_one_cluster_untouched() -> None:
    # Three documents in one work and one document in each of two others. The single-work document cannot move
    # the estimate on its own: drawn or not, it arrives with its two siblings.
    differences = [0.10, 0.10, 0.10, -0.30, 0.30]
    weights = [1.0] * 5
    groups = [0, 0, 0, 1, 2]
    result = paired_document_test(differences, weights, groups, seed=5, resamples=500)
    # Pooled: (3*0.10 - 0.30 + 0.30) / 5 = 0.30/5 = 0.06.
    assert result["mean_difference"] == pytest.approx(0.06, abs=1e-12)
    assert result["clusters"] == 3
    assert result["cluster_means"] == {
        "0": pytest.approx(0.10),
        "1": pytest.approx(-0.30),
        "2": pytest.approx(0.30),
    }


def test_one_heavily_weighted_document_can_carry_the_estimate_without_carrying_the_interval() -> (
    None
):
    # One document of 1,000 characters at +0.01 and nine of 1 character at -0.01. The pooled difference is
    # (10 - 0.09)/1009 = 0.009812, so the headline number says the first arm is worse. But a third of the
    # resamples never draw that document at all ((9/10)^10 = 0.349), and those resamples say -0.01: the
    # interval straddles zero. This is the property the campaign needs stated plainly — a weighted mean can
    # sit on one document and the cluster resampling can see that it might not be there.
    differences = [0.01] + [-0.01] * 9
    weights = [1000.0] + [1.0] * 9
    result = paired_document_test(differences, weights, list(range(10)), seed=9, resamples=600)
    assert result["mean_difference"] == pytest.approx((10.0 - 0.09) / 1009.0, abs=1e-12)
    assert result["excludes_zero"] is False
    assert result["interval"][0] == pytest.approx(-0.01, abs=1e-9)


def test_a_lower_confidence_buys_a_narrower_interval_on_the_same_draws() -> None:
    differences = [0.05, -0.04, 0.02, -0.01, 0.06, -0.03, 0.01, -0.02]
    weights = [1.0] * 8
    groups = list(range(8))
    wide = paired_document_test(differences, weights, groups, seed=2, resamples=1500)
    narrow = paired_document_test(
        differences, weights, groups, seed=2, resamples=1500, probability=0.90
    )
    assert narrow["probability"] == pytest.approx(0.90)
    assert narrow["interval"][0] >= wide["interval"][0]
    assert narrow["interval"][1] <= wide["interval"][1]
    assert narrow["mean_difference"] == pytest.approx(wide["mean_difference"])


def test_inputs_that_do_not_describe_the_same_documents_are_refused() -> None:
    with pytest.raises(PairedLossError, match="caller supplied 1 weights"):
        paired_document_test([0.1, 0.2], [1.0], [0, 1], seed=1, resamples=100)
    with pytest.raises(PairedLossError, match="cluster keys"):
        paired_document_test([0.1, 0.2], [1.0, 1.0], [0], seed=1, resamples=100)
    with pytest.raises(PairedLossError, match="positive"):
        paired_document_test([0.1, 0.2], [1.0, 0.0], [0, 1], seed=1, resamples=100)
    with pytest.raises(PairedLossError, match="finite"):
        paired_document_test([0.1, float("nan")], [1.0, 1.0], [0, 1], seed=1, resamples=100)
    with pytest.raises(PairedLossError, match="at least one document"):
        paired_document_test([], [], [], seed=1, resamples=100)
    with pytest.raises(PairedLossError, match="resamples"):
        paired_document_test([0.1], [1.0], [0], seed=1, resamples=0)
    with pytest.raises(PairedLossError, match="probability"):
        paired_document_test([0.1], [1.0], [0], seed=1, resamples=10, probability=1.5)


def test_the_paired_error_is_the_package_error_type() -> None:
    assert issubclass(PairedLossError, TrainingError)


def test_a_leave_one_out_table_is_the_estimate_over_what_is_left() -> None:
    # Differences +0.10 and -0.30, equal weights: without the first the estimate is -0.30, without the second
    # +0.10. Centre -0.10, squared deviations 0.04 and 0.04, SE = sqrt((1/2) x 0.08) = 0.2.
    result = jackknife([0.10, -0.30], [1.0, 1.0])
    assert result["estimate_without"] == pytest.approx([-0.30, 0.10])
    assert (result["minimum_without"], result["maximum_without"]) == pytest.approx((-0.30, 0.10))
    assert result["standard_error"] == pytest.approx(0.2, abs=1e-12)


def test_the_table_shows_which_single_cluster_the_headline_number_rests_on() -> None:
    # One cluster of 100 characters at +1.0 and two of 1 character at zero: the pooled estimate is 100/102 =
    # 0.98039, and removing the big one gives exactly zero. A bootstrap interval on three clusters could still
    # clear zero; this table says plainly that one document carries the whole claim.
    result = jackknife([1.0, 0.0, 0.0], [100.0, 1.0, 1.0])
    assert result["estimate_without"][0] == pytest.approx(0.0, abs=1e-15)
    assert result["estimate_without"][1] == pytest.approx(100.0 / 101.0)
    assert result["minimum_without"] == pytest.approx(0.0, abs=1e-15)
    assert result["maximum_without"] == pytest.approx(100.0 / 101.0)


def test_a_leave_one_out_table_needs_two_clusters_and_the_same_documents_as_the_test() -> None:
    with pytest.raises(PairedLossError, match="at least two clusters"):
        jackknife([0.1], [1.0])
    with pytest.raises(PairedLossError, match="caller supplied 1 weights"):
        jackknife([0.1, 0.2], [1.0])
