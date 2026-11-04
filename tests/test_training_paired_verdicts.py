"""Tests for the work-clustered paired verdict test: hand-derived p-values, and where it parts from McNemar.

Every expected number here is derived in the test body or in a comment, not copied from the function: a randomization
p-value is a count of sign assignments, which is small enough to enumerate by hand for these fixtures, and the design
effect is the textbook ``1 + (m̄ - 1) * ICC`` for a perfectly correlated cluster. The properties that matter most are
the two that make this a *generalisation* rather than a second instrument: with one item per cluster it must reproduce
exact McNemar, and with all items in one work it must report that nothing can be tested.
"""

from __future__ import annotations

import itertools
import math

import pytest

from chakaso.training.paired_stats import mcnemar_exact
from chakaso.training.paired_verdicts import (
    ClusteredVerdictError,
    clustered_paired_verdicts,
)

_SEED = 7


def _flip_p_by_hand(sums: list[float]) -> float:
    """The two-sided randomization p-value, enumerated over every sign assignment by an independent route.

    Deliberately written as ``itertools.product`` over all ``2 ** clusters`` assignments: the module under test
    halves the work by symmetry, so this is a second implementation of the same count, not a restatement of its loop.
    """
    observed = abs(sum(sums))
    extreme = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=len(sums)):
        value = abs(sum(sign * part for sign, part in zip(signs, sums, strict=True)))
        if value >= observed - 1e-12:
            extreme += 1
        total += 1
    return extreme / total


# ------------------------------------------------------------------ the estimate


def test_the_point_estimate_is_the_paired_difference_of_the_two_accuracies() -> None:
    result = clustered_paired_verdicts(
        [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], ["a", "b", "c", "d"], seed=_SEED
    )

    assert result["mean_accuracy_first"] == 0.5
    assert result["mean_accuracy_second"] == 0.5
    assert result["mean_difference"] == 0.0
    assert result["counts"]["first_passes"] == 2
    assert result["counts"]["second_passes"] == 2


def test_a_unanimous_difference_of_four_items_over_four_works_is_one_bit_of_evidence_each() -> None:
    result = clustered_paired_verdicts(
        [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], ["w0", "w1", "w2", "w3"], seed=_SEED
    )

    assert result["mean_difference"] == 1.0
    # 4 discordant items, all one way: McNemar's exact two-sided p is 2 · (1/16).
    assert result["item_level_mcnemar"]["discordant"] == [4, 0]
    assert result["item_level_mcnemar"]["p_value"] == pytest.approx(0.125)
    assert result["cluster_sign_flip"]["p_value"] == pytest.approx(0.125)
    assert result["cluster_sign_flip"]["smallest_reportable_p"] == pytest.approx(0.125)


# ------------------------------------------------------------------ the equivalence that makes it a generalisation


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ([1.0] * 8 + [0.0] * 4, [0.0] * 8 + [1.0] * 4),
        ([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
    ],
)
def test_one_item_per_work_gives_exactly_the_p_mcnemar_gives_them(
    first: list[float], second: list[float]
) -> None:
    """The null McNemar randomizes is the item's sign; with singletons that is all this test can flip.

    This is the property that says the clustered reading is McNemar *with* works rather than a different test:
    same vectors, same p, to the last bit. Pass-or-fail vectors only — the case where a tie is scored is
    :func:`test_a_tie_against_a_fail_is_half_an_item_here_and_no_item_for_mcnemar`, where the two readings
    deliberately disagree.
    """
    names = [f"w{index}" for index in range(len(first))]
    result = clustered_paired_verdicts(first, second, names, seed=_SEED)

    _, _, _, mcnemar_p = mcnemar_exact(first, second)
    assert result["cluster_sign_flip"]["method"] == "exact enumeration"
    assert result["cluster_sign_flip"]["p_value"] == mcnemar_p
    assert result["design_effect"]["ratio"] == pytest.approx(1.0)
    assert result["design_effect"]["effective_items"] == pytest.approx(len(first))


def test_the_enumerated_p_equals_an_independent_enumeration_of_every_assignment() -> None:
    """A mixed-sign case, where the observed estimate is not the largest possible one so the count is not trivial.

    Seven singleton works with differences ``[1, 1, -1, 1, 1, -1, 1]``: the observed estimate is 3 of a possible 7,
    and the assignments that reach 3 or more are the 58 with at most two or at least five terms flipped.
    """
    differences = [1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0]
    first = [1.0 if delta > 0 else 0.0 for delta in differences]
    second = [0.0 if delta > 0 else 1.0 for delta in differences]
    result = clustered_paired_verdicts(
        first, second, [f"w{index}" for index in range(7)], seed=_SEED
    )

    assert result["mean_difference"] == pytest.approx(3 / 7)
    assert result["cluster_sign_flip"]["p_value"] == pytest.approx(_flip_p_by_hand(differences))
    assert result["cluster_sign_flip"]["p_value"] == pytest.approx(58 / 128)


# ------------------------------------------------------------------ why the two readings differ


def test_the_same_disagreement_inside_one_work_cannot_be_tested_at_all() -> None:
    """Ten items one way: McNemar says p = 0.00195, the clustered reading says there is nothing to conclude.

    Flipping the only work negates the estimate, so every assignment is exactly as extreme as the observed one.
    This is the case the item-level p gets wrong, and it is why E2633's leave-one-out range crossed zero.
    """
    first = [1.0] * 10
    second = [0.0] * 10
    result = clustered_paired_verdicts(first, second, ["one-book"] * 10, seed=_SEED)

    assert result["clusters"] == 1
    assert result["cluster_sign_flip"]["p_value"] == 1.0
    assert result["item_level_mcnemar"]["p_value"] == pytest.approx(0.001953125)
    assert result["interval"] is None
    assert result["excludes_zero"] is None
    assert "resampling works cannot move the estimate" in result["interval_note"]
    assert result["leave_one_cluster_out"] is None


def test_four_works_are_worth_more_than_two_works_at_the_same_item_count() -> None:
    spread = clustered_paired_verdicts(
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        ["w0", "w1", "w2", "w3"],
        seed=_SEED,
    )
    paired = clustered_paired_verdicts(
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        ["w0", "w0", "w1", "w1"],
        seed=_SEED,
    )

    assert spread["items"] == paired["items"] == 4
    assert spread["cluster_sign_flip"]["p_value"] == pytest.approx(0.125)
    # Two works: the assignments are ++ and +-, whose estimates are 1 and 0, so half of them are as extreme.
    assert paired["cluster_sign_flip"]["p_value"] == pytest.approx(0.5)
    assert paired["cluster_sign_flip"]["smallest_reportable_p"] == pytest.approx(0.5)


def test_a_tie_against_a_fail_is_half_an_item_here_and_no_item_for_mcnemar() -> None:
    """McNemar conditions on items one arm passed and the other did not; a tie is neither, so it drops them.

    Two works, each a tie for the first arm and a fail for the second: no discordant item by McNemar's definition
    (p = 1.0), while the paired difference is +0.5 in both and the randomization reading gives the assignments'
    own answer.
    """
    result = clustered_paired_verdicts([0.5, 0.5], [0.0, 0.0], ["w0", "w1"], seed=_SEED)

    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["item_level_mcnemar"]["discordant"] == [0, 0]
    assert result["item_level_mcnemar"]["p_value"] == 1.0
    assert result["cluster_sign_flip"]["p_value"] == pytest.approx(_flip_p_by_hand([0.5, 0.5]))


def test_both_arms_tied_on_every_item_is_reported_as_no_difference_not_as_a_degenerate_failure() -> (
    None
):
    result = clustered_paired_verdicts(
        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], ["w0", "w1", "w2"], seed=_SEED
    )

    assert result["mean_difference"] == 0.0
    assert result["cluster_sign_flip"]["p_value"] == 1.0
    assert result["counts"]["both_tied"] == 3
    assert result["signal_spread"]["discordant_items"] == 0
    assert result["signal_spread"]["largest_work"] is None
    assert result["design_effect"] is None
    assert result["interval"] == [0.0, 0.0]
    assert result["excludes_zero"] is False


# ------------------------------------------------------------------ the design effect


def test_two_items_that_always_move_together_are_worth_one_observation() -> None:
    """1 + (m̄ - 1) * ICC with m̄ = 2 and ICC = 1 is 2.0, by hand: half the items are duplicates of their partner."""
    result = clustered_paired_verdicts(
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        ["w0", "w0", "w1", "w1"],
        seed=_SEED,
    )

    assert result["mean_difference"] == 0.0
    assert result["design_effect"]["ratio"] == pytest.approx(2.0)
    assert result["design_effect"]["effective_items"] == pytest.approx(2.0)


def test_an_uneven_set_of_works_reports_the_sizes_it_actually_used() -> None:
    result = clustered_paired_verdicts(
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ["big", "big", "big", "small", "small", "one"],
        seed=_SEED,
    )

    assert result["items"] == 6
    assert result["clusters"] == 3
    assert result["cluster_sizes"] == {"minimum": 1, "maximum": 3}
    assert result["signal_spread"]["works_with_discordance"] == 2
    assert result["signal_spread"]["largest_work"] == "big"
    assert result["signal_spread"]["largest_work_share"] == pytest.approx(3 / 5)


# ------------------------------------------------------------------ the interval and the seed


def test_the_interval_resamples_works_and_carries_the_seed_that_drew_it() -> None:
    first = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    second = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    names = ["w0", "w1", "w2", "w3", "w4", "w5"]

    result = clustered_paired_verdicts(first, second, names, seed=11, resamples=200)
    again = clustered_paired_verdicts(first, second, names, seed=11, resamples=200)

    assert result["interval"] == again["interval"]
    assert result["bootstrap"] == {"seed": 11, "resamples": 200}
    assert result["interval"][0] <= result["mean_difference"] <= result["interval"][1]


def test_a_difference_that_survives_every_work_has_an_interval_clear_of_zero() -> None:
    """Eleven works favour one arm and one favours the other: the resampled bounds stay above zero, the sign is stable.

    The one contrary work is what makes this a reading rather than a tautology — its weight in a draw can pull the
    estimate down, and the interval still does not reach zero at twelve works.
    """
    first = [1.0] * 11 + [0.0]
    second = [0.0] * 11 + [1.0]

    result = clustered_paired_verdicts(
        first, second, [f"w{i}" for i in range(12)], seed=9, resamples=500
    )

    assert result["mean_difference"] == pytest.approx(10 / 12)
    assert result["interval"][0] > 0.0
    assert result["excludes_zero"] is True
    assert result["cluster_sign_flip"]["p_value"] < 0.05


def test_a_lower_confidence_level_buys_a_narrower_interval_on_the_same_draws() -> None:
    first = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    second = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    names = ["w0", "w0", "w1", "w1", "w2", "w2", "w3", "w3"]

    wide = clustered_paired_verdicts(first, second, names, seed=3, probability=0.95)
    tight = clustered_paired_verdicts(first, second, names, seed=3, probability=0.5)

    assert tight["interval"][0] >= wide["interval"][0]
    assert tight["interval"][1] <= wide["interval"][1]


def test_the_resampled_path_agrees_with_the_exact_one_it_replaces() -> None:
    """Twenty singleton works are 2 ** 19 assignments — past enumeration — so this checks convergence, not faith.

    With one item per work the exact answer is McNemar's, so the resampled p-value is compared against a value
    derived by an independent route rather than against itself.
    """
    first = [1.0] * 12 + [0.0] * 8
    second = [0.0] * 12 + [1.0] * 8
    names = [f"w{index}" for index in range(20)]

    result = clustered_paired_verdicts(first, second, names, seed=5, resamples=4000)
    exact = mcnemar_exact(first, second)[3]

    assert result["cluster_sign_flip"]["method"] == "resampled assignments"
    assert result["cluster_sign_flip"]["p_value"] == pytest.approx(exact, abs=0.03)
    assert result["cluster_sign_flip"]["seed"] == 5


# ------------------------------------------------------------------ which work carries it


def test_the_leave_one_out_table_names_the_work_the_headline_rests_on() -> None:
    """Four items of one work and nothing else disagree: remove that work and the difference is zero."""
    result = clustered_paired_verdicts(
        [1.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
        ["carries-it", "carries-it", "carries-it", "carries-it", "quiet", "quiet"],
        seed=_SEED,
    )

    table = result["leave_one_cluster_out"]
    assert table["most_influential_work"] == "carries-it"
    assert table["estimate_without"]["carries-it"] == 0.0
    assert table["estimate_without"]["quiet"] == 1.0
    assert table["minimum"] == 0.0
    assert table["maximum"] == 1.0
    assert result["mean_difference"] == pytest.approx(4 / 6)
    assert result["interval"][0] <= 0.0 <= result["interval"][1]
    assert result["excludes_zero"] is False


# ------------------------------------------------------------------ refusals


def test_an_item_scored_by_one_arm_and_not_the_other_is_refused_not_dropped() -> None:
    with pytest.raises(ClusteredVerdictError, match="one verdict and one cluster per item"):
        clustered_paired_verdicts([1.0, 1.0], [0.0], ["w0", "w1"], seed=_SEED)


def test_a_missing_observation_read_as_a_non_number_is_refused() -> None:
    with pytest.raises(ClusteredVerdictError, match="not a finite verdict pair"):
        clustered_paired_verdicts([1.0, math.nan], [0.0, 1.0], ["w0", "w1"], seed=_SEED)


def test_a_verdict_outside_the_item_is_refused_rather_than_rescaled() -> None:
    with pytest.raises(ClusteredVerdictError, match="share of the item answered"):
        clustered_paired_verdicts([1.0, 1.4], [0.0, 1.0], ["w0", "w1"], seed=_SEED)


def test_an_unnamed_work_is_refused_because_its_items_would_pool_into_one_cluster() -> None:
    with pytest.raises(ClusteredVerdictError, match="blank one"):
        clustered_paired_verdicts([1.0, 0.0], [0.0, 1.0], ["w0", " "], seed=_SEED)


def test_no_items_is_not_a_measurement() -> None:
    with pytest.raises(ClusteredVerdictError, match="zero items"):
        clustered_paired_verdicts([], [], [], seed=_SEED)


@pytest.mark.parametrize(
    ("resamples", "probability"), [(0, 0.95), (2.5, 0.95), (True, 0.95), (10, 1.0)]
)
def test_an_unusable_resample_count_or_confidence_is_refused(
    resamples: int, probability: float
) -> None:
    with pytest.raises(ClusteredVerdictError):
        clustered_paired_verdicts(
            [1.0], [0.0], ["w0"], seed=_SEED, resamples=resamples, probability=probability
        )


def test_the_clustered_error_is_the_package_error_type() -> None:
    from chakaso.training.errors import TrainingError

    assert issubclass(ClusteredVerdictError, TrainingError)


def test_the_reading_that_is_not_a_capability_claim_travels_in_the_artifact() -> None:
    result = clustered_paired_verdicts([1.0, 1.0], [0.0, 0.0], ["w0", "w1"], seed=_SEED)

    assert "not a capability claim" in result["claim_limit"]
    assert "independent" in result["item_level_mcnemar"]["assumption"]
