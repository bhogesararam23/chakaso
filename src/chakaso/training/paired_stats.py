"""Paired statistics for comparing two measurements of the *same* items.

The repository's capability probes (E2621, E2622, E2623) compared models by setting their Wilson intervals
beside each other. That is the right test for one model against chance and the wrong test for two models on the
same 300 pairs: the pairs are shared, so what carries the information is the items on which the two
*disagree*, and a marginal interval throws that away. E2622's own addendum had to withdraw an ordering because
the probe's seed spread (0.015-0.048 on 300 pairs) exceeded the differences between arms - which is a statement
about the power of the comparison it made, not about the instrument.

Everything here is arithmetic over two equal-length vectors of per-item outcomes, with no numerical library, so
it runs in the dependency-free matrix. Two models scored by the same probe on the same items produce those
vectors; :func:`chakaso.training.minimal_pairs.score_pairs` and
:func:`chakaso.training.continuation_items.score_items` report theirs.

* :func:`mcnemar_exact` - did the two models differ on more items than chance can explain, given that both
  were asked the same questions?
* :func:`spearman_rho` - do two orderings agree, without assuming the gap between neighbours is equal?
* :func:`holm_rejections` - a family of p-values judged together, so a sweep cannot buy a discovery by
  testing enough pairs.
* :func:`student_t_quantile` and :func:`mean_confidence_interval` - the interval a handful of paired draws
  deserves, which is what E2614 asked for when it said a spread from three runs is known only to within a
  factor, and what E2630 needs to say whether a mean difference clears zero.
* :func:`sign_test_p_value` - the exact probability of a run of same-signed draws under a fair coin, which is
  the only test a difference can pass when each draw is one number and nothing per-item is shared.
* :func:`percentile` - the quantile a bootstrap interval is read off its sorted draws, used by both resamplings in
  the series (:mod:`chakaso.training.paired_loss` for loss, :mod:`chakaso.training.paired_verdicts` for verdicts),
  so that an interval in one record and an interval in another mean the same thing by construction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from math import comb, fabs

from chakaso.training.errors import TrainingError

__all__ = [
    "holm_rejections",
    "mcnemar_exact",
    "mean_confidence_interval",
    "percentile",
    "sign_agreement",
    "sign_test_p_value",
    "spearman_rho",
    "student_t_quantile",
]


class PairedStatsError(TrainingError):
    """A paired statistic was asked about vectors that do not describe the same items."""


def _check_vectors(*vectors: Sequence[float]) -> None:
    first = len(vectors[0])
    for vector in vectors:
        if len(vector) != first:
            message = (
                f"paired statistics need equal-length vectors, got {first} and {len(vector)}; two models "
                "scored on different item sets are not a paired comparison"
            )
            raise PairedStatsError(message)
    if first == 0:
        message = "a paired statistic over zero items is not a measurement"
        raise PairedStatsError(message)


def mcnemar_exact(first: Sequence[float], second: Sequence[float]) -> tuple[int, int, int, float]:
    """McNemar's exact test on two vectors of per-item scores: return ``(b, c, n, p)``.

    Only the **discordant** items matter - the ones where one model passed and the other failed. With a tie
    counted as neither a pass nor a fail, ``b`` is the items the first model got right and the second wrong, ``c``
    the reverse, and under the null that the two models are interchangeable each discordant item is a fair coin,
    so ``min(b, c)`` follows ``Binomial(b + c, 0.5)``. The p-value is two-sided and exact: the probability of a
    split at least as lopsided, summed over the tails, never a normal approximation - at ``n`` = 300 the
    approximation is adequate, and this is not a place to be adequate.

    Items on which both models tie are not discordant and are not counted: a tie carries no direction. A
    comparison with no discordant item returns ``p = 1.0`` - two models that answered every item alike have not
    been shown to differ.
    """
    _check_vectors(first, second)
    b = c = 0
    for a, d in zip(first, second, strict=True):
        a_won, d_won = a > 0.5, d > 0.5
        if a_won and not d_won:
            b += 1
        elif d_won and not a_won:
            c += 1
    total = b + c
    if total == 0:
        return 0, 0, 0, 1.0
    lesser = min(b, c)
    tail = sum(comb(total, k) for k in range(lesser + 1)) / 2.0**total
    p_value = min(1.0, 2.0 * tail)
    return b, c, total, p_value


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks over ties, lowest value first."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    out: list[float] = [0.0] * len(values)
    position = 0
    while position < len(order):
        cursor = position
        while cursor + 1 < len(order) and values[order[cursor + 1]] == values[order[position]]:
            cursor += 1
        average = (position + cursor) / 2.0 + 1.0
        for index in order[position : cursor + 1]:
            out[index] = average
        position = cursor + 1
    return out


def spearman_rho(first: Sequence[float], second: Sequence[float]) -> float:
    """The rank correlation between two vectors: +1 when one ordering is a monotone image of the other.

    Taken over ranks, not values, because the gaps between two learning stages or two mixtures are not equal
    and a linear correlation would pay for a big jump in the middle of the range and ignore a consistent order
    elsewhere. Ties receive average ranks, so an instrument that reports the same score for several rungs is
    penalised rather than rewarded. With fewer than two points there is no ordering to compare and the result
    is 0.0, not a division by zero dressed as a number.
    """
    _check_vectors(first, second)
    if len(first) < 2:
        return 0.0
    a = _ranks(first)
    b = _ranks(second)
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b, strict=True))
    spread_a = sum((x - mean_a) ** 2 for x in a)
    spread_b = sum((y - mean_b) ** 2 for y in b)
    if spread_a == 0.0 or spread_b == 0.0:
        return 0.0
    # `float(...)` and `math.sqrt` rather than `x ** 0.5`: a power of a sum is typed loosely enough that
    # mypy reports the quotient as Any, and the division below has no business being Any.
    return float(covariance / (math.sqrt(spread_a) * math.sqrt(spread_b)))


def holm_rejections(p_values: Sequence[float], *, alpha: float = 0.05) -> list[bool]:
    """Which p-values survive Holm's step-down correction, in the order they were given.

    A sweep of rung pairs, register partitions or phenomena is a *family* of tests, and testing enough of them
    buys a false positive: with thirty comparisons at 0.05 the expected count of spurious rejections is
    greater than one. Holm orders the p-values ascending and compares the (i+1)-th smallest against
    ``alpha / (m - i)``, stopping at the first failure — so a p-value that would clear ``alpha`` on its own is
    *not* rejected when an earlier one failed, which is what makes it a step-down rather than a per-test
    threshold. It controls the family-wise error rate and costs less power than Bonferroni, which charges every
    test the same ``alpha / m`` whether or not an earlier one already stopped the family.
    """
    if not p_values:
        message = "holm_rejections needs at least one p-value"
        raise PairedStatsError(message)
    if not 0.0 < alpha < 1.0:
        message = f"alpha must lie strictly between 0 and 1, got {alpha!r}"
        raise PairedStatsError(message)
    count = len(p_values)
    order = sorted(range(count), key=lambda index: (p_values[index], index))
    rejections = [False] * count
    for rank, index in enumerate(order):
        threshold = alpha / (count - rank)
        if p_values[index] <= threshold:
            rejections[index] = True
        else:
            break
    return rejections


def sign_agreement(first: Sequence[float], second: Sequence[float]) -> float:
    """The share of items on which both vectors say the same thing, counting a tie on either as no answer.

    Reported beside a test because a 0.94 and a 0.96 agree on most items; how much agreement that is tells a
    reader whether a separation came from a handful of flips or from a rebuilt behaviour.
    """
    _check_vectors(first, second)
    same = 0
    answered = 0
    for a, d in zip(first, second, strict=True):
        a_tie, d_tie = fabs(a - 0.5) < 1e-12, fabs(d - 0.5) < 1e-12
        if a_tie or d_tie:
            continue
        answered += 1
        if a == d:
            same += 1
    return same / answered if answered else 1.0


def percentile(sorted_values: Sequence[float], quantile: float) -> float:
    """A linear-interpolated quantile of an already-sorted sequence.

    The bootstrap intervals in :mod:`chakaso.training.paired_loss` and :mod:`chakaso.training.paired_verdicts` read
    their bounds off the sorted draws with this, so the two resamplings state their intervals the same way: one
    function, not two that a reader has to compare.
    """
    if not sorted_values:
        message = "a percentile of no draws is not a bound"
        raise PairedStatsError(message)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """The continued fraction behind :func:`_regularised_incomplete_beta`, evaluated by modified Lentz.

    Numerical Recipes' form, with the odd contractions removed by the standard even/odd pairing. Bounded
    iteration because a continued fraction that has not converged in two hundred steps is not the one being
    evaluated here, and a silently non-terminating statistic is worse than a named failure.
    """
    tiny = 1e-300
    convergence = 3e-14
    contract = 1.0 - (a + b) * x / (a + 1.0)
    if fabs(contract) < tiny:
        contract = tiny
    contract = 1.0 / contract
    expand = 1.0
    result = contract
    for order in range(1, 201):
        even = 2 * order
        coefficient = order * (b - order) * x / ((a - 1 + even) * (a + even))
        contract = 1.0 + coefficient * contract
        if fabs(contract) < tiny:
            contract = tiny
        expand = 1.0 + coefficient / expand
        if fabs(expand) < tiny:
            expand = tiny
        contract = 1.0 / contract
        result *= contract * expand
        coefficient = -(a + order) * (a + b + order) * x / ((a + even) * (a + 1 + even))
        contract = 1.0 + coefficient * contract
        if fabs(contract) < tiny:
            contract = tiny
        expand = 1.0 + coefficient / expand
        if fabs(expand) < tiny:
            expand = tiny
        contract = 1.0 / contract
        step = contract * expand
        result *= step
        if fabs(step - 1.0) < convergence:
            break
    return result


def _regularised_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)``, the CDF of a Beta(a, b) variable at ``x`` — the function a Student-t tail is one line of.

    Evaluated on whichever side converges fast, which is the standard symmetry switch: the fraction is rapid
    for ``x`` below ``(a + 1) / (a + b + 2)`` and the complement is rapid above it.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def _student_t_upper_tail(value: float, degrees: int) -> float:
    """``P(T > value)`` for a Student-t with ``degrees`` degrees of freedom, at ``value`` >= 0."""
    return 0.5 * _regularised_incomplete_beta(
        degrees / 2.0, 0.5, degrees / (degrees + value * value)
    )


def student_t_quantile(probability: float, degrees: int) -> float:
    """The value below which ``probability`` of a Student-t with ``degrees`` degrees of freedom falls.

    Bisection on the upper tail above, for the same reason :func:`chakaso.training.seed_study.chi_square_quantile`
    is: the project needs three digits of a table and does not take a statistics dependency to get them. Used
    at ``probability = 0.975`` for a two-sided 95 % interval on a mean of a few paired draws, where the normal
    quantile 1.96 would be a claim about a sample this project does not have — at four draws the honest
    critical value is 3.182, and the difference is the whole argument of E2614's limits section.
    """
    if not isinstance(degrees, int) or degrees < 1:
        message = f"degrees must be an integer >= 1, got {degrees!r}"
        raise PairedStatsError(message)
    if not 0.0 < probability < 1.0:
        message = f"probability must lie strictly between 0 and 1, got {probability!r}"
        raise PairedStatsError(message)
    if probability <= 0.5:
        # Every use here is an upper quantile; a lower one is the symmetric negative and is not what an
        # interval half-width wants. Refusing beats returning a negative number into a `critical` field.
        message = f"student_t_quantile is the upper quantile, got probability {probability!r}"
        raise PairedStatsError(message)
    target = 1.0 - probability
    low, high = 0.0, 1.0
    while _student_t_upper_tail(high, degrees) > target and high < 1e12:
        high *= 2.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if _student_t_upper_tail(middle, degrees) > target:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def mean_confidence_interval(
    values: Sequence[float], *, probability: float = 0.95
) -> tuple[float, float, float, int, float]:
    """The Student-t interval on the mean of ``values``: ``(mean, low, high, degrees, critical)``.

    Two-sided at ``probability``, using the sample deviation of the draws themselves. Returned with its degrees
    of freedom and critical value rather than only the bounds, because a mean of three draws and a mean of
    thirty with the same interval width are different measurements and a reader should see which they got.
    """
    count = len(values)
    if count < 2:
        message = f"a mean needs at least two draws, got {count}"
        raise PairedStatsError(message)
    if any(not math.isfinite(value) for value in values):
        message = "a mean confidence interval needs finite draws"
        raise PairedStatsError(message)
    if not 0.0 < probability < 1.0:
        message = f"probability must lie strictly between 0 and 1, got {probability!r}"
        raise PairedStatsError(message)
    degrees = count - 1
    mean = math.fsum(values) / count
    deviation = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / degrees)
    critical = student_t_quantile(0.5 + probability / 2.0, degrees)
    half = critical * deviation / math.sqrt(count)
    return mean, mean - half, mean + half, degrees, critical


def sign_test_p_value(successes: int, *, total: int) -> float:
    """The exact one-sided probability that a fair coin is at least this lopsided: ``P(X >= max(k, n-k))``.

    The count read is the *larger* of the two signs, so 3 of 4 and 1 of 4 are the same evidence — which is what
    a sign test means, and is stated here because a reader who assumes ``k`` is read as given would compute
    15/16 for the second. Each draw contributes one bit and no draw contributes a magnitude, so this is the
    test available when each draw is a single difference: E2614's seeds, E2630's paired arms, a register's
    sign across partitions. A run of five same-signed draws is p = 0.03125 and four is 0.0625, which is the
    arithmetic behind E2614's note that three and five seeds buy different claims.
    """
    if total < 1:
        message = f"a sign test needs at least one draw, got total {total}"
        raise PairedStatsError(message)
    if not 0 <= successes <= total:
        message = f"successes must lie between 0 and total, got {successes} of {total}"
        raise PairedStatsError(message)
    extreme = max(successes, total - successes)
    tail = math.fsum(comb(total, index) for index in range(extreme, total + 1))
    return tail / 2.0**total
