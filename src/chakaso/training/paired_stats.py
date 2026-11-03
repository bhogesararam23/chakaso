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
"""

from __future__ import annotations

from collections.abc import Sequence
from math import comb, fabs

from chakaso.training.errors import TrainingError

__all__ = ["holm_rejections", "mcnemar_exact", "spearman_rho"]


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
    return covariance / (spread_a**0.5 * spread_b**0.5)


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
