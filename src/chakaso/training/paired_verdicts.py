"""A paired test on binary item verdicts, randomized and resampled by work rather than by row.

:mod:`chakaso.training.paired_loss` made a held-out loss difference into a test by resampling the *works* behind it,
and E2633 showed why: a difference the pooled estimate called large crossed zero once one book was removed. The
capability probes have the same dependence and a sharper version of it. Their per-item verdicts are compared with
exact McNemar, which conditions on the discordant items and treats each of them as an independent coin. Two models
asked 745 passages from 25 held-out works have not made 745 independent observations, and E2634's own limits section
says its p-values should be read as **"optimistic by an unknown margin"**. This module replaces *unknown* with a
number.

The statistic is the paired difference of the verdict vectors, ``d_i = first_i - second_i``, averaged over the items:
the accuracy difference the record prints, read as paired rather than as two marginal intervals. Two readings of its
uncertainty come out, and they answer different questions:

* :func:`clustered_paired_verdicts`' **sign-flip p-value** is a randomization test. Under the null that the two arms
  are interchangeable *within* a work, swapping the arm labels of a whole work is as likely as not, so the null
  distribution is over the ``2 ** clusters`` assignments of one sign per work. This is exact — no asymptotics, no
  normal approximation — and it is *the* reading the project should claim a separation on. Enumerated exactly while
  the assignment count is small, resampled beyond that, and the method is stated in the result either way.
* the **cluster bootstrap interval** resamples works with replacement and re-computes the same mean, which is what
  says how well the size of the difference is known rather than whether it is nonzero.

Both reduce to what E2626-E2634 already computed when every item is its own cluster: on pass-or-fail verdicts the
sign-flip p-value is exactly the two-sided exact McNemar p, and the design effect is exactly 1 whatever the verdicts.
A tie is where the two part even then — McNemar discards an item where neither arm passed, including a
pass-versus-tie, while this reading counts it at a half — so the equivalence is asserted on tie-free vectors against
:func:`chakaso.training.paired_stats.mcnemar_exact`, and the tie case is asserted separately rather than smoothed
over.

Three things the reading is *not*, and the artifact says them in its own fields:

* Not a capability claim. A separation here says these arms answered these items differently, with the works behind
  them taken into account. It is not evidence that either model is useful, and K-001 is unchanged by it.
* Not a replacement for McNemar's conditioning. McNemar counts only the items the arms answered *unlike*; this test
  averages every paired item, so a pass-versus-tie enters at a half. On 0/0.5/1 verdicts the two agree in sign and
  can disagree in size, and the reported pair of p-values is where a reader sees that.
* Not a fix for few clusters. With one work the sign-flip p is 1.0 by construction — flipping the only work negates
  the estimate, so every assignment is exactly as extreme — and the bootstrap interval is ``null``, because
  resampling one cluster cannot move anything. That is the honest statement, and it is reported as an absence rather
  than a number.

Standard library only, so the statistic runs in the dependency-free CI matrix; the scoring that produces its inputs
needs ``[nn]`` and lives in the probe runners.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from chakaso.training.errors import TrainingError
from chakaso.training.paired_stats import mcnemar_exact, percentile

__all__ = ["ClusteredVerdictError", "clustered_paired_verdicts"]

#: Sign assignments are enumerated exactly up to this many, so a small sample gets an exact randomization p-value
#: instead of a Monte Carlo one. 32,768 assignments is a fraction of a second here and covers 16 clusters.
_EXACT_ASSIGNMENT_LIMIT = 1 << 15

#: Float summation order is not part of a randomization test, so "at least as extreme" is read with this slack.
_EXTREME_SLACK = 1e-12

#: The observed assignment and its global negation are the two most extreme of the ``2 ** clusters`` label swaps,
#: so the smallest two-sided p-value an exact randomization test over ``clusters`` works can report is 2 over that
#: many assignments. Reported with the p-value because a reader of ``p = 0.0625`` on three works should know that
#: 0.25 was the floor.
_MIN_EXTREME_ASSIGNMENTS = 2


class ClusteredVerdictError(TrainingError):
    """A clustered paired verdict test was asked about vectors or clusters that do not line up."""


def _validate(
    first: Sequence[float], second: Sequence[float], clusters: Sequence[object]
) -> tuple[list[float], list[float], list[str]]:
    """The three sequences as lists, refusing a comparison that does not describe one item set."""
    count = len(first)
    if count == 0:
        message = "a clustered paired verdict test over zero items is not a measurement"
        raise ClusteredVerdictError(message)
    if len(second) != count or len(clusters) != count:
        message = (
            f"{count} items for the first arm, {len(second)} for the second and {len(clusters)} "
            "cluster keys; a paired test needs one verdict and one cluster per item"
        )
        raise ClusteredVerdictError(message)
    values_first = [float(value) for value in first]
    values_second = [float(value) for value in second]
    names = [str(cluster) for cluster in clusters]
    for index, (a, b) in enumerate(zip(values_first, values_second, strict=True)):
        if not math.isfinite(a) or not math.isfinite(b):
            message = f"item {index} is not a finite verdict pair: {a!r} against {b!r}"
            raise ClusteredVerdictError(message)
        if not 0.0 <= a <= 1.0 or not 0.0 <= b <= 1.0:
            message = (
                f"item {index} has verdicts {a!r} and {b!r}; a verdict is a share of the item answered — "
                "1.0 a pass, 0.0 a fail, 0.5 a tie"
            )
            raise ClusteredVerdictError(message)
    if any(not name.strip() for name in names):
        message = (
            "every item needs a cluster name; a blank one would pool verdicts into an unnamed work"
        )
        raise ClusteredVerdictError(message)
    return values_first, values_second, names


def _cluster_sums(
    names: Sequence[str], differences: Sequence[float]
) -> tuple[list[str], dict[str, float], dict[str, int]]:
    """The work names in order of first appearance, with each one's difference-sum and item count.

    First-appearance order is what makes a seeded draw reproducible across callers: the resampler draws over this
    list, so a set built by iterating a ``set()`` would move the interval without moving the data.
    """
    order: list[str] = []
    sums: dict[str, float] = {}
    sizes: dict[str, int] = {}
    for name, delta in zip(names, differences, strict=True):
        if name not in sums:
            order.append(name)
            sums[name] = 0.0
            sizes[name] = 0
        sums[name] += delta
        sizes[name] += 1
    return order, sums, sizes


def _sign_flip_extreme_count(values: Sequence[float], observed_sum: float) -> int:
    """How many of the ``2 ** (clusters - 1)`` assignments with the first work unflipped are this extreme.

    ``ε`` and ``-ε`` give the same absolute estimate, so half the assignments determine the other half and the
    two-sided p-value over the half is the p-value over the whole.
    """
    extreme = 0
    for assignment in range(1 << (len(values) - 1)):
        total = values[0]
        for position in range(1, len(values)):
            total += -values[position] if (assignment >> (position - 1)) & 1 else values[position]
        if abs(total) >= observed_sum - _EXTREME_SLACK:
            extreme += 1
    return extreme


def _sign_flip_p_value(
    names: Sequence[str],
    sums: dict[str, float],
    observed_sum: float,
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    """The two-sided randomization p-value for the paired mean, over assignments of one sign per work.

    The statistic is compared as a *sum* of the works' difference-sums; dividing by the item count is a constant, so
    it moves neither the ordering nor the p-value, and keeping the sum avoids a second rounding of every draw.
    """
    values = [sums[name] for name in names]
    assignments = 1 << (len(names) - 1)
    floor = _MIN_EXTREME_ASSIGNMENTS / (2.0 ** len(names))
    if assignments <= _EXACT_ASSIGNMENT_LIMIT:
        extreme = _sign_flip_extreme_count(values, observed_sum)
        return {
            "p_value": extreme / assignments,
            "method": "exact enumeration",
            "assignments": assignments,
            "smallest_reportable_p": floor,
            "note": (
                "every sign assignment of the works was enumerated, so this p-value is exact for the null that "
                "the two arms are interchangeable within each work"
            ),
        }
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        total = 0.0
        for value in values:
            total += -value if rng.getrandbits(1) else value
        if abs(total) >= observed_sum - _EXTREME_SLACK:
            extreme += 1
    return {
        "p_value": (1 + extreme) / (1 + resamples),
        "method": "resampled assignments",
        "assignments": assignments,
        "resamples": resamples,
        "seed": seed,
        "smallest_reportable_p": 1.0 / (1 + resamples),
        "note": (
            f"{assignments} assignments is more than this code enumerates, so {resamples} were drawn; "
            "the p-value carries Monte Carlo error and its seed is reported with it"
        ),
    }


def clustered_paired_verdicts(
    first: Sequence[float],
    second: Sequence[float],
    clusters: Sequence[object],
    *,
    seed: int,
    resamples: int = 2000,
    probability: float = 0.95,
) -> dict[str, object]:
    """Compare two arms item by item over one probe, taking the work behind each item into account.

    ``first`` and ``second`` are the two models' per-item verdicts on the same items in the same order — 1.0 a pass,
    0.5 a tie, 0.0 a fail, as :mod:`chakaso.training.paired_stats` reads them — and ``clusters`` names, for each
    item, the unit whose items are not independent of each other. For a probe cut from a corpus that is the held-out
    work (``TextUnit.group``, carried on the artifact by :mod:`chakaso.training.item_clusters`); for the authored
    pair probe it is the phenomenon. The caller does not have to prove which: the result echoes
    ``clusters_used`` as the distinct names it saw.

    ``seed`` and ``resamples`` travel with the result because a resampled p-value and a bootstrap interval are only
    reproducible if their parameters do.

    Raises:
        ClusteredVerdictError: the three sequences disagree in length or are empty, a verdict is not a finite
            fraction of its item, a cluster name is blank, or the resample count and confidence are not usable.
    """
    values_first, values_second, names = _validate(first, second, clusters)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        message = f"resamples must be an integer >= 1, got {resamples!r}"
        raise ClusteredVerdictError(message)
    if not 0.0 < probability < 1.0:
        message = f"probability must lie strictly between 0 and 1, got {probability!r}"
        raise ClusteredVerdictError(message)

    count = len(values_first)
    differences = [a - b for a, b in zip(values_first, values_second, strict=True)]
    order, sums, sizes = _cluster_sums(names, differences)
    estimate = math.fsum(differences) / count
    work_names = list(order)

    # The bootstrap resamples works, so a work that enters twice brings all its items with it. Pooling the sums of
    # the drawn works is the estimate on that resample; a single work leaves every draw identical, which is why the
    # interval is then reported as absent rather than as a point.
    draws: list[float] = []
    if len(work_names) > 1:
        rng = random.Random(seed)
        for _ in range(resamples):
            picked = rng.choices(work_names, k=len(work_names))
            numerator = math.fsum(sums[name] for name in picked)
            denominator = math.fsum(sizes[name] for name in picked)
            draws.append(numerator / denominator)
        draws.sort()
        tail = (1.0 - probability) / 2.0
        interval: list[float] | None = [percentile(draws, tail), percentile(draws, 1.0 - tail)]
        interval_note = ""
    else:
        interval = None
        interval_note = (
            "one cluster: resampling works cannot move the estimate, so no interval is available and the "
            "difference is known only as well as one work can know it"
        )

    sign_flip = _sign_flip_p_value(
        work_names,
        sums,
        abs(math.fsum(sums[name] for name in work_names)),
        seed=seed,
        resamples=resamples,
    )

    # The design effect is the ratio of two variances of the *same* estimate: what clustering the items into works
    # costs in independent information. Both are written without a small-sample correction so that one item per
    # cluster gives exactly 1.0 — the property that makes the number comparable with the item-level readings. The
    # cluster sum of the centred differences is `sums[name] - sizes[name] * estimate`, which keeps this one pass.
    item_variance = math.fsum((delta - estimate) ** 2 for delta in differences) / (count * count)
    cluster_variance = math.fsum(
        (sums[name] - sizes[name] * estimate) ** 2 for name in work_names
    ) / (count * count)
    design_effect: dict[str, object] | None
    if item_variance > 0.0:
        ratio = cluster_variance / item_variance
        design_effect = {
            "ratio": ratio,
            "effective_items": count / ratio if ratio > 0.0 else None,
            "note": (
                "1.0 is what independent items give; above it the items of one work move together and the "
                "item-level readings are optimistic by that factor"
            ),
        }
    else:
        design_effect = None

    # McNemar's exact test on the same vectors, as the optimistic reading: same items, same pairing, no clusters.
    b, c, discordant, mcnemar_p = mcnemar_exact(values_first, values_second)
    # E2634 had to rebuild its item set from the corpus to answer "how much of this separation is one book?". The
    # answer is computed here so the next record does not: which works hold the disagreement, and how much of it
    # the largest of them holds.
    discordance: dict[str, int] = {}
    for name, delta in zip(names, differences, strict=True):
        if delta != 0.0:
            discordance[name] = discordance.get(name, 0) + 1
    if discordant:
        largest_work = max(
            discordance, key=lambda name: (discordance[name], -work_names.index(name))
        )
        spread = {
            "discordant_items": discordant,
            "works_with_discordance": len(discordance),
            "largest_work": largest_work,
            "largest_work_share": discordance[largest_work] / discordant,
        }
    else:
        spread = {
            "discordant_items": 0,
            "works_with_discordance": 0,
            "largest_work": None,
            "largest_work_share": None,
        }

    leave_out: dict[str, object] | None = None
    if len(work_names) > 1:
        without = {
            name: (estimate * count - sums[name]) / (count - sizes[name]) for name in work_names
        }
        influential = max(
            work_names, key=lambda name: (abs(without[name] - estimate), -work_names.index(name))
        )
        leave_out = {
            "estimate_without": without,
            "most_influential_work": influential,
            "most_influential_difference": without[influential] - estimate,
            "minimum": min(without.values()),
            "maximum": max(without.values()),
        }

    return {
        "unit": "paired verdict difference, first minus second; 1.0 a pass, 0.5 a tie, 0.0 a fail",
        "items": count,
        "clusters": len(work_names),
        "cluster_sizes": {"minimum": min(sizes.values()), "maximum": max(sizes.values())},
        "mean_accuracy_first": math.fsum(values_first) / count,
        "mean_accuracy_second": math.fsum(values_second) / count,
        "mean_difference": estimate,
        "interval": interval,
        "interval_note": interval_note,
        "probability": probability,
        "excludes_zero": (None if interval is None else (interval[0] > 0.0 or interval[1] < 0.0)),
        "cluster_sign_flip": sign_flip,
        "item_level_mcnemar": {
            "discordant": [b, c],
            "n": discordant,
            "p_value": mcnemar_p,
            "assumption": (
                "each discordant item is an independent coin — optimistic when a work's items move together, "
                "and it conditions on the discordant items rather than averaging every paired one"
            ),
        },
        "design_effect": design_effect,
        "signal_spread": spread,
        "leave_one_cluster_out": leave_out,
        "counts": {
            "first_passes": sum(1 for value in values_first if value > 0.5),
            "second_passes": sum(1 for value in values_second if value > 0.5),
            "both_tied": sum(
                1
                for a, b_value in zip(values_first, values_second, strict=True)
                if a == 0.5 == b_value
            ),
            "items_first_ahead": sum(1 for delta in differences if delta > 0.0),
            "items_second_ahead": sum(1 for delta in differences if delta < 0.0),
        },
        "bootstrap": {"seed": seed, "resamples": resamples},
        "claim_limit": (
            "a separation here says these two arms answered these items differently once the works behind them "
            "were counted as clusters; it is not evidence that either model is useful, and it is not a "
            "capability claim"
        ),
    }
