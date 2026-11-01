"""Reading a prose-share sweep without declaring a winner the noise does not support (H9).

A sweep of training mixtures produces, for every share, one bits-per-character number per register. Which
share is "best" is not a property of the runs alone: it depends on how the registers are weighted, and the
repository's own whole-pack number weights them by an accident of one seeded sample (E2612's whole test pack is
41.9 % prose, 45.3 % science, 5.8 % technology, 7.0 % mathematics and no medicine). So the primary reading here
fixes the weighting in advance as the **equal-weight mean of the register partitions**, and everything else is
reported per register.

Two ideas keep the reading honest. A difference is judged against the seed noise E2614 measured, in three
grades (a tie, indeterminate, separable) rather than as above or below one line. And an optimum is claimed only
when the best arm beats the arms it must beat by more than the separable margin; otherwise the outcome is
``unresolved`` and the arms within reach of the best are listed instead of one being chosen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "Margins",
    "classify_difference",
    "classify_share_sweep",
    "dominance",
    "equal_weight_mean",
    "margins_from_stdev",
]

#: A 95 % two-sided band for the difference of two single-seed runs: 1.96 * sqrt(2) standard deviations.
PAIR_FACTOR = 2.77

#: Slack for floating-point subtraction, so a difference that is exactly the margin on paper is not
#: classified by the last bit of a double.
_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Margins:
    """The two thresholds a difference between two single-seed arms is judged against.

    ``central`` is the band from the sample standard deviation: at or under it a difference is a tie. ``upper``
    is the band from the upper end of that deviation's 95 % interval: over it a difference is separable even if
    the true spread is as large as the interval allows. Between the two the difference is indeterminate, and the
    record says it needs more seeds.
    """

    central: float
    upper: float


def margins_from_stdev(
    stdev: float,
    stdev_upper: float,
    *,
    declared_band: float | None = None,
    band_replaces_central: bool = False,
) -> Margins:
    """Margins from a measured spread, honouring a band the repository already declared.

    ``declared_band`` is a threshold E2614 kept or set: 0.02 for prose (kept, because it sat above the
    interval) and 0.035 for mathematics (set, because 0.02 sat inside the noise). A kept band can only raise the
    separable margin; a replacement band (``band_replaces_central``) also stands in for the central one.
    """
    central = PAIR_FACTOR * stdev
    upper = PAIR_FACTOR * stdev_upper
    if declared_band is not None:
        upper = max(upper, declared_band)
        if band_replaces_central:
            central = declared_band
    return Margins(central=central, upper=upper)


def classify_difference(difference: float, margins: Margins) -> str:
    """``tie``, ``indeterminate`` or ``separable`` for the size of a difference between two arms."""
    size = abs(difference)
    if size <= margins.central + _EPSILON:
        return "tie"
    if size <= margins.upper + _EPSILON:
        return "indeterminate"
    return "separable"


def equal_weight_mean(values: Mapping[str, float], partitions: Sequence[str]) -> float:
    """The unweighted mean over ``partitions``; every register counts once however much text it holds."""
    missing = [partition for partition in partitions if partition not in values]
    if missing:
        message = f"no value for partitions {missing}"
        raise ValueError(message)
    if not partitions:
        message = "an equal-weight mean needs at least one partition"
        raise ValueError(message)
    return sum(values[partition] for partition in partitions) / len(partitions)


def classify_share_sweep(
    objective_by_share: Mapping[float, float], margin: float
) -> dict[str, object]:
    """Say what a sweep of one objective over training shares supports, and no more.

    Outcomes:

    * ``interior_optimum`` - the best arm is not an endpoint and beats **both** endpoints by more than ``margin``.
    * ``boundary_optimum`` - the best arm is an endpoint and beats **every other arm** by more than ``margin``.
    * ``unresolved`` - anything else, including a best arm that beats only one endpoint, or arms that are
      within ``margin`` of one another. No share is chosen; ``within_margin_of_best`` lists the arms the data
      cannot tell apart from the best one.

    Lower is better. ``margin`` is the separable margin for the objective (see :class:`Margins`).
    """
    if len(objective_by_share) < 3:
        message = "a sweep needs at least three shares to have an interior"
        raise ValueError(message)
    if margin < 0:
        message = f"margin must be >= 0, got {margin!r}"
        raise ValueError(message)
    shares = sorted(objective_by_share)
    low, high = shares[0], shares[-1]
    best = min(shares, key=lambda share: (objective_by_share[share], share))
    best_value = objective_by_share[best]
    others = [share for share in shares if share != best]
    within = [
        share for share in shares if objective_by_share[share] - best_value <= margin + _EPSILON
    ]

    def beats(rivals: Sequence[float]) -> bool:
        return all(objective_by_share[rival] - best_value > margin + _EPSILON for rival in rivals)

    if best not in (low, high) and beats([low, high]):
        outcome = "interior_optimum"
    elif best in (low, high) and beats(others):
        outcome = "boundary_optimum"
    else:
        outcome = "unresolved"
    return {
        "outcome": outcome,
        "best_share": best,
        "best_value": best_value,
        "margin": margin,
        "within_margin_of_best": within,
        "values": {share: objective_by_share[share] for share in shares},
    }


def dominance(
    bits_by_share: Mapping[float, Mapping[str, float]], upper: Mapping[str, float]
) -> list[tuple[float, float]]:
    """Ordered pairs ``(a, b)`` where arm ``a`` is better than arm ``b`` on **every** register, separably.

    This is the reading that needs no weighting: if one mixture is separably better everywhere, any weighting
    prefers it. A trade-off between registers - the case a mixture sweep is expected to show - produces no
    pairs, and that absence is the finding, not an omission.
    """
    pairs: list[tuple[float, float]] = []
    for first in sorted(bits_by_share):
        for second in sorted(bits_by_share):
            if first == second:
                continue
            if all(
                bits_by_share[second][register] - bits_by_share[first][register] > upper[register]
                for register in upper
            ):
                pairs.append((first, second))
    return pairs
