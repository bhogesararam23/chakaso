"""Read E2618's seed replication of the prose-share arms by the rules its record declared before any new seed ran.

Each arm is supplied as ``SHARE=study.json``, a seed-study aggregate whose ``per_run`` holds every seed's metrics. The
tool uses the seeds **every arm completed** (a prefix of the round-robin campaign is balanced), computes the primary
objective O1 (the equal-weight mean of the five register partitions) per seed, pools the arms' within-arm variances,
and grades each pair of arms:

* ORDERED     the Student-t interval for the difference of means excludes 0 (flag ``negligible`` when the whole
              interval also lies inside the equivalence margin);
* TIED        the interval lies inside the margin and includes 0;
* UNRESOLVED  otherwise.

The interval uses the pooled degrees of freedom and a Bonferroni level over the pairs; the margin is E2614's O1 tie
margin. The headline then follows the record: H9-a' (an interior share beats the boundary), H9-b' (no prose beats both
others) or H9-c' (unresolved). Per-register contrasts and the record's descriptive predictions are reported and are not
decision-bearing.

Usage:
    python scripts/analyse_arm_replication.py --arm 0.0=data/experiments/E2615-s00/study-5-seeds.json \\
        --arm 0.25=data/experiments/E2615-s25/study-5-seeds.json --arm 0.5=data/experiments/E2615-s50/study-5-seeds.json \\
        --noise-study experiments/E2614/results.json --out experiments/E2618/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.training.seed_study import aggregate_metric

PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
TECHNICAL = PARTITIONS[1:]

#: Fixed in experiments/E2618/record.md before any new seed ran; here only so they are visible and testable.
EQUIVALENCE_MARGIN = 0.0062
FAMILY_ALPHA = 0.05
MINIMUM_SEEDS = 3

__all__ = ["analyse", "main", "student_t_quantile"]


# ------------------------------------------------------------------ Student t, in the standard library


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """The regularised incomplete beta function I_x(a, b), by Lentz's continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _incomplete_beta(1.0 - x, b, a)
    tiny = 1e-300
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 400):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m))
        else:
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0))
        d = 1.0 + numerator * d
        d = tiny if abs(d) < tiny else d
        d = 1.0 / d
        c = 1.0 + numerator / c
        c = tiny if abs(c) < tiny else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-15:
            break
    return math.exp(log_front) * (f - 1.0) / a


def _t_cdf(t: float, df: float) -> float:
    tail = 0.5 * _incomplete_beta(df / (df + t * t), df / 2.0, 0.5)
    return 1.0 - tail if t >= 0 else tail


def student_t_quantile(p: float, df: float) -> float:
    """The ``p`` quantile of Student's t with ``df`` degrees of freedom, by bisection on the CDF.

    Checked against published table values in the tests; accurate to well under 1e-6 for the degrees of freedom used here.
    """
    if not 0.0 < p < 1.0 or df <= 0:
        message = f"need 0 < p < 1 and df > 0, got p={p}, df={df}"
        raise ValueError(message)
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -student_t_quantile(1.0 - p, df)
    low, high = 0.0, 1.0
    while _t_cdf(high, df) < p:
        high *= 2.0
        if high > 1e6:
            break
    for _ in range(200):
        middle = (low + high) / 2.0
        if _t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


# ------------------------------------------------------------------ the reading


def _bits(metrics: Mapping[str, Any], prefix: str) -> float:
    return float(metrics[f"partition_{prefix}_bits_per_character"])


def _series(runs: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, list[float]]:
    table: dict[str, list[float]] = {
        prefix: [_bits(runs[str(seed)]["metrics"], prefix) for seed in seeds]
        for prefix in PARTITIONS
    }
    table["whole_pack"] = [
        float(runs[str(seed)]["metrics"]["transformer_bits_per_character"]) for seed in seeds
    ]
    table["O1"] = [
        sum(table[prefix][index] for prefix in PARTITIONS) / len(PARTITIONS)
        for index in range(len(seeds))
    ]
    return table


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _sample_variance(values: Sequence[float]) -> float:
    centre = _mean(values)
    return sum((value - centre) ** 2 for value in values) / (len(values) - 1)


def _grade(difference: float, standard_error: float, quantile: float) -> dict[str, Any]:
    half = quantile * standard_error
    low, high = difference - half, difference + half
    excludes_zero = low > 0.0 or high < 0.0
    inside_margin = low >= -EQUIVALENCE_MARGIN and high <= EQUIVALENCE_MARGIN
    if excludes_zero:
        grade = "ORDERED"
    elif inside_margin:
        grade = "TIED"
    else:
        grade = "UNRESOLVED"
    return {
        "difference": difference,
        "interval": [low, high],
        "grade": grade,
        "negligible": bool(excludes_zero and inside_margin),
    }


def _contrasts(
    series: Mapping[str, Mapping[str, list[float]]],
    shares: Sequence[str],
    metric: str,
    *,
    pooled_variance: float,
    df: int,
    n: int,
    pair_count: int,
) -> list[dict[str, Any]]:
    quantile = student_t_quantile(1.0 - FAMILY_ALPHA / (2.0 * pair_count), df)
    standard_error = math.sqrt(pooled_variance * 2.0 / n)
    pairs = []
    for low, high in combinations(shares, 2):
        difference = _mean(series[low][metric]) - _mean(series[high][metric])
        graded = _grade(difference, standard_error, quantile)
        graded["pair"] = f"{low} vs {high}"
        graded["better"] = low if difference < 0 else high
        pairs.append(graded)
    return pairs


def analyse(
    arms: Mapping[str, Mapping[str, Any]], noise_study: Mapping[str, Any]
) -> dict[str, Any]:
    """The record's reading of the arms. ``arms`` maps a share label to a study aggregate."""
    shares = sorted(arms, key=float)
    if len(shares) < 2:
        message = "need at least two arms to compare"
        raise ValueError(message)
    common = sorted(set.intersection(*(set(map(int, arm["per_run"])) for arm in arms.values())))
    if len(common) < MINIMUM_SEEDS:
        message = (
            f"the arms share only {len(common)} completed seed(s) {common}; the record needs at least "
            f"{MINIMUM_SEEDS}"
        )
        raise ValueError(message)
    n = len(common)
    series = {share: _series(arms[share]["per_run"], common) for share in shares}
    pair_count = len(shares) * (len(shares) - 1) // 2
    df = len(shares) * (n - 1)

    def pooled(metric: str) -> float:
        return sum(_sample_variance(series[share][metric]) * (n - 1) for share in shares) / df

    primary_variance = pooled("O1")
    primary_pairs = _contrasts(
        series, shares, "O1", pooled_variance=primary_variance, df=df, n=n, pair_count=pair_count
    )
    means = {share: _mean(series[share]["O1"]) for share in shares}
    lowest = min(shares, key=lambda share: means[share])
    graded = {pair["pair"]: pair for pair in primary_pairs}

    def against(other: str) -> dict[str, Any]:
        key = f"{lowest} vs {other}"
        return graded[key] if key in graded else graded[f"{other} vs {lowest}"]

    def clearly_better(other: str) -> bool:
        pair = against(other)
        return pair["grade"] == "ORDERED" and not pair["negligible"] and pair["better"] == lowest

    others = [share for share in shares if share != lowest]
    boundary = shares[0]
    if lowest != boundary and clearly_better(boundary):
        outcome = "H9-a'"
    elif lowest == boundary and all(clearly_better(other) for other in others):
        outcome = "H9-b'"
    else:
        outcome = "H9-c'"

    noise_o1 = _series(noise_study["per_run"], sorted(map(int, noise_study["per_run"])))["O1"]
    noise_summary = aggregate_metric(noise_o1)
    pooled_sd = math.sqrt(primary_variance)
    secondary = {
        metric: {
            "means": {share: _mean(series[share][metric]) for share in shares},
            "pooled_sd": math.sqrt(pooled(metric)),
            "pairs": _contrasts(
                series,
                shares,
                metric,
                pooled_variance=pooled(metric),
                df=df,
                n=n,
                pair_count=pair_count,
            ),
        }
        for metric in (*PARTITIONS, "whole_pack")
    }
    return {
        "n": n,
        "seeds_used": common,
        "df": df,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "family_alpha": FAMILY_ALPHA,
        "t_quantile": student_t_quantile(1.0 - FAMILY_ALPHA / (2.0 * pair_count), df),
        "arms": {
            share: {
                "per_seed_O1": dict(zip(common, series[share]["O1"], strict=True)),
                "mean_O1": means[share],
                "sd_O1": math.sqrt(_sample_variance(series[share]["O1"])),
            }
            for share in shares
        },
        "primary": {
            "objective": "O1, the equal-weight mean of the five register partitions",
            "pairs": primary_pairs,
            "lowest_mean_arm": lowest,
            "outcome": outcome,
            "noise_check": {
                "pooled_sd": pooled_sd,
                "e2614_sd": noise_summary["sample_stdev"],
                "e2614_sd_interval": noise_summary["stdev_95_interval"],
                "exceeds_e2614_upper_bound": pooled_sd > noise_summary["stdev_95_interval"][1],
            },
        },
        "secondary_not_decision_bearing": secondary,
        "predictions": _predictions(series, shares, secondary, noise_summary),
    }


def _predictions(
    series: Mapping[str, Mapping[str, list[float]]],
    shares: Sequence[str],
    secondary: Mapping[str, Mapping[str, Any]],
    noise_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """The record's descriptive predictions P1'-P3', evaluated and labelled, never used to decide."""

    def ordered_in(metric: str, order: Sequence[str]) -> bool | None:
        """Whether ``metric`` falls along ``order`` with every adjacent pair graded ORDERED."""
        if any(share not in shares for share in order):
            return None
        pairs = {p["pair"]: p for p in secondary[metric]["pairs"]}
        means = secondary[metric]["means"]
        for left, right in pairwise(order):
            key = f"{min(left, right, key=float)} vs {max(left, right, key=float)}"
            if key not in pairs or pairs[key]["grade"] != "ORDERED":
                return False
            if not means[left] < means[right]:
                return False
        return True

    interval = noise_summary["stdev_95_interval"]
    spreads = {share: math.sqrt(_sample_variance(series[share]["O1"])) for share in shares}
    return {
        "P1_prose_lower_with_more_prose_and_ordered": ordered_in("prose-", ["0.5", "0.25", "0.0"]),
        "P2_technology_lower_with_less_prose_and_ordered": ordered_in(
            "technology-", ["0.0", "0.25", "0.5"]
        ),
        "P3_each_arm_sd_within_e2614_interval": {
            share: interval[0] <= spreads[share] <= interval[1] for share in shares
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--arm", action="append", required=True, metavar="SHARE=study.json")
    parser.add_argument("--noise-study", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        arms: dict[str, Any] = {}
        for item in args.arm:
            share, _, location = item.partition("=")
            float(share)
            arms[share] = json.loads(Path(location).read_text(encoding="utf-8"))
        result = analyse(arms, json.loads(args.noise_study.read_text(encoding="utf-8")))
    except (OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
