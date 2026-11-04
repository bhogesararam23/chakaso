"""Read a budget-extension pair: the same run for longer, and whether its validation curve ever turned.

Every arm this repository has trained at the reference configuration stopped at a declared step ceiling with its
validation loss still falling — E2630's five adopted-pre-tokenizer arms all record `best_step == max_steps` and
`completion: budget_ceiling`, so every absolute loss figure in that series is a floor and no validation minimum has ever
been observed for a trained arm. The question that follows is about the budget, not the tokenizer: **triple the ceiling
and does the curve turn?**

A pair is one seed at two budgets. The tool proves the two arms are the same run before it reads the difference: the
corpus, the split, the vocabulary, the architecture, the parameter count, the seed, the whole optimizer recipe, the
packed document pool and both evaluation packs must be identical, and `max_steps` is the one field allowed to differ, by
the ratio the record declared. Anything else is a protocol change and is refused by name. The reason that guard is here
rather than in `chakaso.training.arm_comparison` is that its `_MATCHED` table — which every other record in this series
relies on — says two arms with different update counts are not comparable. Here they are, and that difference *is* the
experiment, so the record that asks for it carries its own check instead of loosening everyone else's.

Two readings come out of one command:

* **the curve**, from the longer arm's own validation series — where its minimum falls, whether that is strictly inside
  the horizon, and how steep the last quarter still is. `budget.best_step` is cross-checked against the argmin this tool
  computes, because a verdict about convergence that rests on a field whose provenance is unverified is not a verdict.
* **the difference**, held-out bits per character of the short arm minus the long arm's, per seed and aggregated, read
  against **two bands neither of which this tool invents**: the adopted rule's own, derived here as 2.77 times the measured s from the
  committed record handed to `--bands` (E2614's declared derivation), and the inherited 0.02 that record's `margins`
  block still carries. The first is the band the new rule licenses; the second is kept beside it rather than retired,
  because every older verdict in the series was read against it.

Nothing here trains, scores or loads a checkpoint: it reads reports that already exist. Usage:

    python scripts/read_budget_extension.py \
        --short 7=data/experiments/E2636/seed-7-short/report.json --long 7=data/experiments/E2636/seed-7-long/report.json \
        --bands experiments/E2630/results.json \
        --outcomes configs/analyses/budget-extension-e2636-v1.json \
        --out experiments/E2636/results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.training.outcome_rules import decide, load_vocabulary
from chakaso.training.paired_stats import mean_confidence_interval, sign_test_p_value

__all__ = ["KNOWN_FACTS", "PARTITIONS", "read", "resolve_bands", "validation_curve_reading"]

#: The five register partitions O1 is the equal-weight mean of, in the order the reports name them.
PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")

#: E2614's declared derivation, quoted here as the number it is: a two-sided band for the difference between two
#: single draws at 95 % is 1.96*sqrt(2)*s, and every tie margin in this series is that product.
BAND_MULTIPLIER = 1.96 * math.sqrt(2.0)

#: E2614's replacement rule for a retired band ("2.77·s rounded up to the next 0.005"), kept so the derived band is
#: the same quantity the older records printed rather than a new convention.
BAND_ROUNDING = 0.005

#: Conditions both arms of a pair must share. A difference in any of these is a protocol change, not a budget.
_MATCHED: tuple[str, ...] = (
    ("corpus", "fingerprint"),
    ("corpus", "split_fingerprint"),
    ("corpus", "split_seed"),
    ("corpus", "training_tokens_packed"),
    ("corpus", "test_documents_packed"),
    ("tokenizer", "fingerprint"),
    ("tokenizer", "size"),
    ("tokenizer", "pre_tokenizer"),
    ("transformer", "parameter_count"),
    ("transformer", "architecture"),
    ("transformer", "budget", "seed"),
    ("transformer", "budget", "schedule"),
    ("transformer", "budget", "decay_to"),
    ("transformer", "budget", "warmup_steps"),
    ("transformer", "budget", "learning_rate"),
    ("transformer", "budget", "weight_decay"),
    ("transformer", "budget", "optimizer"),
    ("transformer", "budget", "effective_batch_size"),
    ("transformer", "budget", "eval_every"),
    ("transformer", "budget", "max_training_tokens"),
    ("transformer", "budget", "patience"),
    ("transformer", "budget", "gradient_clip_norm"),
    ("transformer", "test", "positions_scored"),
    ("transformer", "test", "characters_scored"),
    ("evaluation_sample",),
)

#: The facts a declared vocabulary may name. Kept in step with :func:`_facts` by a test, so a condition that typos a
#: fact name is refused at load rather than silently never firing. Comparisons against a band are computed *here*, from
#: the derived figure, so a vocabulary declares letters over booleans and never has to restate a threshold the record
#: already measures.
KNOWN_FACTS: frozenset[str] = frozenset(
    {
        "all_declared_pairs_present",
        "every_long_arm_still_falling",
        "every_long_arm_turned_up",
        "inherited_o1_band",
        "longest_tail_slope_per_100_updates",
        "mean_delta_clears_derived_band",
        "mean_delta_clears_inherited_band",
        "mean_delta_excludes_zero",
        "mean_delta_o1",
        "mean_delta_whole_pack",
        "o1_band",
        "pairs",
        "regression_beyond_band",
        "shortest_tail_slope_per_100_updates",
        "worst_seed_regresses_beyond_band",
    }
)


class BudgetReadingError(ValueError):
    """A report is missing a field, a pair is not matched, or a run did not reach the budget it declares."""


def _dig(report: Mapping[str, Any], path: Sequence[str], *, where: str) -> Any:
    current: Any = report
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            message = f"{where} has no {'.'.join(path)} in its report"
            raise BudgetReadingError(message)
        current = current[key]
    return current


def _pair_spec(value: str) -> tuple[str, Path]:
    """One ``seed=path`` command-line pair: the seed label the record gave the arm, and its report."""
    seed, separator, path = value.partition("=")
    if not separator or not seed or not path:
        message = f"--short/--long entry {value!r} must be SEED=report.json"
        raise BudgetReadingError(message)
    return seed, Path(path)


def _load(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        message = f"cannot read the report {str(path)!r}: {error}"
        raise BudgetReadingError(message) from error
    if not isinstance(document, dict):
        message = f"the report {str(path)!r} is not a JSON object"
        raise BudgetReadingError(message)
    return document


def _matched_problems(short: Mapping[str, Any], long_: Mapping[str, Any], seed: str) -> list[str]:
    """Every shared condition the two arms disagree on, by name, or none."""
    problems: list[str] = []
    for path in _MATCHED:
        left = _dig(short, path, where=f"seed {seed} short arm")
        right = _dig(long_, path, where=f"seed {seed} long arm")
        if left != right:
            problems.append(
                f"seed {seed}: {'.'.join(path)} is {left!r} in the short arm and {right!r} in the long arm"
            )
    return problems


def validation_curve_reading(curve: Sequence[Mapping[str, Any]], *, where: str) -> dict[str, Any]:
    """Where a validation series bottoms out, and how fast it is still moving at its last quarter.

    A curve with fewer than three points cannot show a minimum inside the horizon at all — one endpoint is not a turn —
    so it is refused rather than reported as "still falling". The slope is per 100 updates, in nats per token, over the
    final quarter of the series, because the question is whether the run was anywhere near a floor when it stopped.
    """
    if len(curve) < 3:
        message = f"{where}: a validation curve of {len(curve)} points cannot show a minimum inside its horizon"
        raise BudgetReadingError(message)
    steps = [int(point["step"]) for point in curve]
    values = [float(point["validation_loss"]) for point in curve]
    lowest = min(range(len(values)), key=lambda index: values[index])
    tail_start = max(0, len(values) - 1 - max(1, math.ceil(len(values) / 4)))
    span = steps[-1] - steps[tail_start]
    slope = (values[-1] - values[tail_start]) / (span / 100.0) if span else 0.0
    return {
        "points": len(curve),
        "steps": steps,
        "validation_min_step": steps[lowest],
        "validation_min_nats_per_token": values[lowest],
        "validation_final_nats_per_token": values[-1],
        "final_above_min_nats_per_token": values[-1] - values[lowest],
        "turned_up_inside_horizon": lowest < len(values) - 1,
        "tail_slope_nats_per_100_updates": round(slope, 8),
    }


def resolve_bands(source: Mapping[str, Any]) -> dict[str, Any]:
    """The adopted rule's own band, derived from a committed record's measured s, beside the inherited one.

    ``source`` is a committed aggregate carrying `treatment_spread` and `margins` — E2630 in this series. The derived
    figure is 2.77 times the measured s of O1 under the adopted rule, rounded up to the next 0.005 exactly as E2614
    declared a retired band is replaced; the inherited figure is the 0.02 that record still carries. Nothing here
    retires the older band: E2614's prose and whole-pack verdicts were read against it, and a record that replaces a
    threshold silently makes those verdicts unreadable.
    """
    spread = source.get("treatment_spread")
    margins = source.get("margins")
    if not isinstance(spread, Mapping) or "o1" not in spread:
        message = "the bands source has no treatment_spread.o1, so the adopted rule's seed spread is not in committed evidence"
        raise BudgetReadingError(message)
    if not isinstance(margins, Mapping) or "o1_separable" not in margins:
        message = "the bands source has no margins.o1_separable, so there is no inherited band to compare against"
        raise BudgetReadingError(message)
    o1_stdev = float(spread["o1"]["sample_stdev"])
    raw = BAND_MULTIPLIER * o1_stdev
    rounded = math.ceil(raw / BAND_ROUNDING) * BAND_ROUNDING
    return {
        "band_source_sigma_o1": o1_stdev,
        "band_multiplier": round(BAND_MULTIPLIER, 6),
        "derived_band_unrounded": round(raw, 6),
        "o1_band": round(rounded, 6),
        "o1_band_rule": "1.96*sqrt(2)*s of the adopted rule's own seed spread, rounded up to the next 0.005 (E2614's rule)",
        "inherited_o1_band": float(margins["o1_separable"]),
        "inherited_o1_tie": float(margins.get("o1_tie", float("nan"))),
        "whole_pack_note": (
            "the whole pack has no measured seed spread under the adopted rule in the bands source, so its band is the inherited "
            "0.00933 and is labelled as such"
        ),
        "relationship": (
            f"the derived band is {rounded / float(margins['o1_separable']):.3f} of the inherited 0.02, so a "
            "difference the old band called indeterminate can separate under the new one; both are printed and "
            "the verdict names which one it was read on"
        ),
    }


def _per_seed(
    short: Mapping[str, Any], long_: Mapping[str, Any], bands: Mapping[str, Any]
) -> dict[str, Any]:
    o1_band = float(bands["o1_band"])
    short_o1 = _bits(short)
    long_o1 = _bits(long_)
    difference = short_o1 - long_o1
    return {
        "o1_short": short_o1,
        "o1_long": long_o1,
        "delta_o1": difference,
        "whole_pack_short": float(
            _dig(short, ("transformer", "test", "bits_per_character"), where="short")
        ),
        "whole_pack_long": float(
            _dig(long_, ("transformer", "test", "bits_per_character"), where="long")
        ),
        "delta_whole_pack": float(
            _dig(short, ("transformer", "test", "bits_per_character"), where="short")
        )
        - float(_dig(long_, ("transformer", "test", "bits_per_character"), where="long")),
        "clears_derived_band": abs(difference) >= o1_band,
        "clears_inherited_band": abs(difference) >= float(bands["inherited_o1_band"]),
        "partitions": {
            name: {
                "short": _partition_bits(short, name),
                "long": _partition_bits(long_, name),
                "delta": _partition_bits(short, name) - _partition_bits(long_, name),
            }
            for name in PARTITIONS
        },
    }


def _bits(report: Mapping[str, Any]) -> float:
    """O1 — the equal-weight mean of the five register partitions, not a pooled figure (K-027)."""
    return sum(_partition_bits(report, name) for name in PARTITIONS) / len(PARTITIONS)


def _partition_bits(report: Mapping[str, Any], name: str) -> float:
    return float(
        _dig(
            report,
            ("transformer", "test_partitions", f"{name}", "metrics", "bits_per_character"),
            where=f"{name} partition",
        )
    )


def _facts(
    per_seed: Sequence[Mapping[str, Any]],
    curves: Mapping[str, Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    bands: Mapping[str, Any],
    expected_pairs: int,
) -> dict[str, Any]:
    turned = [bool(curve["turned_up_inside_horizon"]) for curve in curves.values()]
    slopes = [float(curve["tail_slope_nats_per_100_updates"]) for curve in curves.values()]
    o1_band = float(bands["o1_band"])
    mean_delta = float(aggregate["o1"]["mean"])
    return {
        "pairs": len(per_seed),
        "all_declared_pairs_present": len(per_seed) >= expected_pairs,
        "mean_delta_o1": mean_delta,
        "mean_delta_whole_pack": float(aggregate["whole_pack"]["mean"]),
        "mean_delta_excludes_zero": bool(aggregate["o1"]["excludes_zero"]),
        "mean_delta_clears_derived_band": mean_delta >= o1_band,
        "mean_delta_clears_inherited_band": mean_delta >= float(bands["inherited_o1_band"]),
        "worst_seed_regresses_beyond_band": any(
            float(row["delta_o1"]) <= -o1_band for row in per_seed
        ),
        "regression_beyond_band": mean_delta <= -o1_band,
        "every_long_arm_turned_up": all(turned) if turned else False,
        "every_long_arm_still_falling": not any(turned) if turned else False,
        "longest_tail_slope_per_100_updates": max(slopes) if slopes else 0.0,
        "shortest_tail_slope_per_100_updates": min(slopes) if slopes else 0.0,
        "o1_band": o1_band,
        "inherited_o1_band": float(bands["inherited_o1_band"]),
    }


def read(
    shorts: Mapping[str, Path],
    longs: Mapping[str, Path],
    *,
    bands_source: Mapping[str, Any],
    expected_ratio: float = 3.0,
    expected_pairs: int | None = None,
    vocabulary_path: Path | None = None,
    statistics_only: bool = False,
) -> dict[str, Any]:
    """Compare each declared budget pair and report the curve, the difference and the letter the record declared.

    ``shorts`` and ``longs`` map a seed label to that arm's report path; both must carry the same keys. The pair is
    refused unless it is matched on every shared condition, unless the long arm ran exactly ``expected_ratio`` times the
    short arm's updates, and unless each arm finished at its own ceiling — a run that early-stopped is not a longer
    budget. A vocabulary may be handed in by path or the caller may ask for the statistics alone; a verdict is printed
    in no other case (ADR-0052).

    Raises:
        BudgetReadingError: any of the refusals above, or a report missing a field the reading needs.
    """
    if sorted(shorts) != sorted(longs):
        message = f"the short arms cover {sorted(shorts)} and the long arms cover {sorted(longs)}"
        raise BudgetReadingError(message)
    if not shorts:
        message = "no pairs were handed to the reader"
        raise BudgetReadingError(message)
    bands = resolve_bands(bands_source)
    pairs: list[dict[str, Any]] = []
    curves: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for seed in sorted(shorts, key=lambda value: (len(value), value)):
        short, long_ = _load(shorts[seed]), _load(longs[seed])
        problems.extend(_matched_problems(short, long_, seed))
        short_steps = int(
            _dig(short, ("transformer", "budget", "max_steps"), where=f"seed {seed} short")
        )
        long_steps = int(
            _dig(long_, ("transformer", "budget", "max_steps"), where=f"seed {seed} long")
        )
        if long_steps <= short_steps:
            problems.append(
                f"seed {seed}: the long arm ran {long_steps} updates against the short arm's {short_steps}"
            )
        elif short_steps <= 0 or abs(long_steps / short_steps - expected_ratio) > 1e-9:
            problems.append(
                f"seed {seed}: {long_steps} against {short_steps} updates is not the declared "
                f"{expected_ratio:g} times the short arm's budget"
            )
        for tag, report in (("short", short), ("long", long_)):
            steps = int(
                _dig(report, ("transformer", "budget", "max_steps"), where=f"seed {seed} {tag}")
            )
            completion = _dig(
                report, ("transformer", "budget", "completion"), where=f"seed {seed} {tag}"
            )
            if completion != "budget_ceiling":
                problems.append(
                    f"seed {seed} {tag} arm finished by {completion!r}, not at its ceiling"
                )
            last = int(
                _dig(report, ("transformer", "curve"), where=f"seed {seed} {tag}")[-1]["step"]
            )
            if last != steps:
                problems.append(
                    f"seed {seed} {tag} curve ends at step {last}, not at its {steps}-update ceiling"
                )
        curve = validation_curve_reading(
            _dig(long_, ("transformer", "curve"), where=f"seed {seed} long"),
            where=f"seed {seed} long arm",
        )
        reported_best = _dig(
            long_, ("transformer", "budget", "best_step"), where=f"seed {seed} long"
        )
        if int(reported_best) != int(curve["validation_min_step"]):
            problems.append(
                f"seed {seed}: the report's best_step is {reported_best} and the validation argmin is at "
                f"{curve['validation_min_step']}"
            )
        curves[seed] = curve
        pairs.append({"seed": seed, **_per_seed(short, long_, bands)})
    if problems:
        message = "these arms are not a matched budget comparison:\n  " + "\n  ".join(problems)
        raise BudgetReadingError(message)

    if len(pairs) < 2:
        message = (
            "a budget extension is read as a mean over paired draws, and one draw has no spread to average: "
            f"the record declared {len(pairs)} completed pair(s). Report the single difference as a shortfall "
            "with the arm that did not complete named, which is what E2630 did with an incomplete set."
        )
        raise BudgetReadingError(message)
    deltas_o1 = [float(row["delta_o1"]) for row in pairs]
    deltas_pack = [float(row["delta_whole_pack"]) for row in pairs]
    mean, low, high, degrees, critical = mean_confidence_interval(deltas_o1)
    pack_mean, pack_low, pack_high, pack_df, pack_critical = mean_confidence_interval(deltas_pack)
    aggregate = {
        "unit": "bits per character, short arm minus long arm (positive means the longer budget is lower)",
        "o1": {
            "values": deltas_o1,
            "mean": round(mean, 6),
            "degrees_of_freedom": degrees,
            "t_critical": round(critical, 6),
            "interval_95": [round(low, 6), round(high, 6)],
            "excludes_zero": low > 0.0 or high < 0.0,
            "positive_draws": sum(1 for value in deltas_o1 if value > 0.0),
            "sign_test_p_value": sign_test_p_value(
                successes=sum(1 for value in deltas_o1 if value > 0.0), total=len(deltas_o1)
            ),
        },
        "whole_pack": {
            "values": [round(value, 6) for value in deltas_pack],
            "mean": round(pack_mean, 6),
            "degrees_of_freedom": pack_df,
            "t_critical": round(pack_critical, 6),
            "interval_95": [round(pack_low, 6), round(pack_high, 6)],
            "excludes_zero": pack_low > 0.0 or pack_high < 0.0,
        },
    }
    facts = _facts(pairs, curves, aggregate, bands, expected_pairs or len(pairs))
    outcome: dict[str, Any] | None = None
    if vocabulary_path is not None and not statistics_only:
        outcome = decide(facts, load_vocabulary(vocabulary_path, known_facts=sorted(KNOWN_FACTS)))
    return {
        "schema_version": "chakaso-budget-extension-reading/1",
        "design": {
            "unit": "one seed at two budgets, both arms on one host",
            "expected_ratio": expected_ratio,
            "pairs": len(pairs),
            "seeds": sorted((row["seed"] for row in pairs), key=lambda value: (len(value), value)),
            "primary_metric": "O1, the equal-weight mean of the five register partitions",
            "bands": bands,
        },
        "per_seed": pairs,
        "validation_curves": {seed: curves[seed] for seed in sorted(curves)},
        "aggregate": aggregate,
        "facts": facts,
        "outcome": outcome,
        "statistics_only": statistics_only or vocabulary_path is None,
        "claim_limit": (
            "a longer budget that lowers held-out loss is a statement about this configuration's curve at this "
            "corpus. It is not a capability claim and it is not evidence that any model here produces useful "
            "language, which is K-001 and is unmoved by every number in this artifact."
        ),
        "inputs": {
            "short": {seed: _describe(shorts[seed]) for seed in sorted(shorts)},
            "long": {seed: _describe(longs[seed]) for seed in sorted(longs)},
        },
    }


def _describe(path: Path) -> dict[str, Any]:
    return {
        "path": str(path).replace("\\", "/"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read a budget-extension pair from two run reports."
    )
    parser.add_argument(
        "--short", action="append", default=[], help="SEED=report.json, one per pair"
    )
    parser.add_argument(
        "--long", action="append", default=[], help="SEED=report.json, one per pair"
    )
    parser.add_argument(
        "--bands",
        type=Path,
        required=True,
        help="a committed aggregate whose treatment_spread gives the adopted rule's seed spread (E2630/results.json)",
    )
    parser.add_argument("--expected-ratio", type=float, default=3.0)
    parser.add_argument("--expected-pairs", type=int, default=0)
    parser.add_argument("--outcomes", type=Path, help="the record's declared outcome vocabulary")
    parser.add_argument(
        "--statistics-only", action="store_true", help="print the numbers and no letter (ADR-0052)"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(
            f"error: {args.out} already exists; a measured reading is not replaced unasked",
            file=sys.stderr,
        )
        return 2
    try:
        shorts = dict(_pair_spec(item) for item in args.short)
        longs = dict(_pair_spec(item) for item in args.long)
        report = read(
            shorts,
            longs,
            bands_source=json.loads(args.bands.read_text(encoding="utf-8")),
            expected_ratio=args.expected_ratio,
            expected_pairs=args.expected_pairs or None,
            vocabulary_path=args.outcomes,
            statistics_only=args.statistics_only,
        )
    except (BudgetReadingError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict = report["outcome"]["letter"] if report["outcome"] else "no letter (statistics only)"
    print(
        f"{report['design']['pairs']} pair(s), mean Δ(O1) {report['aggregate']['o1']['mean']:+g} against band "
        f"{report['design']['bands']['o1_band']:g} (inherited {report['design']['bands']['inherited_o1_band']:g}): "
        f"{verdict}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
