"""Read a set of space-prefixed tokenizer arms against their same-host controls, as one paired spread.

E2620, E2624, E2625 and E2628 each read **one** treatment against **one** control and printed one difference. E2630
runs the same design at several seeds, which turns the question from "is this difference outside the band" into "what
is the mean of these differences and how well is it known" — a question a single pair cannot answer and a normal
approximation should not either. At four draws the honest two-sided 95 % critical value is 3.182 rather than 1.96, and
whether the interval clears zero is exactly what the record's adoption gate turns on.

Every draw is checked by the guard the single-pair reader uses: a treatment whose updates or training documents differ
from its control's is refused, as is a control whose numbers disagree with the study aggregate it came from. The seeds
are paired by seed, so no draw is compared against the wrong initialisation.

Usage:
    python scripts/analyse_tokenizer_seed_spread.py --spread experiments/E2614/results.json \\
        --treatment 11=experiments/E2628/E2628-report.json \\
        --control 11=experiments/E2628/E2614-seed-11-control-report.json \\
        --out experiments/E2630/results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from read_tokenizer_arm import (
    O1_SEPARABLE,
    O1_TIE,
    PARTITIONS,
    WHOLE_TIE,
    matchedness_problems,
    o1,
    whole_pack,
)

from chakaso.training.paired_stats import mean_confidence_interval, sign_test_p_value

__all__ = ["AnalysisError", "classify", "main", "read", "summarise"]

#: A control extract and the study aggregate it came from are the same measurement of the same run, reported to six
#: decimals in one file and in full in the other. This is the width of that rounding, not a tolerance for disagreement.
_AGGREGATE_TOLERANCE = 5e-7


class AnalysisError(RuntimeError):
    """The reports handed to the analysis do not describe one paired spread."""


def _load(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        message = f"{path} is a JSON {type(loaded).__name__}, expected an object"
        raise AnalysisError(message)
    return loaded


def _entries(values: Sequence[str], *, option: str) -> dict[int, Path]:
    """Parse repeatable ``SEED=path`` arguments, refusing a seed named twice and a seed that is not a number."""
    entries: dict[int, Path] = {}
    for value in values:
        seed_text, _, raw_path = value.partition("=")
        if not raw_path:
            message = f"--{option} wants SEED=path, got {value!r}"
            raise AnalysisError(message)
        try:
            seed = int(seed_text)
        except ValueError as error:
            message = f"--{option} {value!r}: {seed_text!r} is not a seed number"
            raise AnalysisError(message) from error
        if seed in entries:
            message = (
                f"--{option} names seed {seed} twice; two reports for one draw is not a spread"
            )
            raise AnalysisError(message)
        entries[seed] = Path(raw_path)
    return entries


def _check_aggregate(control: Mapping[str, Any], recorded: Mapping[str, Any], *, seed: int) -> None:
    """Refuse a control extract that does not carry the numbers the study aggregate says the run produced."""
    problems: list[str] = []
    checks: list[tuple[str, float, Any]] = [
        ("whole pack", whole_pack(control), recorded.get("transformer_bits_per_character"))
    ]
    checks += [
        (
            prefix,
            float(
                control["transformer"]["test_partitions"][prefix]["metrics"]["bits_per_character"]
            ),
            recorded.get(f"partition_{prefix}_bits_per_character"),
        )
        for prefix in PARTITIONS
    ]
    for label, measured, remembered in checks:
        if remembered is None:
            problems.append(f"seed {seed}: the aggregate records no {label} value")
        elif abs(float(remembered) - measured) > _AGGREGATE_TOLERANCE:
            problems.append(
                f"seed {seed}, {label}: extract {measured!r}, aggregate {float(remembered)!r}"
            )
    if problems:
        message = (
            "the control extract disagrees with E2614's committed aggregate:\n  "
            + "\n  ".join(problems)
        )
        raise AnalysisError(message)


def _interval(values: Sequence[float]) -> dict[str, Any]:
    mean, low, high, degrees, critical = mean_confidence_interval(values)
    return {
        "values": list(values),
        "mean": mean,
        "interval_95": [low, high],
        "degrees_of_freedom": degrees,
        "t_critical": critical,
        "excludes_zero": low > 0.0 or high < 0.0,
        "positive_draws": sum(1 for value in values if value > 0.0),
        "sign_test_p_value": sign_test_p_value(
            sum(1 for value in values if value > 0.0), total=len(values)
        ),
    }


def classify(
    deltas_o1: Sequence[float],
    o1_interval: Mapping[str, Any],
    whole_interval: Mapping[str, Any],
    *,
    adoption_seeds: int,
) -> str:
    """E2630's Y2 outcomes, from the rules its record fixed before any arm ran."""
    if (
        any(delta < -O1_TIE for delta in deltas_o1)
        or abs(float(o1_interval["mean"])) <= O1_SEPARABLE
    ):
        return "Y2-c"
    if all(delta > 0.0 for delta in deltas_o1) and float(o1_interval["mean"]) > O1_SEPARABLE:
        cleared = bool(o1_interval["excludes_zero"]) and bool(whole_interval["excludes_zero"])
        material = float(whole_interval["mean"]) > WHOLE_TIE
        if len(deltas_o1) >= adoption_seeds and cleared and material:
            return "Y2-a"
        return "Y2-b"
    return "Y2-d"


def read(
    draws: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    *,
    spread: Mapping[str, Any],
    adoption_seeds: int = 4,
) -> dict[str, Any]:
    """The paired spread over ``(seed, treatment report, control report)`` draws, best-draw-ordered input accepted."""
    if len(draws) < 2:
        message = f"a spread needs at least two draws, got {len(draws)}; one difference is E2628, not a spread"
        raise AnalysisError(message)

    per_run = spread.get("per_run")
    if not isinstance(per_run, Mapping):
        message = (
            "the control aggregate carries no per_run block, so no control can be cross-checked"
        )
        raise AnalysisError(message)

    records: list[dict[str, Any]] = []
    for seed, treatment, control in sorted(draws, key=lambda draw: draw[0]):
        problems = matchedness_problems(treatment, control, same_documents=True)
        if problems:
            named = [f"seed {seed}: {problem}" for problem in problems]
            message = (
                "this is not a tokenizer-only draw, so it cannot join the spread:\n  "
                + "\n  ".join(named)
            )
            raise AnalysisError(message)
        remembered = per_run.get(str(seed))
        if not isinstance(remembered, Mapping):
            message = f"seed {seed} is not in the control aggregate (its per_run keys are {sorted(per_run)})"
            raise AnalysisError(message)
        _check_aggregate(control, remembered["metrics"], seed=seed)
        records.append(
            {
                "seed": seed,
                "delta_o1": o1(control) - o1(treatment),
                "delta_whole_pack": whole_pack(control) - whole_pack(treatment),
                "o1": {"control": o1(control), "treatment": o1(treatment)},
                "whole_pack": {"control": whole_pack(control), "treatment": whole_pack(treatment)},
                "partitions": {
                    prefix: {
                        "control": control["transformer"]["test_partitions"][prefix]["metrics"][
                            "bits_per_character"
                        ],
                        "treatment": treatment["transformer"]["test_partitions"][prefix]["metrics"][
                            "bits_per_character"
                        ],
                        "delta": control["transformer"]["test_partitions"][prefix]["metrics"][
                            "bits_per_character"
                        ]
                        - treatment["transformer"]["test_partitions"][prefix]["metrics"][
                            "bits_per_character"
                        ],
                    }
                    for prefix in PARTITIONS
                },
                "characters_per_identifier": {
                    "control": control["tokenizer"].get("characters_per_token_test"),
                    "treatment": treatment["tokenizer"].get("characters_per_token_test"),
                },
                "wall_clock_s": treatment["transformer"]["budget"].get("wall_clock_s"),
            }
        )

    deltas_o1 = [float(record["delta_o1"]) for record in records]
    deltas_whole = [float(record["delta_whole_pack"]) for record in records]
    o1_interval = _interval(deltas_o1)
    whole_interval = _interval(deltas_whole)

    def spread_of(side: str, metric: str) -> dict[str, Any]:
        values = [float(record[metric][side]) for record in records]
        return {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
            "range": max(values) - min(values),
        }

    return {
        "design": {
            "mode": "equal-updates-and-documents",
            "seeds": [record["seed"] for record in records],
            "draws": len(records),
            "adoption_seeds": adoption_seeds,
            "unit": "bits per character, control minus treatment (positive favours the treatment)",
        },
        "margins": {"o1_tie": O1_TIE, "o1_separable": O1_SEPARABLE, "whole_tie": WHOLE_TIE},
        "draw": records,
        "o1": o1_interval,
        "whole_pack": whole_interval,
        "treatment_spread": {
            "o1": spread_of("treatment", "o1"),
            **_partition_spread(records, "treatment"),
        },
        "control_spread": {
            "o1": spread_of("control", "o1"),
            **_partition_spread(records, "control"),
        },
        "outcome": classify(
            deltas_o1,
            o1_interval,
            whole_interval,
            adoption_seeds=adoption_seeds,
        ),
    }


def _partition_spread(records: Sequence[Mapping[str, Any]], side: str) -> dict[str, dict[str, Any]]:
    """The same three numbers for each register, so a reader can see whether one partition carries the mean."""
    spread: dict[str, dict[str, Any]] = {}
    for prefix in PARTITIONS:
        values = [float(record["partitions"][prefix][side]) for record in records]
        spread[prefix] = {
            "values": values,
            "mean": statistics.fmean(values),
            "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
        }
    return spread


def summarise(result: Mapping[str, Any]) -> str:
    """A short text rendering, so the command that ran is also readable on a terminal."""
    o1_stats = result["o1"]
    whole_stats = result["whole_pack"]
    lines = [
        f"outcome {result['outcome']} over {result['design']['draws']} same-host draws "
        f"(seeds {', '.join(str(seed) for seed in result['design']['seeds'])})",
        (
            f"  O1: mean {float(o1_stats['mean']):+.6f} "
            f"[{float(o1_stats['interval_95'][0]):+.6f}, {float(o1_stats['interval_95'][1]):+.6f}] "
            f"t={float(o1_stats['t_critical']):.3f} on {o1_stats['degrees_of_freedom']} df, "
            f"{'clears' if o1_stats['excludes_zero'] else 'does not clear'} zero; "
            f"{o1_stats['positive_draws']}/{len(o1_stats['values'])} positive, "
            f"sign test p={float(o1_stats['sign_test_p_value']):.5f}"
        ),
        (
            f"  whole pack: mean {float(whole_stats['mean']):+.6f} "
            f"[{float(whole_stats['interval_95'][0]):+.6f}, {float(whole_stats['interval_95'][1]):+.6f}], "
            f"{'clears' if whole_stats['excludes_zero'] else 'does not clear'} zero"
        ),
    ]
    for record in result["draw"]:
        lines.append(
            f"  seed {record['seed']}: O1 {record['o1']['control']:.6f} -> {record['o1']['treatment']:.6f} "
            f"({float(record['delta_o1']):+.6f}), pack {float(record['delta_whole_pack']):+.6f}"
        )
    treatment = result["treatment_spread"]["o1"]
    control = result["control_spread"]["o1"]
    lines.append(
        f"  treatment O1 stdev {treatment['sample_stdev']:.6f} against the control's "
        f"{control['sample_stdev']:.6f}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--treatment",
        action="append",
        required=True,
        metavar="SEED=path",
        help="one space-prefixed run report extract, keyed by the seed it trained",
    )
    parser.add_argument(
        "--control",
        action="append",
        required=True,
        metavar="SEED=path",
        help="the default-tokenizer run report extract for the same seed on the same host",
    )
    parser.add_argument(
        "--spread",
        type=Path,
        required=True,
        help="the control study aggregate (experiments/E2614/results.json), used to cross-check every control extract",
    )
    parser.add_argument(
        "--adoption-seeds",
        type=int,
        default=4,
        help="the number of same-host draws the record fixed as its adoption gate (E2630: 4)",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        treatments = _entries(args.treatment, option="treatment")
        controls = _entries(args.control, option="control")
        if set(treatments) != set(controls):
            message = (
                "every treatment needs its own control: treatments "
                f"{sorted(treatments)}, controls {sorted(controls)}"
            )
            raise AnalysisError(message)
        draws = [
            (seed, _load(treatments[seed]), _load(controls[seed])) for seed in sorted(treatments)
        ]
        result = read(draws, spread=_load(args.spread), adoption_seeds=args.adoption_seeds)
    except (AnalysisError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(summarise(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
