"""Read E2617's cross-machine repeat of E2614 seed 7 by the rules its record declared before the run.

The record fixes the reference (E2614's committed per-run summary for seed 7), the reporting tolerance (0.0005, E2606's
``REPRODUCTION_TOLERANCES``), the tie margins (2.77 times the sample standard deviation E2614 measured per metric, with
the mathematics band of 0.035 replacing the central margin) and three outcomes:

* X-a: |difference| <= 0.0005 on every metric         reproduced at the project's reporting resolution;
* X-b: some |difference| > 0.0005, all within margin  a machine effect smaller than one seed's noise;
* X-c: any |difference| over its tie margin           a machine effect beyond seed noise, on the metrics named.

It first refuses a run that is not a repeat (another corpus, split, tokenizer, architecture or evaluation pack), and it
reports both runs' environment blocks side by side because the machine is a bundle (operating system, torch, numpy,
Python, threads) that cannot be taken apart.

Usage:
    python scripts/read_cross_machine_repeat.py --report data/experiments/E2617/seed-7/report.json \\
        --reference experiments/E2614/results.json --out experiments/E2617/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.training.seed_study import aggregate_metric
from chakaso.training.share_sweep import Margins, margins_from_stdev

PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")

#: Fixed in experiments/E2617/record.md before the run; here only so they are visible and testable.
TOLERANCE = 0.0005
MATHEMATICS_BAND = 0.035
REFERENCE_SEED = "7"
_EPSILON = 1e-12


def _reference_value(metrics: Mapping[str, Any], name: str) -> float:
    if name == "whole_pack":
        return float(metrics["transformer_bits_per_character"])
    return float(metrics[f"partition_{name}_bits_per_character"])


def _new_values(report: Mapping[str, Any]) -> dict[str, float]:
    transformer = report["transformer"]
    values = {
        prefix: float(transformer["test_partitions"][prefix]["metrics"]["bits_per_character"])
        for prefix in PARTITIONS
    }
    values["whole_pack"] = float(transformer["test"]["bits_per_character"])
    return values


def _with_mean(values: dict[str, float]) -> dict[str, float]:
    return {**values, "O1": sum(values[prefix] for prefix in PARTITIONS) / len(PARTITIONS)}


def _margins(reference: Mapping[str, Any]) -> dict[str, Margins]:
    """Tie margins per metric from the five seeds E2614 measured, keeping the mathematics band it set."""
    runs = [run["metrics"] for run in reference["per_run"].values()]
    series: dict[str, list[float]] = {
        name: [_reference_value(metrics, name) for metrics in runs]
        for name in (*PARTITIONS, "whole_pack")
    }
    series["O1"] = [
        sum(series[prefix][index] for prefix in PARTITIONS) / len(PARTITIONS)
        for index in range(len(runs))
    ]
    table: dict[str, Margins] = {}
    for name, values in series.items():
        summary = aggregate_metric(values)
        replaces = name == "mathematics-"
        table[name] = margins_from_stdev(
            summary["sample_stdev"],
            summary["stdev_95_interval"][1],
            declared_band=MATHEMATICS_BAND if replaces else None,
            band_replaces_central=replaces,
        )
    return table


def _identity_problems(report: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    protocol = reference["protocol"]
    checks = {
        "corpus fingerprint": (report["corpus"]["fingerprint"], protocol["corpus_fingerprint"]),
        "split fingerprint": (
            report["corpus"]["split_fingerprint"],
            protocol["split_fingerprint"],
        ),
        "tokenizer fingerprint": (
            report["tokenizer"]["fingerprint"],
            protocol["tokenizer_fingerprint"],
        ),
        "parameter count": (report["transformer"]["parameter_count"], protocol["parameter_count"]),
        "positions scored": (
            report["transformer"]["test"]["positions_scored"],
            protocol["test_positions"],
        ),
        "characters scored": (
            report["transformer"]["test"]["characters_scored"],
            protocol["test_characters"],
        ),
    }
    return [
        f"{label}: run {observed!r}, reference {expected!r}"
        for label, (observed, expected) in checks.items()
        if observed != expected
    ]


def read(report: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """The record's reading of one repeat against E2614's seed-7 summary."""
    run = reference["per_run"][REFERENCE_SEED]
    reference_values = _with_mean(
        {name: _reference_value(run["metrics"], name) for name in (*PARTITIONS, "whole_pack")}
    )
    new_values = _with_mean(_new_values(report))
    margins = _margins(reference)
    metrics: dict[str, dict[str, float | bool]] = {}
    beyond: list[str] = []
    beyond_tolerance: list[str] = []
    for name, new in new_values.items():
        delta = new - reference_values[name]
        tie = margins[name].central
        metrics[name] = {
            "reference": reference_values[name],
            "new": new,
            "delta": delta,
            "tie_margin": tie,
            "within_tolerance": abs(delta) <= TOLERANCE + _EPSILON,
            "within_tie_margin": abs(delta) <= tie + _EPSILON,
        }
        if abs(delta) > TOLERANCE + _EPSILON:
            beyond_tolerance.append(name)
        if abs(delta) > tie + _EPSILON:
            beyond.append(name)
    if beyond:
        outcome = "X-c"
    elif beyond_tolerance:
        outcome = "X-b"
    else:
        outcome = "X-a"
    return {
        "outcome": outcome,
        "tolerance": TOLERANCE,
        "beyond_tolerance": beyond_tolerance,
        "beyond_tie_margin": beyond,
        "metrics": metrics,
        "environment": {
            "reference": run.get("environment", {}),
            "new": report.get("environment", {}),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        problems = _identity_problems(report, reference)
        if problems:
            print("error: this run is not a repeat of the reference:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 2
        result = read(report, reference)
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
