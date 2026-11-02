"""Read E2619's passes-versus-updates arms by the rules its record declared before any arm ran.

Three prose-only arms make the same 2,400 updates and draw them from nested pools of 5M, 10M and 20M identifiers
(about four, two and one passes). The primary metric is the held-out prose bits per character of each. With the tie
margin 0.00625 and the separable margin 0.02 (E2614):

* Q-b: the three values lie within the tie margin of one another          repetition is free in this range;
* Q-a: the 5M arm is the worst and is worse than the 20M arm by more than the tie margin   repetition costs prose;
* Q-c: anything else                                                       reported, no mechanism claimed.

The tool first refuses arms that are not a pool-size comparison (different update counts, corpus, split, tokenizer or
architecture, not prose-only, or a cap that is not the one the arm's name declares). It reports the other partitions
and the equal-weight mean, the realised passes, and the E2613-H and E2612-P values (the 10M and 20M configurations from
the other host) beside this host's.

Usage:
    python scripts/read_passes_at_fixed_updates.py --arm 5m=data/experiments/E2619-prose-5m/seed-7/report.json \\
        --arm 10m=... --arm 20m=... --references experiments/E2613/results.json --out experiments/E2619/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")

#: Fixed in experiments/E2619/record.md before any arm ran; here only so they are visible and testable.
TIE_MARGIN = 0.00625
SEPARABLE_MARGIN = 0.02
CAPS = {"5m": 5_000_000, "10m": 10_000_000, "20m": 20_000_000}
EXPECTED_UPDATES = 2400
_EPSILON = 1e-12

#: The earlier-host arm that is the same configuration as each of ours.
_EARLIER_HOST = {"10m": "H", "20m": "P"}


def _grade(difference: float) -> str:
    size = abs(difference)
    if size <= TIE_MARGIN + _EPSILON:
        return "tie"
    if size <= SEPARABLE_MARGIN + _EPSILON:
        return "indeterminate"
    return "separable"


def _identity_problems(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    problems: list[str] = []
    first = next(iter(reports.values()))
    for tag, report in reports.items():
        budget = report["transformer"]["budget"]
        if budget["max_steps"] != EXPECTED_UPDATES:
            problems.append(f"{tag}: {budget['max_steps']} updates, expected {EXPECTED_UPDATES}")
        if budget.get("training_group_prefixes") != ["prose-"]:
            problems.append(f"{tag}: not prose-only ({budget.get('training_group_prefixes')!r})")
        if budget.get("max_training_tokens") != CAPS[tag]:
            problems.append(
                f"{tag}: cap {budget.get('max_training_tokens')!r}, expected {CAPS[tag]}"
            )
        for label, observed, expected in (
            ("corpus fingerprint", report["corpus"]["fingerprint"], first["corpus"]["fingerprint"]),
            (
                "split fingerprint",
                report["corpus"]["split_fingerprint"],
                first["corpus"]["split_fingerprint"],
            ),
            (
                "tokenizer fingerprint",
                report["tokenizer"]["fingerprint"],
                first["tokenizer"]["fingerprint"],
            ),
            (
                "parameter count",
                report["transformer"]["parameter_count"],
                first["transformer"]["parameter_count"],
            ),
            ("batch size", budget["batch_size"], first["transformer"]["budget"]["batch_size"]),
        ):
            if observed != expected:
                problems.append(
                    f"{tag}: {label} {observed!r} differs from the first arm's {expected!r}"
                )
    return problems


def read(reports: Mapping[str, Mapping[str, Any]], references: Mapping[str, Any]) -> dict[str, Any]:
    """The record's reading of the three arms."""
    bits = {
        tag: {
            prefix: float(
                report["transformer"]["test_partitions"][prefix]["metrics"]["bits_per_character"]
            )
            for prefix in PARTITIONS
        }
        for tag, report in reports.items()
    }
    prose = {tag: values["prose-"] for tag, values in bits.items()}
    spread = max(prose.values()) - min(prose.values())
    smallest, largest = "5m", "20m"
    worst = max(prose, key=lambda tag: prose[tag])
    if spread <= TIE_MARGIN + _EPSILON:
        outcome = "Q-b"
    elif worst == smallest and prose[smallest] - prose[largest] > TIE_MARGIN + _EPSILON:
        outcome = "Q-a"
    else:
        outcome = "Q-c"
    differences = {
        f"{left}_minus_{right}": {
            "difference": prose[left] - prose[right],
            "grade": _grade(prose[left] - prose[right]),
        }
        for left, right in (("5m", "10m"), ("10m", "20m"), ("5m", "20m"))
    }
    partitions = references["comparison_reference_T"]["partitions"]
    earlier = {
        tag: {
            "arm": name,
            "reference": float(partitions["prose-"]["values"][name]),
            "this_host_minus_reference": prose[tag] - float(partitions["prose-"]["values"][name]),
        }
        for tag, name in _EARLIER_HOST.items()
    }
    positions = (
        EXPECTED_UPDATES
        * int(next(iter(reports.values()))["transformer"]["budget"]["batch_size"])
        * int(next(iter(reports.values()))["plan"]["context_length"])
    )
    return {
        "outcome": outcome,
        "tie_margin": TIE_MARGIN,
        "separable_margin": SEPARABLE_MARGIN,
        "prose_bits_per_character": prose,
        "range": spread,
        "differences": differences,
        "arms": {
            tag: {
                "packed_identifiers": int(report["corpus"]["training_tokens_packed"]),
                "passes": positions / int(report["corpus"]["training_tokens_packed"]),
                "completion": report["transformer"]["budget"].get("completion"),
                "partitions": bits[tag],
                "equal_weight_mean": sum(bits[tag].values()) / len(PARTITIONS),
            }
            for tag, report in reports.items()
        },
        "earlier_host": earlier,
        "predictions": {
            "prose_non_decreasing_as_the_pool_shrinks": bool(
                prose["5m"] >= prose["10m"] - _EPSILON and prose["10m"] >= prose["20m"] - _EPSILON
            ),
            "ten_and_twenty_million_tie": abs(prose["10m"] - prose["20m"]) <= TIE_MARGIN + _EPSILON,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--arm", action="append", required=True, metavar="TAG=report.json")
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        reports: dict[str, Any] = {}
        for item in args.arm:
            tag, _, location = item.partition("=")
            if tag not in CAPS:
                message = f"arm {tag!r} is not one of {sorted(CAPS)}"
                raise ValueError(message)
            reports[tag] = json.loads(Path(location).read_text(encoding="utf-8"))
        if sorted(reports) != sorted(CAPS):
            message = f"need exactly the arms {sorted(CAPS)}, got {sorted(reports)}"
            raise ValueError(message)
        problems = _identity_problems(reports)
        if problems:
            print("error: these arms are not a pool-size comparison:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 2
        result = read(reports, json.loads(args.references.read_text(encoding="utf-8")))
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
