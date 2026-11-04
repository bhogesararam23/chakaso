"""Print or write the comparison of experiment arms, rebuilt from their report files.

The logic is :func:`chakaso.training.arm_comparison.compare_arms`; this wrapper only reads reports and writes the
result, so a record's committed ``results.json`` can be regenerated from the artifacts the record cites.

Usage:
    python scripts/compare_arms.py --arm P=data/experiments/E2612/prose/report.json \\
        --arm T=data/experiments/E2612/mixture/report.json --reference P \\
        --aggregate technical=science-,technology-,medicine-,mathematics- --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.core.atomic import write_text_atomically
from chakaso.training.arm_comparison import ArmComparisonError, compare_arms


def _pair(value: str, flag: str) -> tuple[str, str]:
    name, separator, rest = value.partition("=")
    if not separator or not name or not rest:
        message = f"{flag} expects NAME=VALUE, got {value!r}"
        raise argparse.ArgumentTypeError(message)
    return name, rest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--arm", action="append", default=[], help="NAME=report.json, one per arm")
    parser.add_argument(
        "--reference", required=True, help="the arm every difference is measured against"
    )
    parser.add_argument(
        "--aggregate",
        action="append",
        default=[],
        help="NAME=prefix,prefix,... : a scored-characters-weighted aggregate over partitions",
    )
    parser.add_argument(
        "--equal-aggregate",
        action="append",
        default=[],
        metavar="NAME=prefix,prefix,...",
        help="the same partitions averaged with equal weight, which is what the records call O1",
    )
    parser.add_argument("--out", type=Path, help="write the comparison here as deterministic JSON")
    args = parser.parse_args(argv)

    try:
        arms = {
            name: json.loads(Path(path).read_text(encoding="utf-8"))
            for name, path in (_pair(item, "--arm") for item in args.arm)
        }
        aggregate = {
            name: [prefix for prefix in rest.split(",") if prefix]
            for name, rest in (_pair(item, "--aggregate") for item in args.aggregate)
        }
        equal_weight = {
            name: [prefix for prefix in rest.split(",") if prefix]
            for name, rest in (_pair(item, "--equal-aggregate") for item in args.equal_aggregate)
        }
        result = compare_arms(
            arms,
            reference=args.reference,
            aggregate=aggregate or None,
            equal_weight_aggregates=equal_weight or None,
        )
    except (OSError, ValueError, ArmComparisonError, argparse.ArgumentTypeError) as error:
        print(f"compare_arms: {error}", file=sys.stderr)
        return 1

    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        write_text_atomically(payload, args.out)
    else:
        print(payload)
    print(
        "all conditions matched" if result["all_conditions_matched"] else "CONDITIONS NOT MATCHED",
        file=sys.stderr,
    )
    return 0 if result["all_conditions_matched"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
