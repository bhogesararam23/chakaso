"""Read E2616's one-pass control against the reference values its record fixed before the run.

The record declares the primary metric (held-out prose bits per character of the one-pass arm, X), the references
(T, H and P from E2612/E2613, read here from E2613's committed results), the tie margin (0.00625) and the separable
margin (0.02), and three outcomes:

* R-a: X >= T - tie           the second pass was the difference (X scores like T or worse);
* R-b: X <= H + tie           technical text displaced prose learning (X scores like H);
* R-c: anything between       reported with X's position between H and T as a fraction.

If X is worse than T by more than the separable margin the output also says so, because the record states that this
makes the arm evidence that its shorter schedule is too short for the budget. It also checks that the packed training
text equals E2613-H's, and reports the other four partitions and the equal-weight mean against H and P, which the
record lists as not decision-bearing.

Usage:
    python scripts/read_one_pass_control.py --report experiments/E2616/E2616-report.json \\
        --references experiments/E2613/results.json --out experiments/E2616/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")

#: Fixed in experiments/E2616/record.md before the run; here only so they are visible and testable.
TIE_MARGIN = 0.00625
SEPARABLE_MARGIN = 0.02
EXPECTED_PACKED_IDENTIFIERS = 10_001_764


def classify(x: float, *, t: float, h: float) -> dict[str, Any]:
    """The record's three outcomes for prose bits per character ``x`` against references ``t`` and ``h``."""
    if t - h <= 2 * TIE_MARGIN:
        message = (
            "the references are within two tie margins of each other: the outcomes would overlap"
        )
        raise ValueError(message)
    if x >= t - TIE_MARGIN:
        outcome = "R-a"
    elif x <= h + TIE_MARGIN:
        outcome = "R-b"
    else:
        outcome = "R-c"
    result: dict[str, Any] = {
        "outcome": outcome,
        "x_minus_t": round(x - t, 6),
        "x_minus_h": round(x - h, 6),
    }
    if outcome == "R-a" and x - t > SEPARABLE_MARGIN:
        result["worse_than_t_by_more_than_the_separable_margin"] = True
    if outcome == "R-c":
        result["fraction_of_the_way_from_h_to_t"] = round((x - h) / (t - h), 6)
    return result


def read(report: Mapping[str, Any], references: Mapping[str, Any]) -> dict[str, Any]:
    partitions = references["comparison_reference_T"]["partitions"]
    bits = {
        prefix: float(
            report["transformer"]["test_partitions"][prefix]["metrics"]["bits_per_character"]
        )
        for prefix in PARTITIONS
    }
    refs = {
        arm: {prefix: float(partitions[prefix]["values"][arm]) for prefix in PARTITIONS}
        for arm in ("T", "H", "P")
    }
    packed = int(report["corpus"]["training_tokens_packed"])
    macro = {
        name: sum(values.values()) / len(PARTITIONS) for name, values in {"X": bits, **refs}.items()
    }
    return {
        "primary": {
            "x_prose_bits_per_character": bits["prose-"],
            "references": {arm: refs[arm]["prose-"] for arm in refs},
            "tie_margin": TIE_MARGIN,
            "separable_margin": SEPARABLE_MARGIN,
            **classify(bits["prose-"], t=refs["T"]["prose-"], h=refs["H"]["prose-"]),
        },
        "training_text_identical_to_e2613_h": packed == EXPECTED_PACKED_IDENTIFIERS,
        "training_tokens_packed": packed,
        "not_decision_bearing": {
            "partitions": {
                prefix: {"X": bits[prefix], **{arm: refs[arm][prefix] for arm in refs}}
                for prefix in PARTITIONS
            },
            "equal_weight_mean": {name: round(value, 6) for name, value in macro.items()},
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        result = read(
            json.loads(args.report.read_text(encoding="utf-8")),
            json.loads(args.references.read_text(encoding="utf-8")),
        )
    except (OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    if not result["training_text_identical_to_e2613_h"]:
        print(
            "error: the packed training text is not E2613-H's; the comparison is not valid",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
