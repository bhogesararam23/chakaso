"""Read E2620's tokenizer arm against its control by the rules the record declared before the run.

The control is E2617's report (the E2614 configuration on this host, default pre-tokenizer); the treatment is the same configuration with the vocabulary
fitted under the space-prefixed rule. Let delta = control - treatment (positive when the treatment is better). With the record's margins:

* V-a better:          delta(O1) > 0.02  and  delta(whole pack) > 0.00933
* V-b worse:           delta(O1) < -0.02 and  delta(whole pack) < -0.00933
* V-c no difference:   |delta(O1)| <= 0.0062 and |delta(whole pack)| <= 0.00933
* V-d indeterminate:   anything else

It first refuses a treatment that is not a tokenizer-only comparison: another corpus or split, another parameter count or budget, a default-rule tokenizer,
no recorded evaluation selection, a selection vocabulary other than the control's, or held-out documents that are not the control's.

Usage:
    python scripts/read_tokenizer_arm.py --treatment data/experiments/E2620/seed-7/report.json \\
        --control experiments/E2617/E2617-report.json --out experiments/E2620/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")

#: Fixed in experiments/E2620/record.md before the run; here only so they are visible and testable.
O1_TIE = 0.0062
O1_SEPARABLE = 0.02
WHOLE_TIE = 0.00933
_EPSILON = 1e-12


def o1(report: Mapping[str, Any]) -> float:
    partitions = report["transformer"]["test_partitions"]
    return sum(
        float(partitions[prefix]["metrics"]["bits_per_character"]) for prefix in PARTITIONS
    ) / len(PARTITIONS)


def whole_pack(report: Mapping[str, Any]) -> float:
    return float(report["transformer"]["test"]["bits_per_character"])


def classify(delta_o1: float, delta_whole: float) -> str:
    """The record's four outcomes for the two decision-bearing differences."""
    if delta_o1 > O1_SEPARABLE + _EPSILON and delta_whole > WHOLE_TIE + _EPSILON:
        return "V-a"
    if delta_o1 < -O1_SEPARABLE - _EPSILON and delta_whole < -WHOLE_TIE - _EPSILON:
        return "V-b"
    if abs(delta_o1) <= O1_TIE + _EPSILON and abs(delta_whole) <= WHOLE_TIE + _EPSILON:
        return "V-c"
    return "V-d"


def matchedness_problems(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    equal_text: bool = False,
    same_documents: bool = False,
) -> list[str]:
    problems: list[str] = []
    for label, left, right in (
        (
            "corpus fingerprint",
            treatment["corpus"]["fingerprint"],
            control["corpus"]["fingerprint"],
        ),
        (
            "split fingerprint",
            treatment["corpus"]["split_fingerprint"],
            control["corpus"]["split_fingerprint"],
        ),
        (
            "parameter count",
            treatment["transformer"]["parameter_count"],
            control["transformer"]["parameter_count"],
        ),
        (
            "batch size",
            treatment["transformer"]["budget"]["batch_size"],
            control["transformer"]["budget"]["batch_size"],
        ),
    ):
        if left != right:
            problems.append(f"{label}: treatment {left!r}, control {right!r}")
    documents = (
        "training documents packed",
        treatment["corpus"]["training_documents_packed"],
        control["corpus"]["training_documents_packed"],
    )
    updates = (
        "updates",
        treatment["transformer"]["budget"]["max_steps"],
        control["transformer"]["budget"]["max_steps"],
    )
    if equal_text:
        # Equal text: the updates differ by design; what must be equal is the training documents the two arms read.
        checks = (documents,)
    elif same_documents:
        # Equal updates and equal documents (E2625): both must match.
        checks = (updates, documents)
    else:
        checks = (updates,)
    for label, left, right in checks:
        if left != right:
            problems.append(f"{label}: treatment {left!r}, control {right!r}")
    rule = treatment["tokenizer"].get("pre_tokenizer")
    if not isinstance(rule, str) or not rule.startswith("space-prefixed"):
        problems.append(f"treatment pre-tokenizer is {rule!r}, not a space-prefixed rule")
    if "pre_tokenizer" in control["tokenizer"]:
        problems.append("the control is not under the default pre-tokenizer")
    selection = treatment["corpus"].get("evaluation_selection")
    if not isinstance(selection, dict):
        problems.append(
            "the treatment records no evaluation_selection: its held-out text is not fixed"
        )
        return problems
    if selection.get("tokenizer_fingerprint") != control["tokenizer"]["fingerprint"]:
        problems.append(
            "the held-out documents were chosen by "
            f"{selection.get('tokenizer_fingerprint')!r}, not by the control's vocabulary "
            f"{control['tokenizer']['fingerprint']!r}"
        )
    if treatment["corpus"]["test_documents_packed"] != control["corpus"]["test_documents_packed"]:
        problems.append(
            "test documents packed differ: treatment "
            f"{treatment['corpus']['test_documents_packed']}, control {control['corpus']['test_documents_packed']}"
        )
    for prefix in PARTITIONS:
        left = treatment["transformer"]["test_partitions"][prefix]["documents_packed"]
        right = control["transformer"]["test_partitions"][prefix]["documents_packed"]
        if left != right:
            problems.append(f"{prefix} documents packed differ: treatment {left}, control {right}")
    return problems


def read(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    equal_text: bool = False,
    same_documents: bool = False,
) -> dict[str, Any]:
    """The record's reading of one treatment against its control."""
    delta_o1 = o1(control) - o1(treatment)
    delta_whole = whole_pack(control) - whole_pack(treatment)
    return {
        "mode": (
            "equal-text"
            if equal_text
            else "equal-updates-and-documents"
            if same_documents
            else "equal-positions"
        ),
        "outcome": classify(delta_o1, delta_whole),
        "delta_o1": delta_o1,
        "delta_whole_pack": delta_whole,
        "o1": {"control": o1(control), "treatment": o1(treatment)},
        "whole_pack": {"control": whole_pack(control), "treatment": whole_pack(treatment)},
        "margins": {"o1_tie": O1_TIE, "o1_separable": O1_SEPARABLE, "whole_tie": WHOLE_TIE},
        "characters_per_identifier": {
            "control": control["tokenizer"].get("characters_per_token_test"),
            "treatment": treatment["tokenizer"].get("characters_per_token_test"),
        },
        "partitions": {
            prefix: {
                "control": control["transformer"]["test_partitions"][prefix]["metrics"][
                    "bits_per_character"
                ],
                "treatment": treatment["transformer"]["test_partitions"][prefix]["metrics"][
                    "bits_per_character"
                ],
                "characters_scored": {
                    "control": control["transformer"]["test_partitions"][prefix]["metrics"].get(
                        "characters_scored"
                    ),
                    "treatment": treatment["transformer"]["test_partitions"][prefix]["metrics"].get(
                        "characters_scored"
                    ),
                },
            }
            for prefix in PARTITIONS
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--equal-text",
        action="store_true",
        help="the arms read the same training documents for different numbers of updates (E2624); the update-count guard becomes a training-documents guard",
    )
    parser.add_argument(
        "--same-documents",
        action="store_true",
        help="the arms make the same updates over the same training documents (E2625); both guards apply",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.equal_text and args.same_documents:
        parser.error("--equal-text and --same-documents describe different designs; give one")
    try:
        treatment = json.loads(args.treatment.read_text(encoding="utf-8"))
        control = json.loads(args.control.read_text(encoding="utf-8"))
        problems = matchedness_problems(
            treatment, control, equal_text=args.equal_text, same_documents=args.same_documents
        )
        if problems:
            print("error: this is not a tokenizer-only comparison:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 2
        result = read(
            treatment, control, equal_text=args.equal_text, same_documents=args.same_documents
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
