"""Read a prose-share sweep from its report files and say what it does and does not support (H9).

The arms are supplied as ``SHARE=report.json``. The tool first proves the arms were matched (the same
comparison E2613 used), then computes per-register bits per character, the equal-weight mean over the
registers (the sweep's primary objective), the scored-characters-weighted technical aggregate, the margins the
seed noise of E2614 sets for each, the sweep's outcome, the arms the noise cannot separate from the best, and
which arms are better than which on every register. It chooses nothing the margins do not support.

Usage:
    python scripts/analyse_share_sweep.py --arm 0.0=data/experiments/E2615-s00/seed-7/report.json \\
        --arm 1.0=data/experiments/E2612/prose/report.json --noise-study data/experiments/E2614/study.json \\
        --natural-mixture-study data/experiments/E2614/study.json --out experiments/E2615/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.core.atomic import write_text_atomically
from chakaso.training.arm_comparison import ArmComparisonError, compare_arms
from chakaso.training.seed_study import aggregate_metric
from chakaso.training.share_sweep import (
    Margins,
    classify_difference,
    classify_share_sweep,
    dominance,
    equal_weight_mean,
    margins_from_stdev,
)

#: Budget fields a comparison requires but that records written before they existed do not carry. Both are fields
#: an arm is allowed to differ in, so an unrecorded value cannot mask a defect in a matched condition.
_DECLARABLE_BUDGET_FIELDS = ("shuffle_training_units", "document_masking")

PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
TECHNICAL = PARTITIONS[1:]


def _bits(report: Mapping[str, Any], prefix: str) -> float:
    return float(report["transformer"]["test_partitions"][prefix]["metrics"]["bits_per_character"])


def _characters(report: Mapping[str, Any], prefix: str) -> int:
    return int(report["transformer"]["test_partitions"][prefix]["metrics"]["characters_scored"])


def declared_share(report: Mapping[str, Any]) -> float | None:
    """The prose share a report's run was composed with: recorded, or implied by a prose-only filter."""
    budget = report["transformer"]["budget"]
    if budget.get("share_of_tokens") is not None:
        return float(budget["share_of_tokens"])
    if budget.get("training_group_prefixes") == ["prose-"]:
        return 1.0
    return None


def _noise_series(study: Mapping[str, Any], prefix: str) -> list[float]:
    key = f"partition_{prefix}_bits_per_character"
    return [float(run["metrics"][key]) for run in study["per_run"].values()]


def margins_by_objective(
    study: Mapping[str, Any], *, prose_band: float, mathematics_band: float, macro_band: float
) -> dict[str, Margins]:
    """Margins per register and for the equal-weight mean, from the seed spread the study measured."""
    series = {prefix: _noise_series(study, prefix) for prefix in PARTITIONS}
    macro = [
        sum(series[prefix][index] for prefix in PARTITIONS) / len(PARTITIONS)
        for index in range(len(series[PARTITIONS[0]]))
    ]
    bands: dict[str, tuple[float | None, bool]] = {
        "prose-": (prose_band, False),
        "mathematics-": (mathematics_band, True),
    }
    table: dict[str, Margins] = {}
    for name, values in (*series.items(), ("macro", macro)):
        summary = aggregate_metric(values)
        band, replaces = bands.get(name, (macro_band, False) if name == "macro" else (None, False))
        table[name] = margins_from_stdev(
            summary["sample_stdev"],
            summary["stdev_95_interval"][1],
            declared_band=band,
            band_replaces_central=replaces,
        )
    return table


def analyse(
    arms: Mapping[float, Mapping[str, Any]],
    *,
    noise_study: Mapping[str, Any],
    natural_mixture_study: Mapping[str, Any] | None,
    prose_band: float = 0.02,
    mathematics_band: float = 0.035,
    macro_band: float = 0.02,
) -> dict[str, Any]:
    """The whole reading of a sweep as one JSON-serialisable mapping."""
    names = {share: f"{share:g}" for share in arms}
    comparison = compare_arms(
        {names[share]: report for share, report in arms.items()}, reference=names[max(arms)]
    )
    for share, report in arms.items():
        recorded = declared_share(report)
        if recorded is None or not math.isclose(recorded, share, abs_tol=1e-9):
            message = (
                f"the arm labelled {share:g} was composed with share {recorded!r}; "
                "a label that disagrees with the report is a mislabelled arm"
            )
            raise ArmComparisonError(message)
    margins = margins_by_objective(
        noise_study,
        prose_band=prose_band,
        mathematics_band=mathematics_band,
        macro_band=macro_band,
    )
    per_arm: dict[float, dict[str, Any]] = {}
    for share, report in sorted(arms.items()):
        bits = {prefix: _bits(report, prefix) for prefix in PARTITIONS}
        weight = sum(_characters(report, prefix) for prefix in TECHNICAL)
        per_arm[share] = {
            "partitions": bits,
            "macro": equal_weight_mean(bits, PARTITIONS),
            "technical_weighted": sum(
                bits[prefix] * _characters(report, prefix) for prefix in TECHNICAL
            )
            / weight,
            "whole_pack": float(report["transformer"]["test"]["bits_per_character"]),
        }
    shares = sorted(per_arm)
    sweep = classify_share_sweep(
        {share: per_arm[share]["macro"] for share in shares}, margins["macro"].upper
    )
    steps: list[dict[str, Any]] = []
    for lower, higher in pairwise(shares):
        row: dict[str, Any] = {"from_share": lower, "to_share": higher}
        for name in (*PARTITIONS, "macro"):
            low = per_arm[lower]["macro"] if name == "macro" else per_arm[lower]["partitions"][name]
            high = (
                per_arm[higher]["macro"] if name == "macro" else per_arm[higher]["partitions"][name]
            )
            row[name] = {
                "difference": round(high - low, 6),
                "reading": classify_difference(high - low, margins[name]),
            }
        steps.append(row)
    result: dict[str, Any] = {
        "all_conditions_matched": comparison["all_conditions_matched"],
        "matched_conditions": {
            name: block["equal"] for name, block in comparison["matched_conditions"].items()
        },
        "declared_differences": sorted(comparison["declared_differences"]),
        "arms": {f"{share:g}": values for share, values in per_arm.items()},
        "margins": {
            name: {"central": margin.central, "upper": margin.upper}
            for name, margin in margins.items()
        },
        "primary_objective": "equal-weight mean of the five register partitions (lower is better)",
        "sweep": {key: value for key, value in sweep.items() if key != "values"},
        "adjacent_steps": steps,
        "dominance_pairs": [
            [first, second]
            for first, second in dominance(
                {share: per_arm[share]["partitions"] for share in shares},
                {prefix: margins[prefix].upper for prefix in PARTITIONS},
            )
        ],
    }
    if natural_mixture_study is not None:
        result["selection_effect"] = _selection_effect(per_arm, natural_mixture_study, margins)
    return result


def _selection_effect(
    per_arm: Mapping[float, Mapping[str, Any]],
    study: Mapping[str, Any],
    margins: Mapping[str, Margins],
) -> dict[str, Any]:
    """The nested half-prose arm against the natural mixture's five-seed mean: which documents were drawn.

    The comparison arm has one run and the natural mixture has ``n``, so the pair band shrinks by
    ``sqrt((1 + 1/n) / 2)``: a single run against a mean of ``n`` has less variance than two single runs.
    """
    runs = len(study["per_run"])
    scale = math.sqrt((1 + 1 / runs) / 2)
    half = per_arm.get(0.5)
    if half is None:
        return {"available": False, "reason": "no arm with share 0.5"}
    rows: dict[str, Any] = {}
    for name in (*PARTITIONS, "macro"):
        series = (
            [
                sum(_noise_series(study, prefix)[index] for prefix in PARTITIONS) / len(PARTITIONS)
                for index in range(runs)
            ]
            if name == "macro"
            else _noise_series(study, name)
        )
        mean = sum(series) / runs
        value = half["macro"] if name == "macro" else half["partitions"][name]
        scaled = Margins(central=margins[name].central * scale, upper=margins[name].upper * scale)
        rows[name] = {
            "nested_half_minus_natural_mean": round(value - mean, 6),
            "reading": classify_difference(value - mean, scaled),
        }
    return {"available": True, "natural_mixture_runs": runs, "differences": rows}


def adapt_committed_arm(block: Mapping[str, Any], donor: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a report-shaped mapping from a committed ``results.json`` arm block.

    A record's committed aggregate keeps an arm flat and without the per-partition character counts that the
    scored-characters weighting (K-027) needs. The held-out packs are identical across arms that passed the
    matched-conditions proof, so those counts are taken from ``donor``, a full report of an arm that did, but only
    after the token count of every partition has been checked equal. What was borrowed is named in the returned
    mapping under ``adapted_from_committed_summary``, so a reader sees that it was not measured on this arm.

    Raises:
        ArmComparisonError: a partition is missing from either side or its scored-token count differs.
    """
    donor_partitions = donor["transformer"]["test_partitions"]
    partitions: dict[str, Any] = {}
    checked = ["tokens_scored"]
    for prefix, entry in block["test_partitions"].items():
        if prefix not in donor_partitions:
            message = (
                f"the committed arm scores partition {prefix!r}, which the donor report does not"
            )
            raise ArmComparisonError(message)
        reference = donor_partitions[prefix]
        if entry["tokens_scored"] != reference["tokens_scored"]:
            message = (
                f"partition {prefix!r} scored {entry['tokens_scored']} tokens in the committed arm and "
                f"{reference['tokens_scored']} in the donor: the held-out packs differ, so characters cannot "
                "be borrowed"
            )
            raise ArmComparisonError(message)
        if "documents" in entry and "documents" in reference:
            if entry["documents"] != reference["documents"]:
                message = (
                    f"partition {prefix!r} holds a different number of documents than the donor's"
                )
                raise ArmComparisonError(message)
            if "documents" not in checked:
                checked.append("documents")
        metrics = {
            "bits_per_character": entry["bits_per_character"],
            "characters_scored": reference["metrics"]["characters_scored"],
        }
        partitions[prefix] = {"tokens_scored": entry["tokens_scored"], "metrics": metrics}
    budget = dict(block["budget"])
    unrecorded = sorted(key for key in _DECLARABLE_BUDGET_FIELDS if key not in budget)
    for key in unrecorded:
        budget[key] = (
            None  # "not recorded", never a guess: these fields are allowed to differ between arms
        )
    return {
        "corpus": block["corpus"],
        "tokenizer": block["tokenizer"],
        "transformer": {
            "parameter_count": block["parameter_count"],
            "architecture": block["architecture"],
            "budget": budget,
            "test": block["test"],
            "test_partitions": partitions,
        },
        "adapted_from_committed_summary": {
            "borrowed_from_a_full_report": ["test_partitions[*].metrics.characters_scored"],
            "checked_equal_first": sorted(checked),
            "not_recorded_in_committed_summary": unrecorded,
        },
    }


def _load_arm(path: Path) -> tuple[Path, str | None]:
    """Split ``results.json#name`` into the file and the arm block name; a plain path has no name."""
    text = str(path)
    file_part, separator, name = text.partition("#")
    return Path(file_part), (name if separator else None)


def _pair(value: str) -> tuple[float, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        message = f"--arm expects SHARE=report.json or SHARE=results.json#arm, got {value!r}"
        raise argparse.ArgumentTypeError(message)
    try:
        return float(label), Path(path)
    except ValueError as error:
        message = f"--arm share must be a number, got {label!r}"
        raise argparse.ArgumentTypeError(message) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--arm", type=_pair, action="append", default=[])
    parser.add_argument("--noise-study", type=Path, required=True)
    parser.add_argument("--natural-mixture-study", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    try:
        arms: dict[float, Any] = {}
        committed: dict[float, Any] = {}
        for share, path in args.arm:
            file_path, block_name = _load_arm(path)
            document = json.loads(file_path.read_text(encoding="utf-8"))
            if block_name is None:
                arms[share] = document
            else:
                committed[share] = document["arms"][block_name]
        if len(arms) + len(committed) != len(args.arm):
            message = "two arms carry the same share"
            raise ArmComparisonError(message)
        if committed and not arms:
            message = "a committed-summary arm needs at least one full report to take the held-out pack from"
            raise ArmComparisonError(message)
        for share, block in committed.items():
            arms[share] = adapt_committed_arm(block, next(iter(arms.values())))
        noise = json.loads(args.noise_study.read_text(encoding="utf-8"))
        natural = (
            json.loads(args.natural_mixture_study.read_text(encoding="utf-8"))
            if args.natural_mixture_study
            else None
        )
        result = analyse(arms, noise_study=noise, natural_mixture_study=natural)
    except (ArmComparisonError, ValueError, OSError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    adapted = {
        f"{share:g}": report["adapted_from_committed_summary"]
        for share, report in arms.items()
        if "adapted_from_committed_summary" in report
    }
    if adapted:
        result["borrowed_from_a_full_report"] = adapted
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomically(text, args.out)
    else:
        sys.stdout.write(text)
    if not result["all_conditions_matched"]:
        print("error: the arms were not matched; no reading of the sweep is valid", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
