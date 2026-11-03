"""Ask the capability probes to rank two independently trained arms, and prove the arms were matched first.

E2626 measured what the probes can see along one training curve and closed with the question a curve cannot answer:
rungs share an initialisation, a data order and nearly all their weights, so a rung pair is the easiest ranking an
instrument gets, and an arm pair is the one that matters. This script hands the probes an arm pair. It reads nothing
but reports and probe results already on disk, so the comparison is rebuilt from committed evidence, and it refuses
the analysis before any statistic is computed when the arms differ in a condition the record did not declare — the
same matchedness guard E2612 and E2613 used for their loss tables (``chakaso.training.arm_comparison``), applied here
to the probes.

The criterion is whole-pack held-out bits per character, lower better, and the materiality band is the caller's. A
pair the band does not separate is still tested, which is what makes it a false-positive check rather than a
left-out case.

Usage:
    python scripts/analyse_arm_pairs.py --arm mixture=experiments/E2612/report.json \\
        --arm prose=experiments/E2612/prose-report.json --reference mixture \\
        --pairs experiments/E2629/minimal-pairs.json --continuations experiments/E2629/continuations.json \\
        --usable-phenomena auxiliary_agreement,determiner_noun_agreement,reflexive_pronoun_agreement \\
        --materiality 0.02 --out experiments/E2629/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyse_probe_sensitivity import (
    CONTROL_MODEL,
    AnalysisError,
    context_outcomes,
    continuation_outcomes,
    criterion_table,
    pair_outcomes,
)

from chakaso.training.arm_comparison import compare_arms
from chakaso.training.probe_sensitivity import compare_instruments

__all__ = ["main", "read"]

#: The shared readers name the probe's untrained floor after a curve's rung 0, because that is what it is there.
#: An arm comparison has no rungs, so a model left out of it is reported under the name the result file uses.
_TO_FILE_NAME = {"step0": CONTROL_MODEL}


def arm_reports(entries: Sequence[str]) -> dict[str, dict[str, Any]]:
    """``{name: report}`` for each ``NAME=path`` entry, refusing a name used twice."""
    reports: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name, _, location = entry.partition("=")
        if not name or not location:
            message = f"--arm takes NAME=path, got {entry!r}"
            raise AnalysisError(message)
        if name in reports:
            message = f"arm name {name!r} is given twice"
            raise AnalysisError(message)
        path = Path(location)
        try:
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            message = f"arm {name!r}: {error}"
            raise AnalysisError(message) from error
        if reports[name].get("experiment") is None:
            message = f"arm {name!r} in {location} is not an experiment report"
            raise AnalysisError(message)
    return reports


def matchedness(reports: Mapping[str, Mapping[str, Any]], *, reference: str) -> dict[str, Any]:
    """The arms' matched-ness table, refusing an undeclared difference before any probe is read."""
    if reference not in reports:
        message = f"reference arm {reference!r} is not among {sorted(reports)}"
        raise AnalysisError(message)
    comparison = compare_arms(reports, reference=reference)
    if not comparison["all_conditions_matched"]:
        unequal = [
            condition
            for condition, block in comparison["matched_conditions"].items()
            if not block["equal"]
        ]
        message = (
            "the arms are not matched in "
            f"{unequal}; a probe ranking of two arms that differ in a condition the record did "
            "not declare ranks the condition, not the quality"
        )
        raise AnalysisError(message)
    return comparison


def read(
    *,
    reports: Mapping[str, Mapping[str, Any]],
    reference: str,
    pairs_result: Mapping[str, Any],
    continuations_result: Mapping[str, Any],
    usable_phenomena: Sequence[str],
    materiality: float,
    decision_bearing: Sequence[str],
    context_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The arm pair, the criterion read from the reports, and each instrument's verdict on every pair."""
    comparison = matchedness(reports, reference=reference)
    table = criterion_table({name: report["transformer"] for name, report in reports.items()})
    instruments: dict[str, dict[str, dict[str, float]]] = {
        "minimal_pairs_usable": pair_outcomes(pairs_result, phenomena=usable_phenomena),
        "minimal_pairs_all_800": pair_outcomes(pairs_result, phenomena=None),
        "continuation_pmi_easy": continuation_outcomes(continuations_result, reading="pmi_easy"),
        "continuation_pmi_hard": continuation_outcomes(continuations_result, reading="pmi_hard"),
    }
    if context_result is not None:
        instruments.update(context_outcomes(context_result))
    # The criterion table, not a result file, defines the population. A probe run usually carries an untrained
    # floor and other arms beside the ones this record compares; the extras are reported and not used, and a
    # missing arm is a refusal, because a pair without a score would silently shrink the family Holm corrects over.
    extras = sorted(
        _TO_FILE_NAME.get(model, model)
        for model in {model for scores in instruments.values() for model in scores} - set(table)
    )
    for name, scores in instruments.items():
        missing = sorted(set(table) - set(scores))
        if missing:
            message = (
                f"instrument {name!r} scored no {missing}; the arms it left out are not compared"
            )
            raise AnalysisError(message)
    instruments = {
        name: {model: scores for model, scores in table_of_model.items() if model in table}
        for name, table_of_model in instruments.items()
    }
    criterion = {
        name: float(block["whole_pack_bits_per_character"]) for name, block in table.items()
    }
    result = compare_instruments(criterion, instruments, materiality=materiality)
    result["arms"] = {
        name: {
            "experiment": report.get("experiment"),
            "seed": report["transformer"]["budget"].get("seed"),
            "max_steps": report["transformer"]["budget"].get("max_steps"),
            "whole_pack_bits_per_character": table[name]["whole_pack_bits_per_character"],
            "o1_equal_weight_mean": table[name]["o1_equal_weight_mean"],
            "vocabulary": report["tokenizer"]["fingerprint"],
            "split_fingerprint": report["corpus"]["split_fingerprint"],
        }
        for name, report in reports.items()
    }
    result["reference_arm"] = reference
    result["models_scored_but_not_analysed"] = extras
    result["matchedness"] = comparison["declared_differences"]
    result["materiality"] = materiality
    result["decision_bearing"] = list(decision_bearing)
    result["usable_phenomena"] = list(usable_phenomena)
    result["outcome"] = arm_outcome(result, decision_bearing)
    return result


def arm_outcome(result: Mapping[str, Any], decision_bearing: Sequence[str]) -> dict[str, Any]:
    """Z-a to Z-d, by the rules the record declares: rank the material pairs, leave the rest alone.

    ``compare_instruments`` gives the per-pair McNemar cells; this folds them into the one answer the record asked
    for, over the instruments the record declared decision-bearing.

    * A pair is **material** when the criterion band separates its arms. Ranking *every* material pair in the
      criterion's direction is what the record asked an instrument to do.
    * A pair the band calls equal is a false-positive check: an instrument that "separates" it is reporting a
      difference the quality measure does not have.
    * **Z-a** — some decision-bearing instrument ranks every material pair in the criterion's direction, and no
      decision-bearing instrument contradicts the criterion or separates the pair inside the band.
    * **Z-b** — some decision-bearing instrument ranks at least one material pair correctly, and none contradicts.
    * **Z-c** — a decision-bearing instrument either contradicts the criterion or separates the inside-the-band pair.
    * **Z-d** — none of the above, which on this design means no instrument separated anything.
    """
    rows = result["pairs"]
    per_instrument: dict[str, dict[str, Any]] = {}
    for name in decision_bearing:
        material = [row for row in rows if row["material"]]
        inside = [row for row in rows if not row["material"]]
        agreed = [
            row for row in material if row[name]["separated"] and row[name]["direction"] == "agrees"
        ]
        contradicted = [
            row
            for row in rows
            if row[name]["separated"] and row[name]["direction"] == "contradicts"
        ]
        false_positive = [row for row in inside if row[name]["separated"]]
        per_instrument[name] = {
            "material_pairs": len(material),
            "material_pairs_ranked_correctly": len(agreed),
            "ranks_every_material_pair": bool(material) and len(agreed) == len(material),
            "contradicted_pairs": [[row["first"], row["second"]] for row in contradicted],
            "inside_band_pairs_separated": [
                [row["first"], row["second"]] for row in false_positive
            ],
            "verdict": result["instruments"][name]["verdict"],
        }
    any_contradiction = any(block["contradicted_pairs"] for block in per_instrument.values())
    any_false_positive = any(
        block["inside_band_pairs_separated"] for block in per_instrument.values()
    )
    every_ranker = [
        name for name, block in per_instrument.items() if block["ranks_every_material_pair"]
    ]
    any_ranker = [
        name
        for name, block in per_instrument.items()
        if block["material_pairs_ranked_correctly"] > 0
    ]
    if every_ranker and not any_contradiction and not any_false_positive:
        letter = "Z-a"
        reading = (
            f"{', '.join(every_ranker)} ranks every material pair in the criterion's direction"
        )
    elif any_ranker and not any_contradiction:
        letter = "Z-b"
        reading = (
            f"{', '.join(any_ranker)} ranks at least one material pair correctly, none contradicts"
        )
    elif any_contradiction or any_false_positive:
        letter = "Z-c"
        reading = (
            "an instrument ranked a pair the wrong way or separated a pair the band calls equal"
        )
    else:
        letter = "Z-d"
        reading = "no decision-bearing instrument separated a material pair"
    return {
        "letter": letter,
        "reading": reading,
        "per_instrument": per_instrument,
        "material_pairs": sum(1 for row in rows if row["material"]),
        "inside_band_pairs": sum(1 for row in rows if not row["material"]),
    }


def summarise(result: Mapping[str, Any]) -> str:
    """A short text rendering, so the command that ran is also readable on a terminal."""
    lines = [f"arms: {', '.join(result['models'])} (best first by whole-pack bits per character)"]
    for row in result["pairs"]:
        gap = float(row["criterion_gap"])
        lines.append(
            f"{'  ' if not row['material'] else '* '}{row['first']} vs {row['second']}: "
            f"gap {abs(gap):.5f} {'material' if row['material'] else 'inside the band'}"
        )
        for name in result["instruments"]:
            cell = row[name]
            lines.append(
                f"      {name}: separated={cell['separated']} "
                f"direction={cell['direction']} discordant={cell['discordant']} p={cell['p_value']:.3g}"
            )
    lines.append("")
    for name, block in result["instruments"].items():
        lines.append(
            f"{name}: rho {block['rho']:.4f}, "
            f"{block['material_pairs_separated']} of {block['material_pairs']} material pairs, "
            f"smallest separated {block['smallest_separated_gap']}, "
            f"largest missed {block['largest_unseparated_material_gap']} -> {block['verdict']}"
        )
    lines.append(f"outcome: {result['outcome']['letter']} — {result['outcome']['reading']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=path",
        help="an arm and the report that scored it; at least two, and one is --reference",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--continuations", type=Path, required=True)
    parser.add_argument(
        "--usable-phenomena",
        required=True,
        help="comma-separated phenomenon names the record declared usable (E2622)",
    )
    parser.add_argument(
        "--context",
        type=Path,
        default=None,
        help="a same-region probe result (E2627); one instrument per distractor kind",
    )
    parser.add_argument("--materiality", type=float, default=0.02)
    parser.add_argument(
        "--decision-bearing",
        default="minimal_pairs_usable,continuation_pmi_easy,continuation_pmi_hard",
        help="comma-separated instrument names whose verdicts decide the outcome",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = read(
            reports=arm_reports(args.arm),
            reference=args.reference,
            pairs_result=_load(args.pairs),
            continuations_result=_load(args.continuations),
            usable_phenomena=args.usable_phenomena.split(","),
            materiality=args.materiality,
            decision_bearing=args.decision_bearing.split(","),
            context_result=_load(args.context) if args.context else None,
        )
    except (AnalysisError, OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(summarise(result), file=sys.stderr)
    return 0


def _load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        message = f"{path}: {error}"
        raise AnalysisError(message) from error


if __name__ == "__main__":
    raise SystemExit(main())
