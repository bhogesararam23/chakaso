"""Read one experiment's training curve and ask what each capability probe could see along it.

The inputs are all files a run and the two probe scripts already produce: the run's report (which names its
rungs and the held-out loss recorded for each), the rung artifacts themselves, and the two probe result files
with their per-item outcomes. The output is the verdict preregistered in ``experiments/E2626/record.md``: for
every pair of models whose held-out bits per character differ by more than the project's materiality band, does
the probe separate them, in the right direction, and what is the smallest gap it can resolve at all.

Nothing is typed in. The bits per character of each rung are computed here from the rung's bytes through
:func:`chakaso.training.subword_experiment.score_artifacts` — the same packing and the same arithmetic the run
used for its own final number — and the script refuses the whole analysis if the endpoint rung does not
reproduce the report's own test block, because that equality is the check that writing the curve did not
perturb it (E2626's prediction P4).

Usage:
    python scripts/analyse_probe_sensitivity.py --report data/experiments/E2626/seed-7/report.json \\
        --protocol configs/experiments/E2626-curve-ladder.toml --work-dir data/experiments/E2626/seed-7 \\
        --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --split-seed 7 --expected-corpus-fingerprint 312571f9fa8351a8 \\
        --pairs experiments/E2626/minimal-pairs.json --continuations experiments/E2626/continuations.json \\
        --usable-phenomena auxiliary_agreement,determiner_noun_agreement,reflexive_pronoun_agreement \\
        --materiality 0.02 --out experiments/E2626/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
#: The name ``evaluate_minimal_pairs.py`` and ``evaluate_continuations.py`` give their untrained floor.
CONTROL_MODEL = "untrained-control"
#: What the record calls rung 0, so the control enters the same table as the rungs.
RUNG_ZERO = "step0"


class AnalysisError(RuntimeError):
    """The files handed to the analysis do not describe one experiment."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rung_names(report: Mapping[str, Any]) -> dict[str, str]:
    """``{rung key: artifact file name}`` for every rung the run wrote, rung 0 first.

    Rung 0 is the random initialisation the run started from, which the probe scripts build with their
    ``--control`` flag; it is in the table because the largest gap on the curve is the one that matters most.
    """
    rungs = report["transformer"].get("curve_artifacts") or []
    if not rungs:
        message = "this report wrote no rungs, so there is no curve to score against"
        raise AnalysisError(message)
    named = {RUNG_ZERO: "untrained"}
    for rung in sorted(rungs, key=lambda entry: int(entry["step"])):
        named[f"step{int(rung['step'])}"] = str(rung["path"])
    return named


def criterion_table(
    scored: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bits per character per rung: the whole held-out pack and the equal-weight register mean.

    The whole pack is the decision-bearing column because that is what E2614's band was measured on; the
    register mean is reported beside it because the mixture records use it as their primary objective.
    """
    table: dict[str, dict[str, Any]] = {}
    for name, block in scored.items():
        partitions = block["test_partitions"]
        missing = [prefix for prefix in PARTITIONS if "metrics" not in partitions.get(prefix, {})]
        if missing:
            message = f"{name} has no scored partition for {missing}"
            raise AnalysisError(message)
        table[name] = {
            "whole_pack_bits_per_character": float(block["test"]["bits_per_character"]),
            # The denominators and the per-partition numbers travel with the headline, so a later analysis can
            # rebuild this table and still run the endpoint guard instead of trusting a bare float.
            "positions_scored": int(block["test"]["positions_scored"]),
            "characters_scored": int(block["test"]["characters_scored"]),
            "test_partitions": {
                prefix: {
                    "bits_per_character": float(partitions[prefix]["metrics"]["bits_per_character"])
                }
                for prefix in PARTITIONS
            },
            "o1_equal_weight_mean": round(
                sum(
                    float(partitions[prefix]["metrics"]["bits_per_character"])
                    for prefix in PARTITIONS
                )
                / len(PARTITIONS),
                6,
            ),
        }
    return table


def check_endpoint_matches_the_report(
    scored: Mapping[str, Mapping[str, Any]], report: Mapping[str, Any], *, endpoint: str
) -> None:
    """Refuse the analysis if the final rung is not the model the run reported (E2626's P4).

    The claim under test is that writing a usable artifact at each evaluation does not change the run. If the
    rung at the selected step scores differently from the report's own test block, the curve this script is
    about to order is not the curve the run trained, and no rung in it means anything.
    """
    recorded = report["transformer"]["test"]
    measured = scored[endpoint]["test"]
    for field in ("bits_per_character", "positions_scored", "characters_scored"):
        if float(recorded[field]) != float(measured[field]):
            message = (
                f"rung {endpoint} scores {measured[field]!r} {field} but the run reported "
                f"{recorded[field]!r}; the ladder did not reproduce its own endpoint, so no rung here is "
                "the curve the run trained"
            )
            raise AnalysisError(message)


def pair_outcomes(
    pairs_result: Mapping[str, Any], *, phenomena: Sequence[str] | None
) -> dict[str, dict[str, float]]:
    """Per-item outcomes of the minimal-pair probe, optionally restricted to named phenomena.

    E2622 found that five of its eight phenomena have an untrained floor that is not at chance, and declared
    them excluded from any per-phenomenon reading; a sensitivity figure computed over all 800 pairs would be
    partly a measurement of whichever bias carries those three. The exclusion is passed in from the record,
    never inferred here.
    """
    tables: dict[str, dict[str, float]] = {}
    for model, block in pairs_result["models"].items():
        outcomes = block["result"].get("outcomes")
        if not outcomes:
            message = (
                f"{model}: this pair result has no per-item outcomes, so it was scored before the probe "
                "recorded them; re-score it"
            )
            raise AnalysisError(message)
        selected = [entry for entry in outcomes if phenomena is None or str(entry[1]) in phenomena]
        if not selected:
            message = f"{model}: no pair matched the usable phenomena {list(phenomena)}"
            raise AnalysisError(message)
        tables[_rename(model)] = {str(entry[0]): float(entry[2]) for entry in selected}
    return tables


def context_outcomes(
    continuations_result: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    """One instrument per same-region distractor kind: ``context_pmi_prior_span`` and its three siblings."""
    from chakaso.training.context_continuations import CONTEXT_DISTRACTOR_KINDS

    tables: dict[str, dict[str, dict[str, float]]] = {}
    for kind in CONTEXT_DISTRACTOR_KINDS:
        reading = f"pmi_{kind}"
        per_model: dict[str, dict[str, float]] = {}
        for model, block in continuations_result["models"].items():
            outcomes = block["result"].get("outcomes") or {}
            if reading not in outcomes:
                message = f"{model}: no per-item outcomes for the same-region reading {reading!r}"
                raise AnalysisError(message)
            per_model[_rename(model)] = {
                str(entry[0]): float(entry[1]) for entry in outcomes[reading]
            }
        tables[f"context_pmi_{kind}"] = per_model
    return tables


def continuation_outcomes(
    continuations_result: Mapping[str, Any], *, reading: str
) -> dict[str, dict[str, float]]:
    """Per-item outcomes of one reading of the continuation probe."""
    tables: dict[str, dict[str, float]] = {}
    for model, block in continuations_result["models"].items():
        outcomes = block["result"].get("outcomes")
        if not outcomes or reading not in outcomes:
            message = f"{model}: no per-item outcomes for reading {reading!r}"
            raise AnalysisError(message)
        tables[_rename(model)] = {str(entry[0]): float(entry[1]) for entry in outcomes[reading]}
    return tables


def _rename(model: str) -> str:
    """The probe scripts' control name becomes rung 0; every other name is already a rung."""
    return RUNG_ZERO if model == CONTROL_MODEL else model


def check_same_models(
    criterion: Mapping[str, float], instruments: Mapping[str, Mapping[str, Mapping[str, float]]]
) -> list[str]:
    """Every rung the criterion names must be scored by every instrument; extra models are reported, not used.

    A probe result usually carries more checkpoints than one analysis needs — E2627 scores a dozen further arms
    beside the curve — and the criterion table, not the result file, defines the population. Dropping the extras
    is stated rather than silent; missing a rung is a refusal, because a pair with no score in it would quietly
    shrink the family Holm corrects over.
    """
    extras: set[str] = set()
    for instrument, table in instruments.items():
        missing = sorted(set(criterion) - set(table))
        if missing:
            message = (
                f"{instrument} is missing {missing} of the curve {sorted(criterion)}; a rung without a "
                "score would silently shrink the family of pairs being corrected over"
            )
            raise AnalysisError(message)
        extras |= set(table) - set(criterion)
    if RUNG_ZERO not in criterion:
        message = "the untrained floor is missing from the criterion table"
        raise AnalysisError(message)
    return sorted(extras)


def read(
    *,
    report: Mapping[str, Any],
    scored: Mapping[str, Mapping[str, Any]],
    pairs_result: Mapping[str, Any],
    continuations_result: Mapping[str, Any],
    usable_phenomena: Sequence[str],
    materiality: float,
    endpoint: str,
    context_result: Mapping[str, Any] | None = None,
    criterion_from: Path | None = None,
) -> dict[str, Any]:
    """Assemble the tables and hand them to the library that decides the verdict."""
    from chakaso.training.probe_sensitivity import compare_instruments

    check_endpoint_matches_the_report(scored, report, endpoint=endpoint)
    table = criterion_table(scored)
    usable = pair_outcomes(pairs_result, phenomena=usable_phenomena)
    all_pairs = pair_outcomes(pairs_result, phenomena=None)
    instruments: dict[str, dict[str, dict[str, float]]] = {
        "minimal_pairs_usable": usable,
        "minimal_pairs_all_800": all_pairs,
        "continuation_pmi_easy": continuation_outcomes(continuations_result, reading="pmi_easy"),
        "continuation_pmi_hard": continuation_outcomes(continuations_result, reading="pmi_hard"),
    }
    if context_result is not None:
        instruments.update(context_outcomes(context_result))
    extras = check_same_models(table, instruments)
    criterion = {
        name: float(block["whole_pack_bits_per_character"]) for name, block in table.items()
    }
    result = compare_instruments(criterion, instruments, materiality=materiality)
    result["criterion"] = table
    result["models_scored_but_not_analysed"] = extras
    if criterion_from is not None:
        result["criterion_reused_from"] = str(criterion_from.name)
    result["materiality_declared_in"] = "experiments/E2626/record.md"
    result["usable_phenomena"] = list(usable_phenomena)
    result["endpoint"] = {
        "rung": endpoint,
        "bits_per_character": criterion[endpoint],
        "o1": table[endpoint]["o1_equal_weight_mean"],
    }
    return result


def rescored_from(earlier: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the scored table a previous analysis stored, so a second instrument needs no re-scoring.

    The stored table carries the whole pack, every partition and both denominators, which is what lets the
    endpoint guard still run against the report: reusing a criterion is not the same as trusting a bare float.
    """
    rebuilt: dict[str, Any] = {}
    for name, values in earlier["criterion"].items():
        for field in ("positions_scored", "characters_scored", "test_partitions"):
            if field not in values:
                message = (
                    f"{name} in {earlier.get('criterion_reused_from', 'the given file')!r} carries no "
                    f"{field!r}, so this table was written before the denominators were stored and the "
                    "endpoint guard could not be re-run from it; analyse the curve again without "
                    "--criterion-from"
                )
                raise AnalysisError(message)
        rebuilt[name] = {
            "test": {
                "bits_per_character": float(values["whole_pack_bits_per_character"]),
                "positions_scored": int(values["positions_scored"]),
                "characters_scored": int(values["characters_scored"]),
            },
            "test_partitions": {
                prefix: {"metrics": {"bits_per_character": float(row["bits_per_character"])}}
                for prefix, row in values["test_partitions"].items()
            },
        }
    return rebuilt


def summarise(result: Mapping[str, Any]) -> str:
    """The reader-facing lines: one verdict per instrument, with the two gaps that define it."""
    lines = [
        f"{len(result['models'])} models, materiality {result['materiality']} bits per character, "
        f"{len(result['pairs'])} pairs tested"
    ]
    for name, verdict in sorted(result["instruments"].items()):
        resolved = verdict["smallest_separated_gap"]
        missed = verdict["largest_unseparated_material_gap"]
        lines.append(
            f"{name}: {verdict['verdict']} rho {verdict['rho']:+.3f}, "
            f"{verdict['material_pairs_separated']}/{verdict['material_pairs']} material pairs separated"
            + (f", smallest gap seen {resolved:.4f}" if resolved is not None else ", no gap seen")
            + (f", largest missed {missed:.4f}" if missed is not None else "")
        )
        if verdict["contradicted_pairs"]:
            lines.append(f"    contradicts the criterion on {verdict['contradicted_pairs']}")
    return "\n".join(lines)


def _build_plan(args: argparse.Namespace) -> Any:
    from chakaso.training.protocol import load_plan

    return load_plan(
        args.protocol,
        root=Path.cwd(),
        work_dir=args.work_dir,
        fixed={
            "split_seed": args.split_seed,
            "expected_corpus_fingerprint": args.expected_corpus_fingerprint,
            "corpus_snapshot": args.corpus_snapshot,
        },
    )


def report_experiment(path: Path) -> str:
    """The experiment identifier a report was written under, for naming the artifact this script builds."""
    return str(_load(path)["experiment"])


def _score_rungs(args: argparse.Namespace, names: Mapping[str, str]) -> dict[str, Any]:
    """Score every rung, and the untrained floor, over the pack the plan prepares."""
    from chakaso.training.nn_checkpoint import ModelArtifact, load_artifact, save_artifact
    from chakaso.training.nn_evaluation import untrained_model_like
    from chakaso.training.subword_experiment import score_artifacts

    plan = _build_plan(args)
    endpoint_path = args.work_dir / names[args.endpoint]
    artifacts = {
        name: args.work_dir / file_name
        for name, file_name in names.items()
        if name != RUNG_ZERO and file_name != "untrained"
    }
    if RUNG_ZERO in names and names[RUNG_ZERO] == "untrained":
        source = load_artifact(endpoint_path)
        control = untrained_model_like(source, plan.seed)
        control_path = (
            args.work_dir
            / f"{report_experiment(args.report)}-rung0-untrained-{plan.model_family}.pt"
        )
        if not control_path.exists():
            save_artifact(
                ModelArtifact(
                    config=source.config,
                    state=control.state_dict(),
                    vocabulary=source.vocabulary,
                    corpus_fingerprint=source.corpus_fingerprint,
                    run_metadata={"kind": "usable", "curve_rung_step": 0, "untrained": True},
                    step=0,
                    environment=source.environment,
                    tokenizer_min_frequency=source.vocabulary.min_frequency,
                ),
                control_path,
            )
        artifacts[RUNG_ZERO] = control_path
    return score_artifacts(plan, artifacts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--corpus-snapshot", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--expected-corpus-fingerprint", required=True)
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
    parser.add_argument(
        "--criterion-from",
        type=Path,
        default=None,
        help="an earlier run of this script: reuse its criterion table instead of scoring every rung again",
    )
    parser.add_argument("--materiality", type=float, default=0.02)
    parser.add_argument(
        "--endpoint", default=None, help="rung to check against the report (default: the last)"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = _load(args.report)
    names = rung_names(report)
    if args.endpoint is None:
        args.endpoint = max(names, key=lambda key: int(key.removeprefix("step")))
    try:
        if args.criterion_from is not None:
            # A second instrument on one curve needs the same rung loss numbers, and recomputing them costs a
            # corpus preparation and one evaluation per rung. The table is carried in the first analysis's
            # output beside the checksum of every artifact it was measured from, so reusing it is stated in
            # the result rather than implicit: the endpoint guard below then reads the stored blocks.
            scored = rescored_from(_load(args.criterion_from))
            print(
                "note: the criterion table was reused from "
                f"{args.criterion_from.name}; no rung was re-scored here",
                file=sys.stderr,
            )
        else:
            scored = _score_rungs(args, names)
        result = read(
            report=report,
            scored=scored,
            pairs_result=_load(args.pairs),
            continuations_result=_load(args.continuations),
            usable_phenomena=args.usable_phenomena.split(","),
            materiality=args.materiality,
            endpoint=args.endpoint,
            context_result=_load(args.context) if args.context else None,
            criterion_from=args.criterion_from,
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
    print(summarise(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
