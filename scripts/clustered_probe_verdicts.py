"""Ask a probe's committed per-item verdicts the question item-level McNemar cannot: separated by *work*.

E2629 and E2634 judged every arm pair with exact McNemar over the paired rows, and both records said in their own
limits section that the p-values should be read as optimistic, because the items of one held-out book are not
independent observations. ``chakaso.training.paired_verdicts`` is the reading that takes the clustering into account,
and this script hands it the rows a record already committed, so the two readings sit beside each other on the same
items instead of being argued about.

Nothing is re-scored. The verdicts come from a probe result or a compacted extract (K-032's form is rebuilt
transparently), and the clusters come from the probe's own cluster block when it has one (K-035). A probe written
before that block has to have its item set re-cut from the pinned snapshot — the step E2634 ran by hand — and that
rebuild is **refused unless its digest equals the digest the probe file recorded**, because clusters describing a
different item set would be worse than no clusters: they would be a pairing a reader would believe.

Holm's step-down is applied to each reading across the pairs of one instrument, which is the correction the records
already used, so "separated" means the same thing in both columns.

Usage:
    python scripts/clustered_probe_verdicts.py --probe experiments/E2634/context-with-outcomes.json \\
        --reading pmi_permuted --clusters-from rebuild --builder context \\
        --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --vocabulary experiments/E2612/vocabulary-434debe82f949347.json --split-seed 7 \\
        --registers prose-:400,science-:170,technology-:120,mathematics-:55 --max-per-work 20 \\
        --model E2612-T --model E2612-P --model E2613-H --seed 7 --out experiments/E2635/results.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chakaso.training.item_clusters import (
    PHENOMENON_CLUSTER_UNIT,
    WORK_CLUSTER_UNIT,
    ClusterListing,
    ClusterListingError,
    cluster_block,
    keys_for_items,
    load_cluster_block,
)
from chakaso.training.outcome_vectors import is_compacted, restore_probe_result
from chakaso.training.paired_stats import PairedStatsError, holm_rejections
from chakaso.training.paired_verdicts import ClusteredVerdictError, clustered_paired_verdicts

__all__ = ["ClusterAnalysisError", "main", "model_rows", "read"]


class ClusterAnalysisError(RuntimeError):
    """The probe file and the cluster source do not describe one item set."""


def load_probe(path: Path) -> dict[str, Any]:
    """A probe result or a compacted extract of one, with the per-item arrays rebuilt when they are stored compacted."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        message = f"{path}: {error}"
        raise ClusterAnalysisError(message) from error
    return restore_probe_result(document) if is_compacted(document) else document


def model_rows(
    probe: Mapping[str, Any], reading: str, model: str, *, pair_probe: bool
) -> dict[str, float]:
    """One model's per-item verdicts for one reading, as ``{item identifier: verdict}``.

    The pair probe stores one flat array of ``(identifier, phenomenon, verdict)``; the continuation probes store a
    mapping of reading name to ``(identifier, verdict)`` arrays. A missing reading or model is a refusal that names
    both, because an empty vector would read as "these arms agreed on every item".
    """
    models = probe.get("models")
    if not isinstance(models, Mapping) or model not in models:
        message = f"the probe file holds no model {model!r}; its models are {sorted(models or {})}"
        raise ClusterAnalysisError(message)
    outcomes = models[model].get("result", {}).get("outcomes")
    if pair_probe:
        if not isinstance(outcomes, list) or not outcomes:
            message = (
                f"{model}: no flat per-item outcome array, so this is not a minimal-pair result"
            )
            raise ClusterAnalysisError(message)
        return {str(row[0]): float(row[2]) for row in outcomes}
    if not isinstance(outcomes, Mapping) or reading not in outcomes:
        named = sorted(outcomes) if isinstance(outcomes, Mapping) else "nothing"
        message = f"{model}: no per-item outcomes for reading {reading!r}; it carries {named}"
        raise ClusterAnalysisError(message)
    return {str(row[0]): float(row[1]) for row in outcomes[reading]}


def _listing_from_probe(probe: Mapping[str, Any]) -> ClusterListing:
    """The cluster block a probe result carries itself (K-035), refused when the file predates it.

    E2634's committed extracts were written before the block existed, and the honest answer for them is to say so and
    name the two other ways to get the pairing, rather than guess one.
    """
    for key in ("items", "pair_set"):
        block = probe.get(key)
        if isinstance(block, Mapping) and block.get("clusters") is not None:
            return load_cluster_block(block["clusters"])
    message = (
        "this probe file carries no cluster block, so its items cannot be clustered from the artifact alone; "
        "re-cut the item set with --clusters-from rebuild (digest-verified against the probe's own item digest) "
        "or, for an authored pair probe, take the phenomenon column with --clusters-from rows"
    )
    raise ClusterAnalysisError(message)


def _listing_from_rows(probe: Mapping[str, Any]) -> ClusterListing:
    """The authored pair probe: its outcome rows already carry the phenomenon each pair belongs to."""
    first = next(iter(probe["models"].values()), None)
    outcomes = (first or {}).get("result", {}).get("outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        message = "no flat per-item outcome array to read a phenomenon column from"
        raise ClusterAnalysisError(message)
    return ClusterListing(
        unit=PHENOMENON_CLUSTER_UNIT,
        rows=tuple((str(row[0]), str(row[1])) for row in outcomes),
    )


def quotas(registers: str) -> dict[str, int]:
    """``{register: items}`` for a ``prose-:400,science-:170`` list, refusing an entry without its quota.

    A rebuild has to ask for exactly the quotas the scored run asked for, so an uncounted entry is a refusal rather
    than a default.
    """
    wanted: dict[str, int] = {}
    for entry in (part for part in registers.split(",") if part):
        register, _, count = entry.partition(":")
        if not count:
            message = f"--registers entry {entry!r} names no quota; a rebuild must ask for the one that was asked"
            raise ClusterAnalysisError(message)
        wanted[register] = int(count)
    if not wanted:
        message = "--registers named no register to rebuild"
        raise ClusterAnalysisError(message)
    return wanted


def _listing_from_rebuild(args: argparse.Namespace, probe: Mapping[str, Any]) -> ClusterListing:
    """Re-cut the item set from the pinned snapshot, and check by digest that it is the set that was scored.

    Both builders are deterministic, so the same snapshot, vocabulary, lengths and quotas rebuild the same items. The
    digest equality is what turns that into evidence: without it the clusters could pair a *different* item set, and
    every number downstream would inherit the mistake quietly.
    """
    items_block = probe.get("items")
    if not isinstance(items_block, Mapping) or not items_block.get("digest"):
        message = "the probe file records no item digest, so a rebuild has nothing to check itself against"
        raise ClusterAnalysisError(message)
    declared = str(items_block["digest"])
    units, vocabulary = _test_side(args)
    wanted = quotas(args.registers)
    built: list[Any] = []
    for register, per_register in wanted.items():
        built.extend(
            _build(
                args.builder,
                units,
                vocabulary,
                register=register,
                per_register=per_register,
                args=args,
            )
        )
    if not built:
        message = "the rebuild produced no items, so there is nothing to cluster"
        raise ClusterAnalysisError(message)
    digest = _digest(args.builder, built)
    if digest != declared:
        message = (
            f"the rebuilt item set digests to {digest}, but the probe file records {declared}: these are not "
            "the items that were scored, so their works would be attributed to the wrong verdicts"
        )
        raise ClusterAnalysisError(message)
    return load_cluster_block(cluster_block(built, unit=WORK_CLUSTER_UNIT))


def _test_side(args: argparse.Namespace) -> tuple[Any, Any]:
    """The held-out documents and the vocabulary the items were cut with, loaded only when a rebuild was asked for."""
    if args.snapshot is None or args.vocabulary is None:
        message = "--clusters-from rebuild needs --snapshot and --vocabulary"
        raise ClusterAnalysisError(message)
    from evaluate_continuations import test_side

    from chakaso.training.subword import SubwordVocabulary

    vocabulary = SubwordVocabulary.load(args.vocabulary)
    units = test_side(
        args.snapshot,
        split_seed=args.split_seed,
        expected_fingerprint=args.expected_split_fingerprint,
    )
    return units, vocabulary


def _build(
    builder: str,
    units: Any,
    vocabulary: Any,
    *,
    register: str,
    per_register: int,
    args: argparse.Namespace,
) -> Sequence[Any]:
    """One register's items, from the builder the probe file's construction names."""
    if builder == "context":
        from chakaso.training.context_continuations import build_context_items

        return build_context_items(
            units,
            vocabulary,
            prefix_length=args.prefix_length,
            continuation_length=args.continuation_length,
            per_register=per_register,
            registers=(register,),
            max_per_work=args.max_per_work,
        )
    from chakaso.training.continuation_items import build_items

    return build_items(
        units,
        vocabulary,
        prefix_length=args.prefix_length,
        continuation_length=args.continuation_length,
        hard_gap=args.hard_gap,
        per_register=per_register,
        registers=(register,),
        max_per_work=args.max_per_work,
    )


def _digest(builder: str, items: Sequence[Any]) -> str:
    if builder == "context":
        from chakaso.training.context_continuations import context_items_digest

        return context_items_digest(items)
    from chakaso.training.continuation_items import items_digest

    return items_digest(items)


def read(
    probe: Mapping[str, Any],
    *,
    readings: Sequence[str],
    models: Sequence[str],
    listing: ClusterListing,
    seed: int,
    resamples: int,
    probability: float,
    alpha: float,
    pair_probe: bool,
) -> dict[str, Any]:
    """Every pair of models under both readings, with Holm's correction applied to each reading across the family."""
    if len(models) < 2:
        message = f"a pair needs two models, got {models}"
        raise ClusterAnalysisError(message)
    unknown = sorted(set(models) - set(probe.get("models", {})))
    if unknown:
        message = (
            f"the probe file holds no {unknown}; its models are {sorted(probe.get('models', {}))}"
        )
        raise ClusterAnalysisError(message)
    result: dict[str, Any] = {
        "cluster_unit": listing.unit,
        "cluster_digest": listing.digest,
        "clusters": len(listing.clusters),
        "items": len(listing.rows),
        "models": list(models),
        "seed": seed,
        "resamples": resamples,
        "probability": probability,
        "alpha": alpha,
        "readings": {},
    }
    for reading in readings:
        tables = {
            model: model_rows(probe, reading, model, pair_probe=pair_probe) for model in models
        }
        order = sorted(tables[models[0]])
        for model, table in tables.items():
            if sorted(table) != order:
                message = (
                    f"{model} names {len(table)} items and {models[0]} names {len(order)}: two arms whose "
                    "item sets differ are not a paired comparison"
                )
                raise ClusterAnalysisError(message)
        keys = keys_for_items(listing, order)
        rows: list[dict[str, Any]] = []
        for first, second in itertools.combinations(models, 2):
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "clustered": clustered_paired_verdicts(
                        [tables[first][identifier] for identifier in order],
                        [tables[second][identifier] for identifier in order],
                        keys,
                        seed=seed,
                        resamples=resamples,
                        probability=probability,
                    ),
                }
            )
        clustered_rejected = holm_rejections(
            [float(row["clustered"]["cluster_sign_flip"]["p_value"]) for row in rows], alpha=alpha
        )
        item_rejected = holm_rejections(
            [float(row["clustered"]["item_level_mcnemar"]["p_value"]) for row in rows], alpha=alpha
        )
        for row, keep_clustered, keep_item in zip(
            rows, clustered_rejected, item_rejected, strict=True
        ):
            row["separated_clustered"] = bool(keep_clustered)
            row["separated_item_level"] = bool(keep_item)
        result["readings"][reading] = {
            "pairs": rows,
            "separated_clustered": sum(1 for row in rows if row["separated_clustered"]),
            "separated_item_level": sum(1 for row in rows if row["separated_item_level"]),
            "holm_thresholds": sorted(
                (alpha / (len(rows) - index) for index in range(len(rows))), reverse=True
            ),
        }
    return result


def _effect_text(effect: Any) -> str:
    """The design effect as one readable figure, naming the case where there is no spread to measure."""
    if effect is None:
        return "none (every item gives the same difference, so the item-level variance is zero to inflate)"
    return f"{float(effect['ratio']):.2f} (about {float(effect['effective_items']):.0f} independent items)"


def summarise(result: Mapping[str, Any]) -> str:
    """The two readings side by side, one line per pair, so the command that ran is also readable on a terminal."""
    lines = [
        f"{result['items']} items over {result['clusters']} clusters ({result['cluster_unit']}), "
        f"Holm at alpha {result['alpha']}"
    ]
    for reading, block in result["readings"].items():
        lines.append(
            f"{reading}: {block['separated_item_level']} separated item-level, "
            f"{block['separated_clustered']} separated clustered"
        )
        for row in block["pairs"]:
            stats = row["clustered"]
            spread = stats["signal_spread"]
            lines.append(
                f"  {row['first']} / {row['second']}: difference "
                f"{float(stats['mean_difference']):+.5f}, item-level p "
                f"{float(stats['item_level_mcnemar']['p_value']):.5g} "
                f"({'sep' if row['separated_item_level'] else 'no'}), clustered p "
                f"{float(stats['cluster_sign_flip']['p_value']):.5g} "
                f"({'sep' if row['separated_clustered'] else 'no'}), design effect "
                f"{_effect_text(stats['design_effect'])}, "
                f"{spread['works_with_discordance']} works disagree"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--reading", action="append", required=True, help="one reading per flag")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="a model in the probe file; at least two, every pair tested",
    )
    parser.add_argument(
        "--clusters-from",
        choices=("probe", "rebuild", "rows"),
        default="probe",
        help="where the item-to-work pairing comes from: the probe's own block (the default), a digest-verified "
        "rebuild of the item set, or the phenomenon column of an authored pair probe",
    )
    parser.add_argument("--builder", choices=("context", "continuation"), default="context")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--vocabulary", type=Path)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--expected-split-fingerprint")
    parser.add_argument("--registers", default="")
    parser.add_argument("--max-per-work", type=int, default=20)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--continuation-length", type=int, default=24)
    parser.add_argument("--hard-gap", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--probability", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--pair-probe",
        action="store_true",
        help="the file is a minimal-pair result, whose outcomes are one flat (identifier, phenomenon, verdict) array",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        probe = load_probe(args.probe)
        if args.clusters_from == "probe":
            listing = _listing_from_probe(probe)
        elif args.clusters_from == "rows":
            listing = _listing_from_rows(probe)
        else:
            listing = _listing_from_rebuild(args, probe)
        result = read(
            probe,
            readings=args.reading,
            models=args.model,
            listing=listing,
            seed=args.seed,
            resamples=args.resamples,
            probability=args.probability,
            alpha=args.alpha,
            pair_probe=args.pair_probe,
        )
    except (
        ClusterAnalysisError,
        ClusterListingError,
        ClusteredVerdictError,
        PairedStatsError,
        OSError,
        KeyError,
        ValueError,
    ) as error:
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


if __name__ == "__main__":
    raise SystemExit(main())
