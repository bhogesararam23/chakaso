"""Score checkpoints (and untrained controls) on the held-out continuation-discrimination probe.

The probe's items are cut from the **test side** of the whole-work split of a pinned corpus snapshot (the split is rebuilt here and its
fingerprint checked against the one the experiments used), tokenized with the vocabulary the checkpoints share. For each ``NAME=checkpoint.pt``
the tool scores every candidate continuation with and without its passage, and reports pointwise-mutual-information accuracy against an easy
(other work, same register) and a hard (same document, far away) distractor, with the raw-conditional and prefix-blind readings beside it.
``--control``/``--control-seeds`` add untrained models of the first architecture as the floor.

Usage:
    python scripts/evaluate_continuations.py --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --vocabulary experiments/E2612/vocabulary-434debe82f949347.json --expected-split-fingerprint adf650435c3a9068 \\
        --checkpoint E2617=data/experiments/E2617/seed-7/E2617-decoder.pt --control --control-seeds 11,13,17,19,23,29 \\
        --out experiments/E2623/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.training.continuation_items import (
    ContinuationItem,
    build_items,
    items_digest,
    score_items,
)
from chakaso.training.data import TextUnit, split_text_units
from chakaso.training.errors import CorpusError, TrainingError
from chakaso.training.item_clusters import WORK_CLUSTER_UNIT, cluster_block
from chakaso.training.manifest import read_snapshot
from chakaso.training.nn_checkpoint import ModelArtifact, load_artifact
from chakaso.training.nn_evaluation import continuation_log_probabilities, untrained_model_like
from chakaso.training.subword import SubwordVocabulary

REGISTERS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
CONTROL_SEED = 7
#: The split the experiments of this series used (E2612-E2622): validation and test fractions of 0.15 and split seed 7, by whole work.
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15


def test_side(
    snapshot: Path, *, split_seed: int, expected_fingerprint: str | None
) -> tuple[TextUnit, ...]:
    """The held-out documents of the corpus snapshot, rebuilt exactly as the experiments split it."""
    manifest = read_snapshot(snapshot).manifest
    split = split_text_units(
        manifest.text_units(),
        validation_fraction=VALIDATION_FRACTION,
        test_fraction=TEST_FRACTION,
        seed=split_seed,
        whole_works=True,
    )
    if expected_fingerprint is not None and split.fingerprint != expected_fingerprint:
        message = (
            f"the rebuilt split has fingerprint {split.fingerprint!r}, not {expected_fingerprint!r}: "
            "these are not the held-out documents the experiments used"
        )
        raise CorpusError(message)
    return split.test


def score_model(model: Any, items: Sequence[ContinuationItem]) -> dict[str, Any]:
    """The probe's six readings for one model, scoring every distinct (prefix, continuation) request once."""
    requests: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for item in items:
        for continuation in (item.true_continuation, item.easy_distractor, item.hard_distractor):
            for prefix in (item.prefix, ()):
                key = (prefix, continuation)
                if key not in seen:
                    seen.add(key)
                    requests.append(key)
    values = dict(zip(requests, continuation_log_probabilities(model, requests), strict=True))
    return score_items(items, lambda prefix, continuation: values[(prefix, continuation)]).as_dict()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--expected-split-fingerprint")
    parser.add_argument(
        "--expected-items-digest",
        help="the digest a record declared before it scored; the run refuses if the items it built differ, "
        "before any checkpoint is loaded",
    )
    parser.add_argument(
        "--registers",
        default=",".join(REGISTERS),
        help="comma-separated work-group prefixes, each optionally PREFIX:COUNT to override --per-register "
        "(a register with few held-out works cannot supply as many independent passages)",
    )
    parser.add_argument("--max-per-work", type=int, default=2)
    parser.add_argument("--per-register", type=int, default=100)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--continuation-length", type=int, default=24)
    parser.add_argument("--hard-gap", type=int, default=1024)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=path.pt")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--control-seeds", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        vocabulary = SubwordVocabulary.load(args.vocabulary)
        units = test_side(
            args.snapshot,
            split_seed=args.split_seed,
            expected_fingerprint=args.expected_split_fingerprint,
        )
        items = []
        for entry in (item for item in args.registers.split(",") if item):
            register, _, count = entry.partition(":")
            items.extend(
                build_items(
                    units,
                    vocabulary,
                    prefix_length=args.prefix_length,
                    continuation_length=args.continuation_length,
                    hard_gap=args.hard_gap,
                    per_register=int(count) if count else args.per_register,
                    registers=(register,),
                    max_per_work=args.max_per_work,
                )
            )
        items = tuple(items)
        digest = items_digest(items)
        if args.expected_items_digest is not None and digest != args.expected_items_digest:
            message = (
                f"the {len(items)} items these quotas build digest to {digest}, not the "
                f"{args.expected_items_digest} this run was told to expect: the record would be scoring an "
                "item set other than the one it declared"
            )
            raise ValueError(message)
        result: dict[str, Any] = {
            "items": {
                "count": len(items),
                "digest": digest,
                "vocabulary_fingerprint": vocabulary.fingerprint,
                "prefix_length": args.prefix_length,
                "continuation_length": args.continuation_length,
                "hard_gap": args.hard_gap,
                "per_register": {
                    entry.partition(":")[0]: int(entry.partition(":")[2] or args.per_register)
                    for entry in args.registers.split(",")
                    if entry
                },
                "max_per_work": args.max_per_work,
                "split_fingerprint": args.expected_split_fingerprint,
                "test_documents": len(units),
                "clusters": cluster_block(items, unit=WORK_CLUSTER_UNIT),
            },
            "models": {},
        }
        first: ModelArtifact | None = None
        for entry in args.checkpoint:
            name, _, location = entry.partition("=")
            artifact = load_artifact(Path(location))
            if artifact.vocabulary.fingerprint != vocabulary.fingerprint:
                message = (
                    f"checkpoint {name!r} uses tokenizer {artifact.vocabulary.fingerprint!r}, not the "
                    f"{vocabulary.fingerprint!r} the items were cut with"
                )
                raise TrainingError(message)
            first = first or artifact
            result["models"][name] = {
                "checkpoint_checksum": artifact.checksum(),
                "environment": dict(artifact.environment),
                "result": score_model(artifact.to_model(), items),
            }
        if first is not None:
            seeds = ([CONTROL_SEED] if args.control else []) + [
                int(item) for item in args.control_seeds.split(",") if item.strip()
            ]
            for seed in seeds:
                label = (
                    "untrained-control"
                    if seed == CONTROL_SEED
                    else f"untrained-control-seed-{seed}"
                )
                result["models"][label] = {
                    "seed": seed,
                    "result": score_model(untrained_model_like(first, seed), items),
                }
    except (OSError, KeyError, ValueError, CorpusError, TrainingError) as error:
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
