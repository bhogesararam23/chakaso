"""Score checkpoints (and untrained controls) on the same-region continuation probe.

The construction is :mod:`chakaso.training.context_continuations`: for each held-out passage the model must pick
the true next span over the same document's text immediately before the passage and immediately after the
continuation, and over that span's own identifiers permuted and reversed. Topic, register and hand are held
fixed by the distractors, so what remains to detect is that *this* text follows *this* passage.

Everything it shares with the v1 probe is shared by import rather than by copy — the split rebuild, its
fractions, the register names — because two scripts that each reconstructed the held-out side could drift apart
and the two probes would then be measuring different documents while their records say they are comparable.

Usage:
    python scripts/evaluate_context_continuations.py --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --vocabulary experiments/E2612/vocabulary-434debe82f949347.json --expected-split-fingerprint adf650435c3a9068 \\
        --registers prose-:100,science-:55,technology-:40,mathematics-:20 --max-per-work 5 \\
        --checkpoint step200=data/experiments/E2626/seed-7/E2626-step200-decoder.pt \\
        --control --control-seeds 11,13,17,19,23,29 --out experiments/E2627/context.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The split rebuild and the register names, imported so the two probes cannot drift apart.
from evaluate_continuations import CONTROL_SEED, test_side

from chakaso.training.context_continuations import (
    ContextItem,
    build_context_items,
    context_items_digest,
    score_context_items,
)
from chakaso.training.data import TextUnit
from chakaso.training.errors import CorpusError, TrainingError
from chakaso.training.nn_checkpoint import ModelArtifact, load_artifact
from chakaso.training.nn_evaluation import (
    continuation_log_probabilities,
    untrained_model_like,
)
from chakaso.training.subword import SubwordVocabulary

DEFAULT_REGISTERS = "prose-:100,science-:55,technology-:40,mathematics-:20"


def score_model(model: Any, items: Sequence[ContextItem]) -> dict[str, Any]:
    """PMI and raw-conditional readings for every kind, each distinct (prefix, span) request scored once."""
    requests: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for item in items:
        candidates = [item.true_continuation] + [span for _, _, span in item.distractors]
        for continuation in candidates:
            for prefix in (item.prefix, ()):
                key = (prefix, continuation)
                if key not in seen:
                    seen.add(key)
                    requests.append(key)
    values = dict(zip(requests, continuation_log_probabilities(model, requests), strict=True))
    return score_context_items(
        items, lambda prefix, continuation: values[(prefix, continuation)]
    ).as_dict()


def build(
    units: Sequence[TextUnit],
    vocabulary: SubwordVocabulary,
    registers: str,
    args: argparse.Namespace,
) -> tuple[ContextItem, ...]:
    """One same-region item set per register entry, concatenated in the order the caller declared."""
    items: list[ContextItem] = []
    for entry in (part for part in registers.split(",") if part):
        register, _, count = entry.partition(":")
        items.extend(
            build_context_items(
                units,
                vocabulary,
                prefix_length=args.prefix_length,
                continuation_length=args.continuation_length,
                per_register=int(count) if count else args.per_register,
                registers=(register,),
                max_per_work=args.max_per_work,
            )
        )
    return tuple(items)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--expected-split-fingerprint")
    parser.add_argument(
        "--registers",
        default=DEFAULT_REGISTERS,
        help="comma-separated work-group prefixes, each optionally PREFIX:COUNT to override --per-register",
    )
    parser.add_argument("--max-per-work", type=int, default=2)
    parser.add_argument("--per-register", type=int, default=100)
    parser.add_argument("--prefix-length", type=int, default=64)
    parser.add_argument("--continuation-length", type=int, default=24)
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
        items = build(units, vocabulary, args.registers, args)
        result: dict[str, Any] = {
            "items": {
                "count": len(items),
                "digest": context_items_digest(items),
                "vocabulary_fingerprint": vocabulary.fingerprint,
                "prefix_length": args.prefix_length,
                "continuation_length": args.continuation_length,
                "kinds": [kind for kind, _, _ in (items[0].distractors if items else [])],
                "per_register": {
                    entry.partition(":")[0]: int(entry.partition(":")[2] or args.per_register)
                    for entry in args.registers.split(",")
                    if entry
                },
                "max_per_work": args.max_per_work,
                "split_fingerprint": args.expected_split_fingerprint,
                "test_documents": len(units),
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
                "step": artifact.step,
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
    print(
        f"{len(items)} items, digest {result['items']['digest'][:16]}…, "
        f"{len(result['models'])} models"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
