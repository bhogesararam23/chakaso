"""Score trained checkpoints (and an untrained control) on a minimal-pair probe.

For each ``NAME=checkpoint.pt`` the tool loads the artifact (checksum-verified), scores both sentences of every pair
with the model's own tokenizer, and reports accuracy overall and per phenomenon with Wilson intervals, whether chance
lies inside the interval, the length-only reading, and the number of pairs flagged by the contamination screen against a
corpus snapshot. ``--control`` adds an untrained model of the first checkpoint's architecture (seed 7) as the instrument's
floor: a probe that cannot tell a trained model from a random one is not a measurement of what the model learned.

The result carries a cluster listing (K-035) whose unit is the **phenomenon**, not a held-out work: these pairs are
self-authored, so no book stands behind them and the largest shared cause a pair has is the template its phenomenon
follows. A clustered reading of this probe is a statement about templates, which is why the unit is named in the
artifact rather than inferred by a reader.

Usage:
    python scripts/evaluate_minimal_pairs.py --pairs configs/probes/grammatical-minimal-pairs-v1.json \\
        --checkpoint E2617=data/experiments/E2617/seed-7/E2617-decoder.pt --control \\
        --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --out experiments/E2621/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


from chakaso.training.errors import TrainingError
from chakaso.training.item_clusters import PHENOMENON_CLUSTER_UNIT, ClusterListing
from chakaso.training.minimal_pairs import PairSet, contaminated_pairs, load_pair_set, score_pairs
from chakaso.training.nn_checkpoint import ModelArtifact, load_artifact
from chakaso.training.nn_evaluation import sequence_log_probabilities, untrained_model_like

CONTROL_SEED = 7


def score_artifact(
    artifact: ModelArtifact, pair_set: PairSet, *, model: Any | None = None
) -> dict[str, Any]:
    """The probe's result for one artifact, with its tokenizer and (unless ``model`` is given) its own weights."""
    model = artifact.to_model() if model is None else model
    cache: dict[str, float] = {}
    sentences = sorted({s for pair in pair_set.pairs for s in (pair.good, pair.bad)})
    for sentence, value in zip(
        sentences, sequence_log_probabilities(model, artifact.vocabulary, sentences), strict=True
    ):
        cache[sentence] = value
    vocabulary = artifact.vocabulary
    return score_pairs(
        pair_set.pairs,
        cache.__getitem__,
        token_count=lambda sentence: len(vocabulary.encode(sentence)),
    ).as_dict()


def untrained_control(artifact: ModelArtifact, seed: int = CONTROL_SEED) -> Any:
    """A randomly initialised model of the artifact's architecture, seeded so the control is reproducible."""
    return untrained_model_like(artifact, seed)


def _snapshot_texts(path: Path) -> list[str]:
    return [
        document["text"] for document in json.loads(path.read_text(encoding="utf-8"))["documents"]
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=path.pt")
    parser.add_argument(
        "--control",
        action="store_true",
        help="also score an untrained model of the first architecture",
    )
    parser.add_argument(
        "--control-seeds",
        default="",
        help="comma-separated extra seeds: one more untrained control per seed, to see how much a single random "
        "initialisation's per-phenomenon bias varies",
    )
    parser.add_argument(
        "--snapshot", type=Path, help="a corpus snapshot for the contamination screen"
    )
    parser.add_argument("--ngram", type=int, default=5)
    parser.add_argument(
        "--expected-pair-set-digest",
        help="the digest a record declared before it scored; the run refuses a pair file that no longer "
        "matches it, before any checkpoint is loaded",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        pair_set = load_pair_set(args.pairs)
        if (
            args.expected_pair_set_digest is not None
            and pair_set.digest != args.expected_pair_set_digest
        ):
            message = (
                f"{args.pairs} loads as digest {pair_set.digest}, not the "
                f"{args.expected_pair_set_digest} this run was told to expect: the record would be scoring a "
                "pair set other than the one it declared"
            )
            raise ValueError(message)
        clusters = ClusterListing(
            unit=PHENOMENON_CLUSTER_UNIT,
            rows=tuple((pair.identifier, pair.phenomenon) for pair in pair_set.pairs),
        )
        result: dict[str, Any] = {
            "pair_set": {
                "name": pair_set.name,
                "version": pair_set.version,
                "digest": pair_set.digest,
                "clusters": clusters.as_dict(),
            },
            "models": {},
        }
        first: ModelArtifact | None = None
        for item in args.checkpoint:
            name, _, location = item.partition("=")
            artifact = load_artifact(Path(location))
            first = first or artifact
            result["models"][name] = {
                "checkpoint_checksum": artifact.checksum(),
                "tokenizer_fingerprint": artifact.vocabulary.fingerprint,
                "environment": dict(artifact.environment),
                "result": score_artifact(artifact, pair_set),
            }
        if args.control and first is not None:
            result["models"]["untrained-control"] = {
                "tokenizer_fingerprint": first.vocabulary.fingerprint,
                "seed": CONTROL_SEED,
                "result": score_artifact(first, pair_set, model=untrained_control(first)),
            }
        if first is not None and args.control_seeds.strip():
            for seed in (int(item) for item in args.control_seeds.split(",")):
                result["models"][f"untrained-control-seed-{seed}"] = {
                    "tokenizer_fingerprint": first.vocabulary.fingerprint,
                    "seed": seed,
                    "result": score_artifact(first, pair_set, model=untrained_control(first, seed)),
                }
        if args.snapshot is not None:
            flagged = contaminated_pairs(
                pair_set.pairs, _snapshot_texts(args.snapshot), n=args.ngram
            )
            result["contamination"] = {
                "ngram_words": args.ngram,
                "corpus_documents_screened": len(_snapshot_texts(args.snapshot)),
                "pairs_flagged": len(flagged),
                "fraction_flagged": len(flagged) / len(pair_set.pairs),
                "flagged_ids": list(flagged),
            }
    except (OSError, KeyError, ValueError, TrainingError) as error:
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
