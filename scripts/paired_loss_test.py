"""Test whether one trained arm is better than another on the held-out text itself, paired document by document.

The project has compared arms against a *band*: E2614 measured the seed-to-seed spread of the reference
configuration, and every arm difference since has been read against 0.02 bits per character. A band is a threshold,
not a test — it says a difference is bigger than the noise of re-initialising the model, and says nothing about how
well the difference on *these* held-out documents is known. E2629 made the gap concrete from the other side: its
probes could not separate the same arms at all, so the only evidence that an arm was better was a number beside a
threshold.

This script pairs the measurement. Both arms score the identical packed held-out pack; the loss is split into the
documents that produced it (:func:`chakaso.training.nn_evaluation.document_losses`), the per-document difference is
the observation, and the **work** is the unit the cluster bootstrap resamples — because a whole-work split holds out
books, and the dozen documents of one book are not a dozen independent observations.

The pack is rebuilt from the pinned snapshot and verified against the arms' own committed reports before any
statistic is computed: positions scored, documents packed and the whole-pack bits per character must reproduce what
the record published. A comparison whose denominator moved is a comparison of two code paths, not of two models.

Usage:
    python scripts/paired_loss_test.py --snapshot data/corpora/gutenberg-multiregister-v1/…-snapshot.json \\
        --vocabulary experiments/E2612/vocabulary-434debe82f949347.json --expected-split-fingerprint adf650435c3a9068 \\
        --arm "E2612-T=data/experiments/E2612/mixture/E2612-T-decoder.pt" \\
        --arm "E2612-P=data/experiments/E2612/prose/E2612-P-decoder.pt" --reference E2612-T \\
        --report "E2612-T=experiments/E2629/E2612-T-report.json" --pair E2612-T/E2612-P \\
        --out experiments/E2633/results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_continuations import test_side

from chakaso.training.errors import CorpusError, TrainingError
from chakaso.training.nn_checkpoint import load_artifact
from chakaso.training.nn_evaluation import document_losses
from chakaso.training.packing import pack_units, select_units_by_token_cap
from chakaso.training.paired_loss import paired_document_test
from chakaso.training.subword import SubwordVocabulary

#: A faithful rebuild of a published denominator is exact in positions and documents and agrees to the digits the
#: report prints: 1e-9 bits per character is far finer than any difference this test is asked about.
_METRIC_TOLERANCE = 1e-9


class AnalysisError(RuntimeError):
    """The artifacts handed to the test do not describe one held-out pack."""


def _entries(values: Sequence[str], *, option: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for value in values:
        name, _, path = value.partition("=")
        if not path:
            message = f"--{option} wants NAME=path, got {value!r}"
            raise AnalysisError(message)
        if name in out:
            message = f"--{option} names {name!r} twice"
            raise AnalysisError(message)
        out[name] = path
    return out


def build_pack(
    units: Sequence[Any],
    vocabulary: SubwordVocabulary,
    *,
    context_length: int,
    max_tokens: int | None,
    selection_vocabulary: SubwordVocabulary | None,
) -> tuple[Any, Sequence[Any]]:
    """The held-out pack, in the two shapes a selection vocabulary allows, with its document boundaries tracked.

    With a selection vocabulary the documents are the ones a pack capped under *that* vocabulary would hold and
    they are packed uncapped; without one the cap is applied while packing. Both branches are the experiment
    driver's, so an arm scored here is scored on the text its run scored.
    """
    if selection_vocabulary is not None and max_tokens is not None:
        packed_units = select_units_by_token_cap(units, selection_vocabulary, max_tokens)
        return (
            pack_units(
                packed_units, vocabulary, context_length=context_length, track_documents=True
            ),
            packed_units,
        )
    return (
        pack_units(
            units,
            vocabulary,
            context_length=context_length,
            track_documents=True,
            max_tokens=max_tokens,
        ),
        units,
    )


def bits_per_character(nats: float, characters: int) -> float:
    """The series' unit: total nats over total characters, in bits.

    One place computes it, because a per-work figure and a pooled figure that disagree on the arithmetic would be
    two measurements of the same documents.
    """
    if not characters:
        return math.inf
    return math.log2(math.e) * nats / characters


def score_arm(
    artifact_path: Path,
    batch: Any,
    units: Sequence[Any],
    *,
    batch_size: int,
    document_masking: bool,
) -> dict[str, Any]:
    """One arm's per-document losses, pooled into bits per character per work.

    ``document_masking`` must be how the arm trained: scoring with full attention a model that never saw across a
    document boundary measures the mismatch (``nn_evaluation`` says so at length). The caller states it; the
    checkpoint does not record it.
    """
    artifact = load_artifact(artifact_path)
    losses = document_losses(
        artifact.to_model(),
        batch.examples,
        artifact.vocabulary,
        units,
        batch_size=batch_size,
        document_masking=document_masking,
    )
    works: dict[str, dict[str, float]] = {}
    for loss in losses:
        key = loss.group or f"document-{loss.segment}"
        row = works.setdefault(key, {"nats": 0.0, "characters": 0, "documents": 0})
        row["nats"] = math.fsum([row["nats"], loss.nats])
        row["characters"] += loss.characters
        row["documents"] += 1
    for row in works.values():
        row["bits_per_character"] = bits_per_character(float(row["nats"]), int(row["characters"]))
    pooled = bits_per_character(
        math.fsum(loss.nats for loss in losses), sum(loss.characters for loss in losses)
    )
    return {
        "artifact": artifact_path.name,
        "checksum": artifact.checksum(),
        "step": artifact.step,
        "corpus_fingerprint": artifact.corpus_fingerprint,
        "tokenizer_fingerprint": artifact.vocabulary.fingerprint,
        "pooled_bits_per_character": pooled,
        "positions_scored": sum(loss.positions for loss in losses),
        "characters_scored": sum(loss.characters for loss in losses),
        "documents": len(losses),
        "works": works,
    }


def check_against_reports(scored: Mapping[str, Any], reports: Mapping[str, str]) -> list[str]:
    """Every recomputed denominator and pooled metric must equal the committed report's, or the pack is not the run's."""
    problems: list[str] = []
    for name, path in reports.items():
        if name not in scored:
            problems.append(f"{name}: a report was given with no arm of that name")
            continue
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        test = report["transformer"]["test"]
        measured = scored[name]
        checks = (
            ("positions scored", int(test["positions_scored"]), int(measured["positions_scored"])),
            (
                "characters scored",
                int(test["characters_scored"]),
                int(measured["characters_scored"]),
            ),
        )
        for label, published, recomputed in checks:
            if published != recomputed:
                problems.append(
                    f"{name}: {label} — the report says {published}, this pack gives {recomputed}"
                )
        published_metric = float(test["bits_per_character"])
        if abs(published_metric - float(measured["pooled_bits_per_character"])) > _METRIC_TOLERANCE:
            problems.append(
                f"{name}: pooled bits per character — the report says {published_metric!r}, this pack "
                f"gives {float(measured['pooled_bits_per_character'])!r}"
            )
    return problems


def compare(
    scored: Mapping[str, Any], pairs: Sequence[str], *, seed: int, resamples: int
) -> list[dict[str, Any]]:
    """The paired test for each ``first/second`` pair the caller named."""
    readings: list[dict[str, Any]] = []
    for entry in pairs:
        first, _, second = entry.partition("/")
        if first not in scored or second not in scored:
            message = f"pair {entry!r} names arms that were not scored ({sorted(scored)})"
            raise AnalysisError(message)
        works_first = scored[first]["works"]
        works_second = scored[second]["works"]
        if set(works_first) != set(works_second):
            message = (
                f"{first} and {second} were scored on different documents: "
                f"{len(set(works_first) ^ set(works_second))} works are on one side only"
            )
            raise AnalysisError(message)
        keys = sorted(works_first)
        differences = [
            bits_per_character(float(works_first[key]["nats"]), int(works_first[key]["characters"]))
            - bits_per_character(
                float(works_second[key]["nats"]), int(works_second[key]["characters"])
            )
            for key in keys
        ]
        weights = [float(works_first[key]["characters"]) for key in keys]
        groups = list(keys)
        test = paired_document_test(differences, weights, groups, seed=seed, resamples=resamples)
        readings.append(
            {
                "pair": {"first": first, "second": second},
                "published_difference": float(scored[first]["pooled_bits_per_character"])
                - float(scored[second]["pooled_bits_per_character"]),
                "test": test,
            }
        )
    return readings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument(
        "--selection-vocabulary",
        type=Path,
        help="the vocabulary that chose the held-out documents, when the arms' runs named one",
    )
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--expected-split-fingerprint")
    parser.add_argument("--expected-corpus-fingerprint")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--max-test-tokens", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--document-masking",
        action="store_true",
        help="score with attention confined to a document, as a run trained with document_masking must be scored",
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="NAME=path",
        help="a checkpoint to score, named as the record names it",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        metavar="NAME=path",
        help="the arm's committed run report; its denominators and pooled metric must be reproduced",
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        metavar="FIRST/SECOND",
        help="a pair to test; repeatable. The order is the sign: positive favours SECOND",
    )
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        arms = _entries(args.arm, option="arm")
        reports = _entries(args.report, option="report")
        vocabulary = SubwordVocabulary.load(args.vocabulary)
        selection = (
            None
            if args.selection_vocabulary is None
            else SubwordVocabulary.load(args.selection_vocabulary)
        )
        units = test_side(
            args.snapshot,
            split_seed=args.split_seed,
            expected_fingerprint=args.expected_split_fingerprint,
        )
        batch, packed_units = build_pack(
            units,
            vocabulary,
            context_length=args.context_length,
            max_tokens=args.max_test_tokens,
            selection_vocabulary=selection,
        )
        scored = {
            name: score_arm(
                Path(path),
                batch,
                packed_units,
                batch_size=args.batch_size,
                document_masking=args.document_masking,
            )
            for name, path in arms.items()
        }
        for name, row in scored.items():
            if (
                args.expected_corpus_fingerprint is not None
                and row["corpus_fingerprint"] != args.expected_corpus_fingerprint
            ):
                message = (
                    f"{name} was trained on corpus {row['corpus_fingerprint']}, not "
                    f"{args.expected_corpus_fingerprint!r}"
                )
                raise CorpusError(message)
            if row["tokenizer_fingerprint"] != vocabulary.fingerprint:
                message = (
                    f"{name} carries tokenizer {row['tokenizer_fingerprint']}, but the pack was cut with "
                    f"{vocabulary.fingerprint}; the two arms would be scored on different text"
                )
                raise CorpusError(message)
        problems = check_against_reports(scored, reports)
        if problems:
            message = "the rebuilt pack is not the pack these runs scored:\n  " + "\n  ".join(
                problems
            )
            raise AnalysisError(message)
        readings = compare(scored, args.pair, seed=args.seed, resamples=args.resamples)
    except (AnalysisError, CorpusError, OSError, TrainingError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    result = {
        "pack": {
            "document_masking": args.document_masking,
            "snapshot": str(args.snapshot),
            "vocabulary_fingerprint": vocabulary.fingerprint,
            "selection_vocabulary_fingerprint": None
            if selection is None
            else selection.fingerprint,
            "split_seed": args.split_seed,
            "context_length": args.context_length,
            "max_test_tokens": args.max_test_tokens,
            "blocks": len(batch.examples),
            "positions": batch.positions,
            "document_count": batch.document_count,
            "works": len({unit.group for unit in packed_units}),
        },
        "arms": scored,
        "pairs": readings,
        "bootstrap": {"seed": args.seed, "resamples": args.resamples},
        "metric_tolerance": _METRIC_TOLERANCE,
        "reports_checked": sorted(reports),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    for reading in readings:
        pair = reading["pair"]
        stats = reading["test"]
        print(
            f"{pair['first']} vs {pair['second']}: pooled {float(reading['published_difference']):+.6f}, "
            f"paired mean {float(stats['mean_difference']):+.6f} "
            f"[{float(stats['interval'][0]):+.6f}, {float(stats['interval'][1]):+.6f}], "
            f"{'clears' if stats['excludes_zero'] else 'includes'} zero on {stats['clusters']} works",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
