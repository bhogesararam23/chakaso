"""Test whether one trained arm is better than another on the held-out text itself, paired work by work.

The project has compared arms against a *band*: E2614 measured the seed-to-seed spread of the reference
configuration, and every arm difference since has been read against 0.02 bits per character. A band is a threshold,
not a test — it says a difference is bigger than what re-initialising the model produces, and says nothing about how
well the difference on *these* held-out documents is known. E2629 made the gap concrete from the other side: its
probes could not separate the same arms at all, so the only evidence that an arm was better was a number beside a
threshold.

This script pairs the measurement. Both arms score the identical held-out pack through the experiment driver's own
code path (:func:`chakaso.training.subword_experiment.work_losses`); the loss is split into the documents that
produced it, the documents are aggregated to the **works** the whole-work split held out, and the cluster bootstrap
resamples those works — because a book contributes many documents in one voice and they are not independent
observations.

Nothing is computed until the rebuild is checked against each arm's committed run report: positions scored,
characters scored and the pooled bits per character must reproduce what the record published. That guard is not
decorative. An earlier version of this script reconstructed the held-out side from the split in document order and
was refused by the reports it read: 256,256 positions and 483,895 characters against the 254,976 and 535,282 every
report prints, because ``max_test_tokens`` caps a seeded *sample* of the test side, not its prefix.

Usage:
    python scripts/paired_loss_test.py --protocol configs/experiments/E2614-decoder-seed-spread.toml \\
        --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --arm "E2612-T=data/experiments/E2612/mixture/E2612-T-decoder.pt" \\
        --arm "E2612-P=data/experiments/E2612/prose/E2612-P-decoder.pt" \\
        --report "E2612-T=experiments/E2629/E2612-T-report.json" \\
        --pair E2612-T/E2612-P --out experiments/E2633/results.json
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

from chakaso.training.errors import CorpusError, TrainingError
from chakaso.training.paired_loss import jackknife, paired_document_test

#: A faithful rebuild of a published denominator is exact in positions and characters and agrees to the digits the
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


def bits_per_character(nats: float, characters: int) -> float:
    """The series' unit: total nats over total characters, in bits.

    One place computes it, because a per-work figure and a pooled figure that disagree on the arithmetic would be
    two measurements of the same documents.
    """
    if not characters:
        return math.inf
    return math.log2(math.e) * nats / characters


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
        for key, label in (
            ("positions_scored", "positions scored"),
            ("characters_scored", "characters scored"),
        ):
            published, recomputed = int(test[key]), int(measured[key])
            if published != recomputed:
                problems.append(
                    f"{name}: {label} — the report says {published}, this pack gives {recomputed}"
                )
        published_metric = float(test["bits_per_character"])
        recomputed_metric = float(measured["pooled_bits_per_character"])
        if abs(published_metric - recomputed_metric) > _METRIC_TOLERANCE:
            problems.append(
                f"{name}: pooled bits per character — the report says {published_metric!r}, this pack "
                f"gives {recomputed_metric!r}"
            )
    return problems


def compare(
    scored: Mapping[str, Any], pairs: Sequence[str], *, seed: int, resamples: int
) -> list[dict[str, Any]]:
    """The paired test for each ``first/second`` pair the caller named, over the works both arms scored."""
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
        readings.append(
            {
                "pair": {"first": first, "second": second},
                "published_difference": float(scored[first]["pooled_bits_per_character"])
                - float(scored[second]["pooled_bits_per_character"]),
                "test": paired_document_test(
                    differences, weights, list(keys), seed=seed, resamples=resamples
                ),
                "leave_one_out": dict(jackknife(differences, weights), works=list(keys)),
            }
        )
    return readings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        required=True,
        help="the [plan] the record cites; it fixes the corpus, the split and the held-out pack",
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("data/experiments"))
    parser.add_argument("--root", type=Path, default=Path())
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--expected-corpus-fingerprint")
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
        help="a pair to test; repeatable. Positive differences favour SECOND",
    )
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        from chakaso.training.protocol import load_plan
        from chakaso.training.subword_experiment import work_losses

        arms = _entries(args.arm, option="arm")
        reports = _entries(args.report, option="report")
        plan = load_plan(
            args.protocol,
            root=args.root,
            work_dir=args.work_dir,
            fixed={
                "split_seed": args.split_seed,
                "expected_corpus_fingerprint": args.expected_corpus_fingerprint,
                "corpus_snapshot": args.snapshot,
            },
        )
        measured = work_losses(plan, {name: Path(path) for name, path in arms.items()})
        problems = check_against_reports(measured["arms"], reports)
        if problems:
            message = "the rebuilt pack is not the pack these runs scored:\n  " + "\n  ".join(
                problems
            )
            raise AnalysisError(message)
        readings = compare(measured["arms"], args.pair, seed=args.seed, resamples=args.resamples)
    except (AnalysisError, CorpusError, OSError, TrainingError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    result = {
        "pack": measured["pack"],
        "arms": measured["arms"],
        "pairs": readings,
        "bootstrap": {"seed": args.seed, "resamples": args.resamples},
        "metric_tolerance": _METRIC_TOLERANCE,
        "reports_checked": sorted(reports),
        "protocol": str(args.protocol),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    for reading in readings:
        pair, stats, out = reading["pair"], reading["test"], reading["leave_one_out"]
        print(
            f"{pair['first']} vs {pair['second']}: published {float(reading['published_difference']):+.6f}, "
            f"paired mean {float(stats['mean_difference']):+.6f} "
            f"[{float(stats['interval'][0]):+.6f}, {float(stats['interval'][1]):+.6f}], "
            f"{'clears' if stats['excludes_zero'] else 'includes'} zero on {stats['clusters']} works, "
            f"sign test p={float(stats['sign_test_p_value']):.5f}, "
            f"leave-one-out range [{min(out['estimate_without']):+.6f}, {max(out['estimate_without']):+.6f}]",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
