"""Measure how much of a split's held-out text also occurs on its training side, by shared word n-grams.

A whole-work split puts no work on two sides; it cannot see two works that are the same text (editions, anthologies,
translations). This reads a pinned corpus snapshot, reproduces the split a run used (same fractions, seed and
whole-work flag), and reports per held-out work the share of its n-grams found in training
(:func:`chakaso.training.overlap.overlap_report`). It trains nothing and reads no model.

Usage:
    python scripts/measure_split_overlap.py --snapshot SNAPSHOT.json --split-seed 7 --whole-works --out overlap.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.core.atomic import write_text_atomically
from chakaso.training.data import split_text_units
from chakaso.training.manifest import read_snapshot
from chakaso.training.overlap import overlap_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--whole-works", action="store_true")
    parser.add_argument("--side", choices=("validation", "test"), default="test")
    parser.add_argument("--n", type=int, default=8, help="words per window")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    snapshot = read_snapshot(args.snapshot)
    split = split_text_units(
        snapshot.manifest.text_units(),
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
        whole_works=args.whole_works,
    )
    held_out = split.test if args.side == "test" else split.validation
    works: dict[str, list[str]] = defaultdict(list)
    for unit in held_out:
        works[unit.group or unit.identifier].append(unit.text)
    report = overlap_report(
        (unit.text for unit in split.train),
        {name: "\n\n".join(texts) for name, texts in works.items()},
        n=args.n,
    )
    report["corpus_fingerprint"] = snapshot.manifest.fingerprint
    report["split_fingerprint"] = split.fingerprint
    report["side"] = args.side
    report["training_documents"] = len(split.train)
    report["heldout_works"] = len(works)

    payload = json.dumps(report, indent=1, sort_keys=True)
    if args.out is not None:
        write_text_atomically(payload, args.out)
    pooled = report["pooled"]
    print(
        f"{args.side}: {len(works)} works, {pooled['windows']:,} windows, "
        f"{pooled['ratio']:.4%} also in training ({report['training_windows']:,} distinct training windows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
