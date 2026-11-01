"""Count what a protocol's corpus offers each register, in the identifiers a run is budgeted in.

A share-composed run (H9) is told a fraction of an identifier budget, so before a sweep is declared the pool each
register can supply has to be known in the same unit, and so does the mixture the existing arms actually packed.
This tool loads the protocol's plan, prepares the corpus the way the experiment does (same snapshot, same split,
same committed vocabulary), and reports per register and per side: works, documents, characters and identifiers.
It also replays the two arms E2612 ran by filtering and capping (the natural mixture T and the prose control P),
because their reports record the cap and not the composition: the replay is deterministic given the split seed
and the vocabulary, and is labelled as a replay in the output.

Usage:
    python scripts/measure_training_pool.py --protocol configs/experiments/E2614-decoder-seed-spread.toml \\
        --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \\
        --work-dir data/experiments/pool --out data/experiments/pool/training-pool.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.core.atomic import write_text_atomically
from chakaso.training.errors import TrainingError
from chakaso.training.mixture import identifiers_by_register, unit_identifiers
from chakaso.training.protocol import load_plan
from chakaso.training.subword_experiment import prepare_corpus


def _register(group: str, registers: Sequence[str]) -> str:
    return next((register for register in registers if group.startswith(register)), "other")


def _side_table(units: Sequence[object], vocabulary: object, registers: Sequence[str]) -> dict:
    table: dict[str, dict[str, object]] = {register: {} for register in (*registers, "other")}
    works: dict[str, set[str]] = {register: set() for register in table}
    counts: dict[str, dict[str, int]] = {
        register: {"documents": 0, "characters": 0, "identifiers": 0} for register in table
    }
    for unit in units:
        register = _register(unit.group, registers)  # type: ignore[attr-defined]
        works[register].add(unit.group)  # type: ignore[attr-defined]
        counts[register]["documents"] += 1
        counts[register]["characters"] += len(unit.text)  # type: ignore[attr-defined]
        counts[register]["identifiers"] += unit_identifiers(unit, vocabulary)  # type: ignore[arg-type]
    for register in table:
        table[register] = {"works": len(works[register]), **counts[register]}
    return {register: row for register, row in table.items() if row["documents"]}


def _replay_cap(units: Sequence[object], vocabulary: object, cap: int) -> dict[str, int]:
    """Identifiers per work in the stream ``pack_units`` builds: whole units until the stream reaches ``cap``."""
    packed: dict[str, int] = {}
    total = 0
    for unit in units:
        if total >= cap:
            break
        identifiers = unit_identifiers(unit, vocabulary)  # type: ignore[arg-type]
        packed[unit.group] = packed.get(unit.group, 0) + identifiers  # type: ignore[attr-defined]
        total += identifiers
    return packed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    try:
        plan = load_plan(
            args.protocol,
            root=args.root,
            work_dir=args.work_dir,
            fixed={"corpus_snapshot": args.snapshot},
        )
        registers = plan.test_partitions or ()
        # One packed unit is enough: the pool is counted from the split, not from the packed stream.
        prepared = prepare_corpus(replace(plan, max_training_tokens=1))
        vocabulary = prepared.vocabulary
        seed = plan.effective_split_seed
        cap = plan.max_training_tokens or 0

        natural = list(prepared.split.train)
        random.Random(seed).shuffle(natural)
        prose = [u for u in prepared.split.train if u.group.startswith("prose-")]
        random.Random(seed).shuffle(prose)
        # The whole test pack is a seeded sample of every held-out document, packed until it reaches the test
        # cap, so its register weights are whatever that sample drew; they are not the side's proportions.
        test_pack = list(prepared.split.test)
        random.Random(seed).shuffle(test_pack)

        result = {
            "protocol": str(args.protocol),
            "corpus_fingerprint": prepared.manifest.fingerprint,
            "split_fingerprint": prepared.split.fingerprint,
            "tokenizer_fingerprint": vocabulary.fingerprint,
            "split_seed": seed,
            "budget_identifiers": cap,
            "sides": {
                side: _side_table(getattr(prepared.split, side), vocabulary, registers)
                for side in ("train", "validation", "test")
            },
            "replayed_arms": {
                "note": (
                    "E2612's reports record the cap, not the composition; these replay the same shuffle "
                    "and cap and are not read from a report"
                ),
                "T_natural_mixture": identifiers_by_register(
                    _replay_cap(natural, vocabulary, cap), registers
                ),
                "P_prose_control": identifiers_by_register(
                    _replay_cap(prose, vocabulary, cap), registers
                ),
                "whole_test_pack": identifiers_by_register(
                    _replay_cap(test_pack, vocabulary, plan.max_test_tokens or 0), registers
                ),
            },
        }
    except (TrainingError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomically(text, args.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
