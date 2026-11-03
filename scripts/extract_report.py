"""Write a compact, reproducible extract of a run's report, for committing beside a record.

A measured record cites the report that produced its numbers, and the report is a git-ignored artifact under
``data/``. Committing it whole would put a 30 KB file with a 7.5 KB group listing in the repository for every
arm; committing nothing would leave the record's arithmetic uncheckable. The convention the series established
by hand — E2617, E2620, E2624 and E2625 each carry a `*-report.json` whose bulky blocks were replaced by digest
stubs — is here so the next one is produced the same way and a reader can verify the extract against the file it
came from.

A block is replaced when it is a list or mapping with more than ``--max-entries`` members, or whose canonical
JSON exceeds ``--max-bytes``; the stub records the SHA-256 of that canonical JSON, its byte length, and that it
was omitted. Everything else is copied unchanged, key order and all, so a test can assert that the numbers a
record cites survived the trip. The extract names the report file's own digest, so "this extract came from that
run" is checkable rather than asserted.

Usage:
    python scripts/extract_report.py --report data/experiments/E2626/seed-7/report.json \\
        --out experiments/E2626/E2626-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.training.outcome_vectors import ITEMS_KEY, compact_probe_result

MAX_ENTRIES = 6
MAX_BYTES = 2048

#: The two blocks the outcome compaction writes. They are the array, in its compacted form, with the digest that
#: proves a reader can rebuild it — so the walk stops at them rather than stubbing them as bulky.
COMPACT_KEYS = frozenset({"compact_v1", ITEMS_KEY})

#: The keys the series' extracts keep whole whatever their size: the numbers a record's tables are read from.
#: A kept key is descended into and only the bulky leaf inside it becomes a stub, which is what lets the same
#: script compact a run's report and a probe result — both are a thin layer of structure over a small set of
#: numbers, with the big arrays one level further down than the container that holds them.
ALWAYS_KEEP = frozenset(
    {
        "architecture",
        "artifact",
        "baselines_subword",
        "budget",
        "by_phenomenon",
        "corpus",
        "curve",
        "curve_artifacts",
        "environment",
        "experiment",
        "items",
        "metrics",
        "model_family",
        "models",
        "parameter_count",
        "parameters_by_configuration",
        "pair_set",
        "plan",
        "result",
        "test",
        "test_partitions",
        "timings",
        "tokenizer",
        "validation",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stub(value: Any) -> dict[str, Any]:
    document = _canonical(value)
    return {
        "omitted_from_extract": True,
        "canonical_json_sha256": sha256(document.encode("utf-8")).hexdigest(),
        "json_bytes": len(document),
    }


def _is_bulk(value: Any, *, max_entries: int, max_bytes: int) -> bool:
    if isinstance(value, (list, tuple, Mapping)) and not isinstance(value, str):
        if len(value) > max_entries:
            return True
        return len(_canonical(value)) > max_bytes
    return False


def _holds_kept_numbers(value: Any) -> bool:
    """Whether a mapping contains a key the series always keeps whole, one level down.

    A parent block is often large precisely because its children are: ``transformer`` exceeds the byte budget
    only because the curve and the rung list are in it. Stubbing the parent would throw away the test metrics
    along with the lists, so a block that carries a kept key is walked instead of replaced, and only the bulky
    leaf inside it becomes a stub. A block holding a compact outcome array is the same case in miniature — its
    size is the evidence, and replacing it would put the per-item rows back out of a reader's reach.
    """
    if not isinstance(value, Mapping):
        return False
    if any(str(name) in ALWAYS_KEEP or str(name) in COMPACT_KEYS for name in value):
        return True
    return any(isinstance(item, Mapping) and "compact_v1" in item for item in value.values())


def compact(
    value: Any, *, max_entries: int = MAX_ENTRIES, max_bytes: int = MAX_BYTES, key: str = ""
) -> Any:
    """Copy ``value`` for the extract, replacing bulky blocks with digest stubs.

    The walk is top-down and the keep-list is checked at every level, so ``transformer.test`` survives whole
    while a 6,500-entry group listing inside ``corpus`` does not — and a bulky block nested inside a kept key is
    still replaced, because what is being bounded is the file, not the field's name. A compact outcome block and the
    shared item listing beside it end the walk: they are already the compacted array, each with its own digest, and
    stubbing one would put the array back out of a reader's reach — the defect K-032 was filed about.
    """
    if key in COMPACT_KEYS or (isinstance(value, Mapping) and "compact_v1" in value):
        return json.loads(json.dumps(value))
    if (
        key not in ALWAYS_KEEP
        and not _holds_kept_numbers(value)
        and _is_bulk(value, max_entries=max_entries, max_bytes=max_bytes)
    ):
        return _stub(value)
    if isinstance(value, Mapping):
        return {
            str(name): compact(
                item,
                max_entries=max_entries,
                max_bytes=max_bytes,
                key=str(name),
            )
            for name, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [
            compact(item, max_entries=max_entries, max_bytes=max_bytes, key=key) for item in value
        ]
    return value


def build_extract(
    report: Mapping[str, Any],
    source: Path,
    *,
    max_entries: int,
    max_bytes: int,
    compact_outcomes: bool = True,
) -> dict[str, Any]:
    """The extract document: the compacted report, plus the block that says where it came from.

    ``compact_outcomes`` is the K-032 half: a probe result's per-item arrays are put in the digest-checked compact
    form before the bulky-block walk runs, so they survive into the extract as evidence instead of leaving as a
    stub. A run report has no outcome arrays, and the compaction is a no-op on it by construction — it looks for a
    ``models`` block and returns anything else unchanged.
    """
    document = compact_probe_result(report) if compact_outcomes else report
    body = {
        str(key): compact(value, max_entries=max_entries, max_bytes=max_bytes, key=str(key))
        for key, value in document.items()
    }
    document_json = _canonical(body)
    body["extract"] = {
        "note": "compact extract of the run's report.json; bulky blocks replaced by digest stubs",
        "report_file_sha256": sha256(Path(source).read_bytes()).hexdigest(),
        "source_path": str(source),
        "document_sha256": sha256(document_json.encode("utf-8")).hexdigest(),
        "outcomes_compacted": compact_outcomes,
    }
    return body


def render(extract: Mapping[str, Any]) -> str:
    return json.dumps(extract, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=MAX_ENTRIES)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument(
        "--compact-outcomes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="put each model's per-item outcome arrays in the digest-checked compact form (K-032) instead of "
        "stubbing them; --no-compact-outcomes reproduces the extracts written before that fix",
    )
    args = parser.parse_args(argv)
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    extract = build_extract(
        report,
        args.report,
        max_entries=args.max_entries,
        max_bytes=args.max_bytes,
        compact_outcomes=args.compact_outcomes,
    )
    text = render(extract)
    if args.out.exists():
        print(
            f"error: {args.out} already holds an extract; a measured record's evidence is not replaced "
            "unasked",
            file=sys.stderr,
        )
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(
        f"{args.out.name}: {len(text):,} bytes from {args.report.stat().st_size:,}, "
        f"report sha256 {extract['extract']['report_file_sha256'][:16]}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
