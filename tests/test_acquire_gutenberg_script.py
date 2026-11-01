"""Offline tests for the corpus acquisition script's provenance and replay checks (K-029).

The script is not part of the package, so it is loaded by path. Nothing here touches the network: the
selection is drawn over a synthetic catalogue and ``acquire`` is given a zero quota, which writes the report
without downloading anything.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "acquire_gutenberg_corpus.py"


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acquire_gutenberg_corpus_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_catalogue(path: Path, *, extra: bool = False) -> None:
    rows = []
    for number in range(1, 41):
        rows.append((number, "Text", "en", f"Title {number}", f"Author {number}", "PR"))
    for number in range(41, 61):
        rows.append((number, "Text", "en", f"Title {number}", f"Author {number}", "QC"))
    if extra:
        rows.append((61, "Text", "en", "Title 61", "Author 61", "PR"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Text#", "Type", "Language", "Title", "Authors", "LoCC"])
        writer.writerows(rows)


def _selection(script: ModuleType, catalogue: Path) -> dict[str, object]:
    buckets = script.load_candidates(catalogue)
    quotas = {"prose": 3, "science": 2}
    planned = script.plan_selection(buckets, quotas, seed=script.SELECTION_SEED)
    # A zero-download report: the walk is recorded exactly as ``acquire`` would record it.
    records = [
        {"register": register, "ebook_number": candidate.number}
        for register in script.REGISTERS
        for candidate in planned[register]
    ]
    return {
        "selection_seed": script.SELECTION_SEED,
        "quotas": quotas,
        "records": records,
        "catalog": script.describe_catalog(catalogue, retrieved="2026-11-01"),
    }


def test_a_catalogue_description_pins_its_bytes_and_says_how_the_date_is_known(
    script: ModuleType, tmp_path: Path
) -> None:
    catalogue = tmp_path / "pg_catalog.csv"
    _write_catalogue(catalogue)

    declared = script.describe_catalog(catalogue, retrieved="2026-11-01")
    inferred = script.describe_catalog(catalogue)

    assert declared["bytes"] == catalogue.stat().st_size
    assert len(str(declared["sha256"])) == 64
    assert declared["retrieved"] == "2026-11-01"
    assert declared["retrieved_basis"] == "operator-declared"
    assert declared["url"] == script.CATALOG_URL
    assert inferred["retrieved_basis"] == "file-modified-time"
    assert inferred["sha256"] == declared["sha256"]


def test_an_acquisition_report_carries_the_catalogue_block(
    script: ModuleType, tmp_path: Path
) -> None:
    catalogue = tmp_path / "pg_catalog.csv"
    _write_catalogue(catalogue)
    work = tmp_path / "work"
    buckets = script.load_candidates(catalogue)
    quotas = dict.fromkeys(script.REGISTERS, 0)

    report = script.acquire(
        script.plan_selection(buckets, quotas, seed=script.SELECTION_SEED),
        quotas,
        work=work,
        spacing=0.0,
        verbose=False,
        catalog=script.describe_catalog(catalogue, retrieved="2026-11-01"),
    )

    written = json.loads((work / "selection.json").read_text(encoding="utf-8"))
    assert written["catalog"] == report["catalog"]
    assert written["catalog"]["sha256"]


def test_replaying_the_walk_over_the_same_catalogue_reproduces_it(
    script: ModuleType, tmp_path: Path
) -> None:
    catalogue = tmp_path / "pg_catalog.csv"
    _write_catalogue(catalogue)
    report = _selection(script, catalogue)

    assert script.verify_selection(report, script.load_candidates(catalogue)) == []


def test_a_catalogue_with_one_more_row_does_not_reproduce_the_walk(
    script: ModuleType, tmp_path: Path
) -> None:
    original = tmp_path / "original.csv"
    moved = tmp_path / "moved.csv"
    _write_catalogue(original)
    _write_catalogue(moved, extra=True)
    report = _selection(script, original)

    problems = script.verify_selection(report, script.load_candidates(moved))

    assert problems, "a catalogue that gained a row shuffles a different population"
    assert any(problem.startswith("prose:") for problem in problems)
