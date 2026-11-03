"""Tests for the report extractor every measured record's committed evidence is produced by.

The property that matters is that the numbers a record cites survive the trip while the bulky lists do not, and
that "this extract came from that file" is checkable rather than asserted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "extract_report.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_report_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report() -> dict[str, Any]:
    return {
        "experiment": "E9",
        "transformer": {
            "test": {"bits_per_character": 2.05544739454246, "positions_scored": 254976},
            "test_partitions": {
                f"{prefix}-": {"metrics": {"bits_per_character": 1.9}, "works": 3}
                for prefix in ("prose", "science", "technology", "medicine", "mathematics")
            },
            "curve": [{"step": step, "validation_loss": 2.0 + step} for step in range(20)],
            "curve_artifacts": [
                {"step": step, "path": f"E9-step{step}-decoder.pt", "checksum": "0" * 64}
                for step in range(20)
            ],
        },
        "corpus": {
            "fingerprint": "312571f9fa8351a8",
            "groups_per_side": {f"doc-{index}": f"work-{index}" for index in range(500)},
            "duplicate_groups": [["a/part-0", "b/part-1"], ["c/part-0", "d/part-1"]],
        },
        "samples": [{"prompt": "x" * 4000, "text": "y" * 4000}],
    }


def _written(tmp_path: Path) -> tuple[ModuleType, Path]:
    module = _script()
    source = tmp_path / "report.json"
    source.write_text(json.dumps(_report()), encoding="utf-8")
    return module, source


def test_the_numbers_a_record_cites_survive_and_the_bulk_does_not(tmp_path: Path) -> None:
    module, source = _written(tmp_path)

    extract = module.build_extract(
        json.loads(source.read_text("utf-8")), source, max_entries=6, max_bytes=2048
    )

    assert extract["transformer"]["test"]["bits_per_character"] == 2.05544739454246
    assert len(extract["transformer"]["curve"]) == 20, "the ladder itself must not be stubbed"
    assert len(extract["transformer"]["curve_artifacts"]) == 20
    assert extract["corpus"]["fingerprint"] == "312571f9fa8351a8"
    assert extract["corpus"]["duplicate_groups"] == [
        ["a/part-0", "b/part-1"],
        ["c/part-0", "d/part-1"],
    ]
    assert extract["corpus"]["groups_per_side"]["omitted_from_extract"] is True
    assert (
        module.render(extract).count("omitted_from_extract") == 2
    )  # the group map and the samples


def test_a_stub_records_the_digest_and_length_of_what_it_replaced(tmp_path: Path) -> None:
    module, source = _written(tmp_path)
    report = json.loads(source.read_text("utf-8"))

    extract = module.build_extract(report, source, max_entries=6, max_bytes=2048)
    stub = extract["corpus"]["groups_per_side"]
    original = report["corpus"]["groups_per_side"]

    document = json.dumps(original, sort_keys=True, separators=(",", ":"))
    assert stub["canonical_json_sha256"] == sha256(document.encode("utf-8")).hexdigest()
    assert stub["json_bytes"] == len(document)
    assert "value" not in stub


def test_the_provenance_block_names_the_file_and_recomputes_from_the_extract(
    tmp_path: Path,
) -> None:
    module, source = _written(tmp_path)
    report = json.loads(source.read_text("utf-8"))

    extract = module.build_extract(report, source, max_entries=6, max_bytes=2048)

    assert extract["extract"]["report_file_sha256"] == sha256(source.read_bytes()).hexdigest()
    # A reader holding only the extract file can reproduce document_sha256: drop the provenance block,
    # canonicalise what remains, and hash it.
    body = {key: value for key, value in extract.items() if key != "extract"}
    document = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert extract["extract"]["document_sha256"] == sha256(document.encode("utf-8")).hexdigest()


def test_the_output_is_smaller_than_the_input_by_a_margin_that_matters(tmp_path: Path) -> None:
    module, source = _written(tmp_path)
    report = json.loads(source.read_text("utf-8"))

    text = module.render(module.build_extract(report, source, max_entries=6, max_bytes=2048))

    assert len(text) < source.stat().st_size / 3


def test_writing_is_deterministic_and_refuses_to_replace_an_existing_extract(
    tmp_path: Path,
) -> None:
    module, source = _written(tmp_path)
    out = tmp_path / "E9-report.json"

    assert module.main(["--report", str(source), "--out", str(out)]) == 0
    first = out.read_text(encoding="utf-8")

    assert module.main(["--report", str(source), "--out", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == first, (
        "the refused run left the file byte for byte alone"
    )


def test_a_missing_report_is_a_named_failure_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script()

    assert (
        module.main(["--report", str(tmp_path / "nope.json"), "--out", str(tmp_path / "out.json")])
        == 1
    )
    assert "error:" in capsys.readouterr().err
