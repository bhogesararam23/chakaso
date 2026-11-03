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


def test_a_probe_result_keeps_its_tables_and_stubs_only_the_per_item_arrays() -> None:
    """The walk itself: an array that is bulky and not compacted becomes a digest stub, one level below its container.

    ``models`` is a mapping of thirteen entries whose canonical JSON is enormous because each carries 800
    per-item outcomes; stubbing the container would throw away every accuracy the record cites, so the walk
    descends through a kept key and replaces only the leaf that is actually bulky. This is what an extract looks
    like written with ``--no-compact-outcomes``, and it is the shape K-032 filed: the next test is the fix.
    """
    module = _script()
    probe = {
        "pair_set": {"digest": "d" * 64, "name": "grammatical-minimal-pairs-v2"},
        "models": {
            f"step{step}": {
                "checkpoint_checksum": "c" * 64,
                "result": {
                    "accuracy": 0.8,
                    "by_phenomenon": {
                        f"phenomenon_{index}": {"accuracy": 0.9, "pairs": 100} for index in range(8)
                    },
                    "outcomes": [[f"p{index}", "agreement", 1.0] for index in range(800)],
                },
            }
            for step in range(200, 2600, 200)
        },
    }

    compacted = module.compact(probe, max_entries=20, max_bytes=2048, key="root")

    assert len(compacted["models"]) == 12
    first = compacted["models"]["step200"]["result"]
    assert first["accuracy"] == 0.8
    assert len(first["by_phenomenon"]) == 8
    assert first["outcomes"]["omitted_from_extract"] is True
    assert first["outcomes"]["json_bytes"] > 2048


def _probe_result() -> dict[str, Any]:
    return {
        "pair_set": {"digest": "d" * 64, "name": "grammatical-minimal-pairs-v2"},
        "models": {
            f"step{step}": {
                "checkpoint_checksum": "c" * 64,
                "result": {
                    "accuracy": 0.8,
                    "by_phenomenon": {
                        f"phenomenon_{index}": {"accuracy": 0.9, "pairs": 100} for index in range(8)
                    },
                    "outcomes": [[f"p{index}", "agreement", 1.0] for index in range(800)],
                },
            }
            for step in range(200, 2600, 200)
        },
    }


def test_an_extract_of_a_probe_result_keeps_the_per_item_arrays_compact(tmp_path: Path) -> None:
    """K-032's fix through the CLI: the array a McNemar test is computed from stays in the committed file.

    Twelve models, the same 800 items, one bit per item per model and the item identities hoisted once. The
    property is not the size, it is that ``restore_probe_result`` on the bytes that landed in the repository
    rebuilds the arrays the run scored — the reading E2629 was refused.
    """
    from chakaso.training.outcome_vectors import restore_probe_result

    module = _script()
    source = tmp_path / "minimal-pairs.json"
    source.write_text(json.dumps(_probe_result()), encoding="utf-8")
    out = tmp_path / "extract.json"
    assert module.main([f"--report={source}", f"--out={out}"]) == 0
    extract = json.loads(out.read_text(encoding="utf-8"))

    assert extract["extract"]["outcomes_compacted"] is True
    assert len(extract["outcome_items"]["result.outcomes"]) == 800
    for entry in extract["models"].values():
        block = entry["result"]["outcomes"]["compact_v1"]
        assert block["vectors"][""] == "1" * 800
        assert "omitted_from_extract" not in entry["result"]["outcomes"]
        # The tables the record prints still survived alongside the array.
        assert entry["result"]["accuracy"] == 0.8
        assert len(entry["result"]["by_phenomenon"]) == 8
    restored = restore_probe_result(extract)
    assert (
        restored["models"]["step200"]["result"]["outcomes"]
        == _probe_result()["models"]["step200"]["result"]["outcomes"]
    )
    # The size claim, measured on like for like: the extract is rendered with an indent and the fixture is not, so
    # the honest comparison is between the two documents, not the two files.
    probe_canonical = len(json.dumps(_probe_result(), sort_keys=True, separators=(",", ":")))
    extract_canonical = len(json.dumps(extract, sort_keys=True, separators=(",", ":")))
    assert extract_canonical < probe_canonical / 3


def test_the_stub_form_a_committed_extract_was_written_in_is_still_reachable(
    tmp_path: Path,
) -> None:
    """``--no-compact-outcomes`` reproduces the pre-K-032 extract, so an old record's bytes are still producible.

    Kept as a test because E2626's and E2627's committed evidence is in that shape and a reader is entitled to
    regenerate it and compare.
    """
    module = _script()
    source = tmp_path / "minimal-pairs.json"
    source.write_text(json.dumps(_probe_result()), encoding="utf-8")
    out = tmp_path / "extract.json"
    assert module.main([f"--report={source}", f"--out={out}", "--no-compact-outcomes"]) == 0
    extract = json.loads(out.read_text(encoding="utf-8"))

    assert extract["extract"]["outcomes_compacted"] is False
    assert "outcome_items" not in extract
    stub = extract["models"]["step200"]["result"]["outcomes"]
    assert stub["omitted_from_extract"] is True
    assert stub["json_bytes"] > 2048


def test_a_run_report_is_unaffected_because_it_has_no_outcome_arrays(tmp_path: Path) -> None:
    """``--compact-outcomes`` is a no-op on a run report's bytes, because a run report has no outcome arrays.

    Every measured record's arm evidence is written by this one script, so a change that quietly re-styled a run
    report would break the digests the records already cite. The two extracts differ in one honest field: which
    mode produced them.
    """
    module = _script()
    source = tmp_path / "report.json"
    source.write_text(json.dumps(_report()), encoding="utf-8")
    compacted = tmp_path / "with.json"
    plain = tmp_path / "without.json"
    assert module.main([f"--report={source}", f"--out={compacted}"]) == 0
    assert module.main([f"--report={source}", f"--out={plain}", "--no-compact-outcomes"]) == 0
    with_block = json.loads(compacted.read_text(encoding="utf-8"))
    without_block = json.loads(plain.read_text(encoding="utf-8"))
    with_block.pop("extract")
    without_block.pop("extract")
    assert with_block == without_block


def test_a_partition_metrics_table_survives_because_a_record_cites_its_numbers() -> None:
    """The first committed extract of E2626 stubbed every partition's `metrics` block.

    A metrics table is eleven keys, so the entry bound ate it whole — and the per-partition bits per character
    are exactly what a record's table quotes and what an equal-weight mean is recomputed from. The kept set now
    names `metrics`, and this test is the reason: an extract that has lost the numbers a record prints is worse
    than no extract, because it looks like evidence.
    """
    module = _script()
    report = {
        "transformer": {
            "test_partitions": {
                "prose-": {
                    "works": 35,
                    "metrics": {
                        **{f"field_{index}": index for index in range(9)},
                        "bits_per_character": 1.8175,
                    },
                }
            }
        }
    }

    compacted = module.compact(report, max_entries=6, max_bytes=2048, key="root")

    metrics = compacted["transformer"]["test_partitions"]["prose-"]["metrics"]
    assert metrics["bits_per_character"] == 1.8175
    assert "omitted_from_extract" not in metrics


def test_a_missing_report_is_a_named_failure_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script()

    assert (
        module.main(["--report", str(tmp_path / "nope.json"), "--out", str(tmp_path / "out.json")])
        == 1
    )
    assert "error:" in capsys.readouterr().err
