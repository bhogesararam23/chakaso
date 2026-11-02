"""The E2616 reading applies the record's three rules at their boundaries and refuses a mismatched training text."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "read_one_pass_control.py"
_T, _H, _P = 1.83952, 1.80621, 1.80926


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("read_one_pass_control_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _references() -> dict[str, Any]:
    values = {"T": _T, "H": _H, "P": _P}
    return {
        "comparison_reference_T": {
            "partitions": {
                prefix: {
                    "values": {
                        arm: bias + (0.0 if prefix == "prose-" else 0.5)
                        for arm, bias in values.items()
                    }
                }
                for prefix in _PARTITIONS
            }
        }
    }


def _report(prose: float, packed: int = 10_001_764) -> dict[str, Any]:
    return {
        "corpus": {"training_tokens_packed": packed},
        "transformer": {
            "test_partitions": {
                prefix: {"metrics": {"bits_per_character": prose if prefix == "prose-" else 2.5}}
                for prefix in _PARTITIONS
            }
        },
    }


@pytest.mark.parametrize(
    ("x", "outcome"),
    [
        (_T + 0.05, "R-a"),
        (_T - 0.00625, "R-a"),  # exactly on the tie margin of T counts as within it
        (_T - 0.00626, "R-c"),
        (_H + 0.00625, "R-b"),
        (_H + 0.00626, "R-c"),
        (_H - 0.01, "R-b"),
    ],
)
def test_the_outcome_follows_the_records_boundaries(x: float, outcome: str) -> None:
    assert _script().classify(x, t=_T, h=_H)["outcome"] == outcome  # type: ignore[attr-defined]


def test_an_arm_much_worse_than_t_is_flagged_as_evidence_the_schedule_is_too_short() -> None:
    result = _script().classify(_T + 0.0999, t=_T, h=_H)  # type: ignore[attr-defined]

    assert result["outcome"] == "R-a"
    assert result["worse_than_t_by_more_than_the_separable_margin"] is True


def test_an_in_between_result_reports_its_fraction_of_the_way_from_h_to_t() -> None:
    midpoint = (_T + _H) / 2

    result = _script().classify(midpoint, t=_T, h=_H)  # type: ignore[attr-defined]

    assert result["outcome"] == "R-c"
    assert result["fraction_of_the_way_from_h_to_t"] == pytest.approx(0.5, abs=1e-5)


def test_references_that_overlap_are_refused() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _script().classify(1.8, t=1.81, h=1.805)  # type: ignore[attr-defined]


def test_a_different_training_text_fails_the_run(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    refs = tmp_path / "refs.json"
    report.write_text(json.dumps(_report(1.9, packed=10_000_000)), encoding="utf-8")
    refs.write_text(json.dumps(_references()), encoding="utf-8")

    status = _script().main([f"--report={report}", f"--references={refs}"])  # type: ignore[attr-defined]

    assert status == 2


def test_the_reading_reports_the_other_partitions_and_the_mean_without_deciding_on_them(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    refs = tmp_path / "refs.json"
    out = tmp_path / "out.json"
    report.write_text(json.dumps(_report(1.93944)), encoding="utf-8")
    refs.write_text(json.dumps(_references()), encoding="utf-8")

    status = _script().main([f"--report={report}", f"--references={refs}", f"--out={out}"])  # type: ignore[attr-defined]

    result = json.loads(out.read_text(encoding="utf-8"))
    assert status == 0
    assert result["primary"]["outcome"] == "R-a"
    assert result["training_text_identical_to_e2613_h"] is True
    assert set(result["not_decision_bearing"]["equal_weight_mean"]) == {"X", "T", "H", "P"}
