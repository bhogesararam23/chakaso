"""The E2617 reading applies the record's three outcomes at their boundaries and refuses a run that is not a repeat."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# The script imports chakaso.training.seed_study, which needs the optional model-path extras (like the sibling
# analyse_share_sweep tests); the dependency-free tiers skip it.
pytest.importorskip("numpy")
pytest.importorskip("torch")

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "read_cross_machine_repeat.py"
_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_BASE = {
    "prose-": 1.84,
    "science-": 1.95,
    "technology-": 1.90,
    "medicine-": 1.84,
    "mathematics-": 2.58,
}
_WHOLE = 2.0554


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("read_cross_machine_repeat_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(shift: dict[str, float] | None = None, whole_shift: float = 0.0) -> dict[str, float]:
    shift = shift or {}
    values = {
        f"partition_{prefix}_bits_per_character": _BASE[prefix] + shift.get(prefix, 0.0)
        for prefix in _PARTITIONS
    }
    values["transformer_bits_per_character"] = _WHOLE + whole_shift
    return values


def _reference(path: Path, *, spreads: float = 0.002) -> Path:
    """An E2614-shaped results file: seed 7 plus four more seeds spread by ``spreads`` around it."""
    offsets = (0.0, spreads, -spreads, 2 * spreads, -2 * spreads)
    per_run = {
        str(seed): {
            "metrics": _metrics(dict.fromkeys(_PARTITIONS, offset), whole_shift=offset),
            "environment": {"torch": "2.13.0+cpu", "torch_threads": "14", "platform": "Windows-11"},
        }
        for seed, offset in zip((7, 11, 13, 17, 19), offsets, strict=True)
    }
    document = {
        "per_run": per_run,
        "protocol": {
            "corpus_fingerprint": "c" * 16,
            "split_fingerprint": "s" * 16,
            "tokenizer_fingerprint": "t" * 16,
            "parameter_count": 3_413_760,
            "test_positions": 1000,
            "test_characters": 2000,
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _report(
    path: Path,
    shift: dict[str, float] | None = None,
    *,
    whole_shift: float = 0.0,
    corpus: str = "c" * 16,
    parameters: int = 3_413_760,
) -> Path:
    metrics = _metrics(shift, whole_shift)
    report: dict[str, Any] = {
        "corpus": {"fingerprint": corpus, "split_fingerprint": "s" * 16},
        "tokenizer": {"fingerprint": "t" * 16},
        "environment": {"torch": "2.15.0+cpu", "torch_threads": "4", "platform": "Linux"},
        "transformer": {
            "parameter_count": parameters,
            "test": {
                "bits_per_character": metrics["transformer_bits_per_character"],
                "positions_scored": 1000,
                "characters_scored": 2000,
            },
            "test_partitions": {
                prefix: {
                    "metrics": {
                        "bits_per_character": metrics[f"partition_{prefix}_bits_per_character"]
                    }
                }
                for prefix in _PARTITIONS
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _read(tmp_path: Path, **report_kwargs: Any) -> dict[str, Any]:
    out = tmp_path / "out.json"
    status = _script().main(  # type: ignore[attr-defined]
        [
            f"--report={_report(tmp_path / 'report.json', **report_kwargs)}",
            f"--reference={_reference(tmp_path / 'reference.json')}",
            f"--out={out}",
        ]
    )
    assert status == 0
    return json.loads(out.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_an_identical_run_is_x_a(tmp_path: Path) -> None:
    assert _read(tmp_path)["outcome"] == "X-a"


def test_a_difference_exactly_at_the_tolerance_is_still_reproduced(tmp_path: Path) -> None:
    assert _read(tmp_path, shift={"prose-": 0.0005})["outcome"] == "X-a"


def test_a_difference_just_over_the_tolerance_but_inside_the_margin_is_x_b(tmp_path: Path) -> None:
    result = _read(tmp_path, shift={"prose-": 0.0006})

    assert result["outcome"] == "X-b"
    assert result["metrics"]["prose-"]["delta"] == pytest.approx(0.0006, abs=1e-9)


def test_a_difference_over_a_metrics_tie_margin_is_x_c_and_names_the_metric(tmp_path: Path) -> None:
    result = _read(tmp_path, shift={"technology-": 0.05})

    assert result["outcome"] == "X-c"
    # The shift also moves the equal-weight mean (0.05 / 5 = 0.01), which is over its own margin.
    assert result["beyond_tie_margin"] == ["technology-", "O1"]


def test_the_equal_weight_mean_is_part_of_the_reading(tmp_path: Path) -> None:
    result = _read(tmp_path, shift={"prose-": 0.0006})

    assert result["metrics"]["O1"]["delta"] == pytest.approx(0.00012, abs=1e-9)
    assert result["metrics"]["whole_pack"]["delta"] == pytest.approx(0.0, abs=1e-12)


def test_the_environments_of_both_runs_are_reported_side_by_side(tmp_path: Path) -> None:
    result = _read(tmp_path)

    assert result["environment"]["reference"]["torch"] == "2.13.0+cpu"
    assert result["environment"]["new"]["torch"] == "2.15.0+cpu"


def test_a_run_on_another_corpus_is_refused(tmp_path: Path) -> None:
    status = _script().main(  # type: ignore[attr-defined]
        [
            f"--report={_report(tmp_path / 'r.json', corpus='d' * 16)}",
            f"--reference={_reference(tmp_path / 'ref.json')}",
        ]
    )

    assert status == 2


def test_a_run_of_another_architecture_is_refused(tmp_path: Path) -> None:
    status = _script().main(  # type: ignore[attr-defined]
        [
            f"--report={_report(tmp_path / 'r.json', parameters=1_000_000)}",
            f"--reference={_reference(tmp_path / 'ref.json')}",
        ]
    )

    assert status == 2
