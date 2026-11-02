"""The E2619 reading applies Q-a / Q-b / Q-c at their boundaries and refuses arms that are not a pool-size comparison."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "read_passes_at_fixed_updates.py"
_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_CAPS = {"5m": 5_000_000, "10m": 10_000_000, "20m": 20_000_000}


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("read_passes_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(
    path: Path,
    tag: str,
    prose: float,
    *,
    steps: int = 2400,
    prefixes: list[str] | None = None,
    corpus: str = "c" * 16,
) -> str:
    report: dict[str, Any] = {
        "corpus": {
            "fingerprint": corpus,
            "split_fingerprint": "s" * 16,
            "training_tokens_packed": _CAPS[tag] + 1_000,
        },
        "tokenizer": {"fingerprint": "t" * 16},
        "plan": {"context_length": 256, "batch_size": 32},
        "transformer": {
            "parameter_count": 3_413_760,
            "budget": {
                "max_steps": steps,
                "batch_size": 32,
                "max_training_tokens": _CAPS[tag],
                "training_group_prefixes": ["prose-"] if prefixes is None else prefixes,
                "completion": "budget_ceiling",
            },
            "test_partitions": {
                prefix: {"metrics": {"bits_per_character": prose if prefix == "prose-" else 2.0}}
                for prefix in _PARTITIONS
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return f"--arm={tag}={path}"


def _references(path: Path) -> str:
    values = {"H": 1.80621, "P": 1.80926, "T": 1.83952}
    document = {
        "comparison_reference_T": {
            "partitions": {
                prefix: {
                    "values": {
                        arm: value if prefix == "prose-" else 2.0 for arm, value in values.items()
                    }
                }
                for prefix in _PARTITIONS
            }
        }
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return f"--references={path}"


def _read(tmp_path: Path, x5: float, x10: float, x20: float) -> dict[str, Any]:
    out = tmp_path / "out.json"
    status = _script().main(  # type: ignore[attr-defined]
        [
            _report(tmp_path / "a.json", "5m", x5),
            _report(tmp_path / "b.json", "10m", x10),
            _report(tmp_path / "c.json", "20m", x20),
            _references(tmp_path / "refs.json"),
            f"--out={out}",
        ]
    )
    assert status == 0
    return json.loads(out.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_values_within_the_tie_margin_of_one_another_are_q_b(tmp_path: Path) -> None:
    assert _read(tmp_path, 1.8100, 1.8080, 1.8040)["outcome"] == "Q-b"


def test_the_range_exactly_at_the_margin_is_still_q_b(tmp_path: Path) -> None:
    assert _read(tmp_path, 1.81125, 1.8100, 1.8050)["outcome"] == "Q-b"


def test_a_smaller_pool_that_is_worse_by_more_than_the_margin_is_q_a(tmp_path: Path) -> None:
    result = _read(tmp_path, 1.8300, 1.8120, 1.8100)

    assert result["outcome"] == "Q-a"
    assert result["differences"]["5m_minus_20m"]["grade"] == "indeterminate"


def test_a_pool_that_is_worse_but_not_the_smallest_is_q_c(tmp_path: Path) -> None:
    assert _read(tmp_path, 1.8100, 1.8300, 1.8000)["outcome"] == "Q-c"


def test_a_difference_over_the_separable_margin_is_graded_so(tmp_path: Path) -> None:
    result = _read(tmp_path, 1.8500, 1.8120, 1.8100)

    assert result["differences"]["5m_minus_20m"]["grade"] == "separable"


def test_realised_passes_are_positions_over_the_packed_identifiers(tmp_path: Path) -> None:
    result = _read(tmp_path, 1.81, 1.81, 1.81)

    assert result["arms"]["5m"]["passes"] == pytest.approx(19_660_800 / 5_001_000, rel=1e-6)
    assert result["arms"]["20m"]["passes"] == pytest.approx(19_660_800 / 20_001_000, rel=1e-6)


def test_the_earlier_host_values_for_the_10m_and_20m_configurations_are_reported_beside(
    tmp_path: Path,
) -> None:
    result = _read(tmp_path, 1.81, 1.8100, 1.8200)

    earlier = result["earlier_host"]
    assert earlier["10m"]["reference"] == pytest.approx(1.80621)
    assert earlier["10m"]["this_host_minus_reference"] == pytest.approx(1.8100 - 1.80621, abs=1e-9)
    assert earlier["20m"]["reference"] == pytest.approx(1.80926)


def test_the_two_descriptive_predictions_are_reported(tmp_path: Path) -> None:
    result = _read(tmp_path, 1.8200, 1.8100, 1.8090)

    assert result["predictions"]["prose_non_decreasing_as_the_pool_shrinks"] is True
    assert result["predictions"]["ten_and_twenty_million_tie"] is True


def test_arms_with_different_update_counts_are_refused(tmp_path: Path) -> None:
    status = _script().main(  # type: ignore[attr-defined]
        [
            _report(tmp_path / "a.json", "5m", 1.8, steps=1200),
            _report(tmp_path / "b.json", "10m", 1.8),
            _report(tmp_path / "c.json", "20m", 1.8),
            _references(tmp_path / "r.json"),
        ]
    )

    assert status == 2


def test_an_arm_that_is_not_prose_only_is_refused(tmp_path: Path) -> None:
    status = _script().main(  # type: ignore[attr-defined]
        [
            _report(tmp_path / "a.json", "5m", 1.8, prefixes=[]),
            _report(tmp_path / "b.json", "10m", 1.8),
            _report(tmp_path / "c.json", "20m", 1.8),
            _references(tmp_path / "r.json"),
        ]
    )

    assert status == 2


def test_an_arm_on_another_corpus_is_refused(tmp_path: Path) -> None:
    status = _script().main(  # type: ignore[attr-defined]
        [
            _report(tmp_path / "a.json", "5m", 1.8, corpus="d" * 16),
            _report(tmp_path / "b.json", "10m", 1.8),
            _report(tmp_path / "c.json", "20m", 1.8),
            _references(tmp_path / "r.json"),
        ]
    )

    assert status == 2
