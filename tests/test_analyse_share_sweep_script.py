"""Tests for the share-sweep analysis script: matching is proved first, and no winner is named past the margins."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

# The script reads the seed study module, which imports the experiment module and so both extras.
pytest.importorskip("numpy")
pytest.importorskip("torch")

_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyse_share_sweep.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyse_share_sweep_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(share: float | None, bits: dict[str, float], prefixes: list[str] | None = None) -> Any:
    budget: dict[str, Any] = {
        "seed": 7,
        "max_steps": 2400,
        "schedule": "cosine",
        "effective_batch_size": 32,
        "learning_rate": 0.0006,
        "training_group_prefixes": prefixes,
        "max_training_tokens": 20_000_000,
        "shuffle_training_units": True,
        "document_masking": False,
        "grad_accum_steps": 1,
    }
    if share is not None:
        budget["share_group_prefix"] = "prose-"
        budget["share_of_tokens"] = share
    return {
        "corpus": {
            "fingerprint": "c" * 16,
            "split_fingerprint": "s" * 16,
            "split_seed": 7,
            "training_tokens_packed": 20_000_000,
        },
        "tokenizer": {"fingerprint": "t" * 16, "size": 4096},
        "transformer": {
            "parameter_count": 3_413_760,
            "architecture": {"d_model": 256},
            "budget": budget,
            "test": {
                "bits_per_character": sum(bits.values()) / len(bits),
                "positions_scored": 1000,
                "characters_scored": 2000,
            },
            "test_partitions": {
                prefix: {
                    "tokens_scored": 500,
                    "metrics": {"bits_per_character": value, "characters_scored": 1000},
                }
                for prefix, value in bits.items()
            },
        },
    }


def _bits(prose: float, technical: float) -> dict[str, float]:
    return {
        "prose-": prose,
        "science-": technical,
        "technology-": technical,
        "medicine-": technical,
        "mathematics-": technical,
    }


def _study(spread: float = 0.002) -> dict[str, Any]:
    """Five seeds that differ from each other by about ``spread`` on every partition."""
    runs = {}
    for index, seed in enumerate((7, 11, 13, 17, 19)):
        offset = (index - 2) * spread
        runs[str(seed)] = {
            "metrics": {
                f"partition_{prefix}_bits_per_character": 2.0 + offset for prefix in _PARTITIONS
            }
        }
    return {"per_run": runs}


def _write(tmp_path: Path, name: str, payload: Any) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _arms(tmp_path: Path, spec: dict[float, dict[str, float]]) -> list[str]:
    paths = []
    for share, bits in spec.items():
        prefixes = ["prose-"] if share == 1.0 else None
        report = _report(None if share == 1.0 else share, bits, prefixes)
        paths.append(f"--arm={share}={_write(tmp_path, f's{share}.json', report)}")
    return paths


def test_a_clear_interior_optimum_is_named_and_a_trade_off_gives_no_dominance(
    tmp_path: Path,
) -> None:
    spec = {
        0.0: _bits(prose=2.40, technical=1.90),
        0.25: _bits(prose=2.00, technical=1.92),
        0.5: _bits(prose=1.90, technical=1.99),
        1.0: _bits(prose=1.80, technical=2.40),
    }
    out = tmp_path / "out.json"
    noise = _write(tmp_path, "noise.json", _study())

    status = _script().main([*_arms(tmp_path, spec), f"--noise-study={noise}", f"--out={out}"])  # type: ignore[attr-defined]

    assert status == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["all_conditions_matched"] is True
    assert result["sweep"]["outcome"] == "interior_optimum"
    assert result["sweep"]["best_share"] == 0.25
    # A mixture trades prose for technical text, so no arm is better on every register.
    assert result["dominance_pairs"] == []
    assert result["arms"]["0.25"]["macro"] == pytest.approx((2.00 + 4 * 1.92) / 5)


def test_arms_within_the_noise_of_each_other_are_left_unresolved(tmp_path: Path) -> None:
    spec = {
        0.0: _bits(prose=2.000, technical=2.000),
        0.5: _bits(prose=2.004, technical=1.998),
        1.0: _bits(prose=1.998, technical=2.003),
    }
    out = tmp_path / "out.json"
    noise = _write(tmp_path, "noise.json", _study())

    assert _script().main([*_arms(tmp_path, spec), f"--noise-study={noise}", f"--out={out}"]) == 0  # type: ignore[attr-defined]

    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["sweep"]["outcome"] == "unresolved"
    assert sorted(result["sweep"]["within_margin_of_best"]) == [0.0, 0.5, 1.0]


def test_an_arm_whose_label_disagrees_with_its_report_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arms = {
        0.0: _report(0.25, _bits(2.4, 1.9)),
        0.5: _report(0.5, _bits(1.9, 2.0)),
        1.0: _report(None, _bits(1.8, 2.4), ["prose-"]),
    }
    paths = [f"--arm={share}={_write(tmp_path, f'{share}.json', r)}" for share, r in arms.items()]
    noise = _write(tmp_path, "noise.json", _study())

    status = _script().main([*paths, f"--noise-study={noise}"])  # type: ignore[attr-defined]

    assert status == 1
    assert "mislabelled arm" in capsys.readouterr().err


def test_arms_that_were_not_matched_yield_no_valid_reading(tmp_path: Path) -> None:
    spec = {
        0.0: _bits(prose=2.40, technical=1.90),
        0.5: _bits(prose=1.90, technical=1.99),
        1.0: _bits(prose=1.80, technical=2.40),
    }
    paths = _arms(tmp_path, spec)
    changed = _report(0.5, spec[0.5])
    changed["tokenizer"]["fingerprint"] = "x" * 16
    _write(tmp_path, "s0.5.json", changed)
    noise = _write(tmp_path, "noise.json", _study())

    assert _script().main([*paths, f"--noise-study={noise}"]) == 2  # type: ignore[attr-defined]


def test_the_nested_half_arm_is_compared_with_the_natural_mixture_mean(tmp_path: Path) -> None:
    spec = {
        0.0: _bits(prose=2.40, technical=1.90),
        0.5: _bits(prose=2.0, technical=2.0),
        1.0: _bits(prose=1.80, technical=2.40),
    }
    out = tmp_path / "out.json"
    study = _write(tmp_path, "noise.json", _study())

    status = _script().main(  # type: ignore[attr-defined]
        [
            *_arms(tmp_path, spec),
            f"--noise-study={study}",
            f"--natural-mixture-study={study}",
            f"--out={out}",
        ]
    )

    assert status == 0
    effect = json.loads(out.read_text(encoding="utf-8"))["selection_effect"]
    assert effect["available"] is True
    assert effect["natural_mixture_runs"] == 5
    assert effect["differences"]["prose-"]["nested_half_minus_natural_mean"] == pytest.approx(0.0)
    assert effect["differences"]["prose-"]["reading"] == "tie"


def _committed_summary(report: Any, *, tokens_scored: int = 500) -> dict[str, Any]:
    """A report as a committed ``results.json`` keeps it: flat, partitions without a ``metrics`` wrapper or character counts."""
    transformer = report["transformer"]
    return {
        "corpus": report["corpus"],
        "tokenizer": report["tokenizer"],
        "parameter_count": transformer["parameter_count"],
        "architecture": transformer["architecture"],
        # E2612's committed block predates these two fields; they are absent, not false.
        "budget": {
            key: value
            for key, value in transformer["budget"].items()
            if key not in ("shuffle_training_units", "document_masking")
        },
        "test": dict(transformer["test"]),
        "test_partitions": {
            prefix: {
                "bits_per_character": block["metrics"]["bits_per_character"],
                "tokens_scored": tokens_scored,
                "documents": 3,
            }
            for prefix, block in transformer["test_partitions"].items()
        },
    }


def _committed_endpoint_args(tmp_path: Path, *, tokens_scored: int = 500) -> list[str]:
    spec = {0.0: _bits(2.40, 1.90), 0.5: _bits(1.90, 1.99)}
    paths = _arms(tmp_path, spec)
    endpoint = _report(None, _bits(1.80, 2.40), ["prose-"])
    committed = {"arms": {"P": _committed_summary(endpoint, tokens_scored=tokens_scored)}}
    paths.append(f"--arm=1.0={_write(tmp_path, 'results.json', committed)}#P")
    return paths


def test_an_endpoint_read_from_a_committed_summary_is_matched_and_labelled_as_borrowing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out.json"
    noise = _write(tmp_path, "noise.json", _study())

    status = _script().main(  # type: ignore[attr-defined]
        [*_committed_endpoint_args(tmp_path), f"--noise-study={noise}", f"--out={out}"]
    )

    assert status == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["all_conditions_matched"] is True
    assert result["arms"]["1"]["partitions"]["prose-"] == 1.80
    borrowed = result["borrowed_from_a_full_report"]["1"]
    assert borrowed["borrowed_from_a_full_report"] == [
        "test_partitions[*].metrics.characters_scored"
    ]
    assert borrowed["checked_equal_first"] == ["tokens_scored"]
    assert borrowed["not_recorded_in_committed_summary"] == [
        "document_masking",
        "shuffle_training_units",
    ]
    assert result[
        "declared_differences"
    ]  # the unrecorded fields surface as differences, never as matches


def test_a_committed_endpoint_whose_held_out_pack_differs_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    noise = _write(tmp_path, "noise.json", _study())

    status = _script().main(  # type: ignore[attr-defined]
        [*_committed_endpoint_args(tmp_path, tokens_scored=499), f"--noise-study={noise}"]
    )

    assert status == 1
    assert "held-out packs differ" in capsys.readouterr().err
