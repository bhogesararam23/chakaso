"""E2618's analysis: a Student-t quantile that matches published tables, and grades that follow the record's rules."""

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

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyse_arm_replication.py"
_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_SEEDS = (7, 11, 13, 17, 19)
# Deviations from an arm's centre with a sample standard deviation of 0.00316 (the five-seed pattern 0, ±1, ±2 units).
_PATTERN = (0.0, 0.002, -0.002, 0.004, -0.004)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyse_arm_replication_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("p", "df", "expected"),
    [
        (0.975, 4, 2.7764),
        (0.975, 6, 2.4469),
        (0.975, 9, 2.2622),
        (0.975, 12, 2.1788),
        (0.995, 6, 3.7074),
        (0.995, 12, 3.0545),
        (0.99, 10, 2.7638),
        (0.5, 7, 0.0),
    ],
)
def test_the_t_quantile_matches_published_tables(p: float, df: int, expected: float) -> None:
    assert _script().student_t_quantile(p, df) == pytest.approx(expected, abs=1e-3)  # type: ignore[attr-defined]


def test_the_t_quantile_is_symmetric() -> None:
    quantile = _script().student_t_quantile  # type: ignore[attr-defined]

    assert quantile(0.025, 12) == pytest.approx(-quantile(0.975, 12), abs=1e-9)


def _metrics(value: float) -> dict[str, float]:
    values = {f"partition_{prefix}_bits_per_character": value for prefix in _PARTITIONS}
    values["transformer_bits_per_character"] = value
    return values


def _study(
    centre: float, *, offset: float = 0.0, seeds: tuple[int, ...] = _SEEDS
) -> dict[str, Any]:
    return {
        "per_run": {
            str(seed): {"metrics": _metrics(centre + _PATTERN[index] + offset)}
            for index, seed in enumerate(_SEEDS)
            if seed in seeds
        }
    }


def _noise(path: Path) -> Path:
    path.write_text(json.dumps(_study(2.02)), encoding="utf-8")
    return path


def _run(tmp_path: Path, centres: dict[float, float], **kwargs: Any) -> dict[str, Any]:
    arms = []
    for share, centre in centres.items():
        path = tmp_path / f"arm-{share}.json"
        path.write_text(json.dumps(_study(centre, **kwargs)), encoding="utf-8")
        arms.append(f"--arm={share}={path}")
    out = tmp_path / "out.json"
    status = _script().main(
        [*arms, f"--noise-study={_noise(tmp_path / 'noise.json')}", f"--out={out}"]
    )  # type: ignore[attr-defined]
    assert status == 0
    return json.loads(out.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_arms_far_apart_are_ordered_and_the_interior_arm_that_wins_supports_h9_a(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, {0.0: 2.040, 0.25: 2.000, 0.5: 2.020})

    grades = {pair["pair"]: pair for pair in result["primary"]["pairs"]}
    assert grades["0.0 vs 0.25"]["grade"] == "ORDERED"
    assert grades["0.0 vs 0.25"]["better"] == "0.25"
    assert result["primary"]["lowest_mean_arm"] == "0.25"
    assert result["primary"]["outcome"] == "H9-a'"


def test_arms_that_differ_by_far_less_than_the_margin_are_tied_and_unresolved(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, {0.0: 2.0000, 0.25: 2.0002, 0.5: 2.0004})

    assert {pair["grade"] for pair in result["primary"]["pairs"]} == {"TIED"}
    assert result["primary"]["outcome"] == "H9-c'"


def test_no_prose_wins_only_when_it_is_ordered_against_both_other_arms(tmp_path: Path) -> None:
    assert _run(tmp_path, {0.0: 2.000, 0.25: 2.030, 0.5: 2.040})["primary"]["outcome"] == "H9-b'"


def test_a_boundary_arm_ordered_against_only_one_neighbour_is_unresolved(tmp_path: Path) -> None:
    result = _run(tmp_path, {0.0: 2.000, 0.25: 2.0005, 0.5: 2.030})

    assert result["primary"]["lowest_mean_arm"] == "0.0"
    assert result["primary"]["outcome"] == "H9-c'"


def test_a_difference_that_is_detectable_but_inside_the_margin_is_flagged_negligible(
    tmp_path: Path,
) -> None:
    # A spread this small makes 0.005 detectable while it is under the 0.0062 equivalence margin.
    arms = {}
    for share, centre in ((0.0, 2.000), (0.25, 2.005), (0.5, 2.012)):
        path = tmp_path / f"arm-{share}.json"
        tight = {
            "per_run": {
                str(seed): {"metrics": _metrics(centre + factor)}
                for seed, factor in zip(
                    _SEEDS, (0.0, 0.0002, -0.0002, 0.0004, -0.0004), strict=True
                )
            }
        }
        path.write_text(json.dumps(tight), encoding="utf-8")
        arms[share] = path
    out = tmp_path / "out.json"
    status = _script().main(  # type: ignore[attr-defined]
        [
            *[f"--arm={share}={path}" for share, path in arms.items()],
            f"--noise-study={_noise(tmp_path / 'noise.json')}",
            f"--out={out}",
        ]
    )
    result = json.loads(out.read_text(encoding="utf-8"))

    pair = {p["pair"]: p for p in result["primary"]["pairs"]}["0.0 vs 0.25"]
    assert status == 0
    assert pair["grade"] == "ORDERED"
    assert pair["negligible"] is True
    # Negligible differences cannot name an arm.
    assert result["primary"]["outcome"] == "H9-c'"


def test_only_the_seeds_every_arm_completed_are_used(tmp_path: Path) -> None:
    complete = tmp_path / "complete.json"
    partial = tmp_path / "partial.json"
    complete.write_text(json.dumps(_study(2.0)), encoding="utf-8")
    partial.write_text(json.dumps(_study(2.0, seeds=(7, 11, 13))), encoding="utf-8")
    out = tmp_path / "out.json"

    status = _script().main(  # type: ignore[attr-defined]
        [
            f"--arm=0.0={complete}",
            f"--arm=0.25={partial}",
            f"--noise-study={_noise(tmp_path / 'noise.json')}",
            f"--out={out}",
        ]
    )

    result = json.loads(out.read_text(encoding="utf-8"))
    assert status == 0
    assert result["seeds_used"] == [7, 11, 13]
    assert result["n"] == 3


def test_fewer_than_three_common_seeds_is_refused(tmp_path: Path) -> None:
    one = tmp_path / "one.json"
    other = tmp_path / "other.json"
    one.write_text(json.dumps(_study(2.0, seeds=(7, 11))), encoding="utf-8")
    other.write_text(json.dumps(_study(2.0, seeds=(7, 11))), encoding="utf-8")

    status = _script().main(  # type: ignore[attr-defined]
        [f"--arm=0.0={one}", f"--arm=0.25={other}", f"--noise-study={_noise(tmp_path / 'n.json')}"]
    )

    assert status == 1


def test_the_noise_check_reports_when_the_pooled_spread_exceeds_the_published_interval(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, {0.0: 2.0, 0.25: 2.0, 0.5: 2.0})

    check = result["primary"]["noise_check"]
    assert check["pooled_sd"] == pytest.approx(0.0031623, abs=1e-6)
    assert check["e2614_sd"] == pytest.approx(0.0031623, abs=1e-6)
    assert check["exceeds_e2614_upper_bound"] is False
