"""The E2630 seed-spread reader: its Y2 outcomes at their boundaries, and the refusals that keep a draw honest.

Expected numbers are computed by hand beside each test. A mean of four differences with a 0.0028 deviation has a
standard error of 0.0014 and, at 3 degrees of freedom, a half-width of 3.182 x 0.0014 = 0.0045 — which is the whole
difference between "the interval clears zero" and "it does not", so the arithmetic has to be checkable without the
module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_CONTROL_O1 = 1.90
_CONTROL_WHOLE = 2.0554


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(
    path: Path,
    *,
    partition: float,
    whole: float,
    treatment: bool,
    steps: int = 2400,
    training_documents: int = 2756,
) -> dict[str, Any]:
    tokenizer: dict[str, Any] = {
        "fingerprint": "n" * 16 if treatment else "t" * 16,
        "characters_per_token_test": 2.93 if treatment else 2.10,
    }
    corpus: dict[str, Any] = {
        "fingerprint": "c" * 16,
        "split_fingerprint": "s" * 16,
        "test_documents_packed": 34,
        "training_documents_packed": training_documents,
    }
    if treatment:
        tokenizer["pre_tokenizer"] = "space-prefixed-word|space-prefixed-symbol|whitespace-run"
        corpus["evaluation_selection"] = {"tokenizer_fingerprint": "t" * 16, "test_documents": 34}
    report = {
        "corpus": corpus,
        "tokenizer": tokenizer,
        "transformer": {
            "parameter_count": 3_413_760,
            "budget": {"max_steps": steps, "batch_size": 32, "wall_clock_s": 5000.0},
            "test": {"bits_per_character": whole},
            "test_partitions": {
                prefix: {"documents_packed": 34, "metrics": {"bits_per_character": partition}}
                for prefix in _PARTITIONS
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return report


def _aggregate(seeds: list[int]) -> dict[str, Any]:
    """E2614's study aggregate over the controls, as its committed file shapes it: a metrics block per seed."""
    return {
        "per_run": {
            str(seed): {
                "metrics": {
                    "transformer_bits_per_character": _CONTROL_WHOLE,
                    **{
                        f"partition_{prefix}_bits_per_character": _CONTROL_O1
                        for prefix in _PARTITIONS
                    },
                }
            }
            for seed in seeds
        }
    }


def _write(tmp_path: Path, name: str, **kwargs: Any) -> str:
    _report(tmp_path / name, **kwargs)
    return str(tmp_path / name)


def _run(tmp_path: Path, deltas: dict[int, tuple[float, float]]) -> dict[str, Any]:
    """Two-file-per-seed fixtures on disk read through ``main``, so the CLI wiring is covered with the arithmetic."""
    argv = []
    for seed, (delta_o1, delta_whole) in deltas.items():
        argv.append(
            f"--treatment={seed}="
            + _write(
                tmp_path,
                f"treatment-{seed}.json",
                partition=_CONTROL_O1 - delta_o1,
                whole=_CONTROL_WHOLE - delta_whole,
                treatment=True,
            )
        )
        argv.append(
            f"--control={seed}="
            + _write(
                tmp_path,
                f"control-{seed}.json",
                partition=_CONTROL_O1,
                whole=_CONTROL_WHOLE,
                treatment=False,
            )
        )
    spread = tmp_path / "spread.json"
    spread.write_text(json.dumps(_aggregate(sorted(deltas))), encoding="utf-8")
    out = tmp_path / "out.json"
    argv.append(f"--spread={spread}")
    argv.append(f"--out={out}")
    assert _script("analyse_tokenizer_seed_spread").main(argv) == 0
    return json.loads(out.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_four_material_draws_with_a_tight_interval_are_y2_a(tmp_path: Path) -> None:
    # Mean 0.026075, s = 0.002778, SE = 0.001389, t(3) = 3.182 -> half-width 0.004422, interval [0.02165, 0.03050]:
    # clears zero, the mean is above 0.02, n reaches the declared adoption gate, and the whole-pack mean 0.031375
    # clears its own 0.00933.
    result = _run(
        tmp_path,
        {7: (0.0229, 0.0286), 11: (0.0294, 0.0349), 13: (0.0250, 0.0300), 17: (0.0270, 0.0320)},
    )
    assert result["outcome"] == "Y2-a"
    assert result["design"]["draws"] == 4
    assert result["o1"]["mean"] == pytest.approx(0.026075, abs=1e-6)
    assert result["o1"]["interval_95"] == pytest.approx((0.021653, 0.030497), abs=1e-5)
    assert result["o1"]["degrees_of_freedom"] == 3
    assert result["o1"]["t_critical"] == pytest.approx(3.182446, abs=5e-4)
    assert result["o1"]["positive_draws"] == 4
    # Four draws, all one way: P(X >= 4) for a fair coin is C(4,4)/2^4.
    assert result["o1"]["sign_test_p_value"] == pytest.approx(1 / 16)
    assert result["whole_pack"]["excludes_zero"] is True
    assert [draw["seed"] for draw in result["draw"]] == [7, 11, 13, 17]
    assert result["design"]["seeds"] == [7, 11, 13, 17]


def test_three_positive_draws_cannot_meet_a_four_seed_gate(tmp_path: Path) -> None:
    # Same deltas, one seed short: everything else clears, so the outcome is the record's "direction survives,
    # magnitude unresolved" rather than an adoption claim.
    result = _run(tmp_path, {11: (0.0229, 0.0286), 13: (0.0250, 0.0300), 17: (0.0270, 0.0320)})
    assert result["outcome"] == "Y2-b"
    assert result["design"]["draws"] == 3
    assert result["o1"]["excludes_zero"] is True


def test_a_draw_that_reverses_beyond_the_tie_is_y2_c(tmp_path: Path) -> None:
    # One seed where the treatment is 0.01 worse on O1: -0.01 < -0.0062 is the record's falsifier, whatever the
    # other three say — and their mean 0.017325 still sits inside the band, which is the second c-clause.
    result = _run(
        tmp_path,
        {7: (0.0229, 0.0286), 11: (0.0294, 0.0349), 13: (-0.0100, -0.0120), 17: (0.0270, 0.0320)},
    )
    assert result["outcome"] == "Y2-c"
    assert result["o1"]["positive_draws"] == 3


def test_a_mean_inside_the_band_is_y2_c_even_when_every_draw_is_positive(tmp_path: Path) -> None:
    # All four positive but averaging 0.0125, inside the 0.02 separable margin: the gain is not established, and
    # the sign of the draws is not what the record's rule reads.
    result = _run(tmp_path, {7: (0.01, 0.02), 11: (0.01, 0.02), 13: (0.02, 0.02), 17: (0.01, 0.02)})
    assert result["outcome"] == "Y2-c"
    assert result["o1"]["mean"] == pytest.approx(0.0125, abs=1e-9)


def test_positive_draws_with_a_loose_spread_are_y2_b(tmp_path: Path) -> None:
    # Mean 0.022875 is material and every draw is positive, but s = 0.0257 gives SE 0.0128 and a half-width of
    # 0.0409: the interval straddles zero, so the magnitude is unresolved and the record's rule makes that b, not
    # an adoption claim.
    result = _run(
        tmp_path,
        {7: (0.0005, 0.0005), 11: (0.0600, 0.0600), 13: (0.0100, 0.0100), 17: (0.0210, 0.0210)},
    )
    assert result["outcome"] == "Y2-b"
    assert result["o1"]["mean"] == pytest.approx(0.022875, abs=1e-6)
    assert result["o1"]["excludes_zero"] is False


def test_a_draw_the_wrong_side_of_the_tie_but_inside_the_band_is_y2_d(tmp_path: Path) -> None:
    # -0.004 is not the record's falsifier (it does not pass -0.0062) and the mean 0.02275 is material, but "every
    # draw positive" is false, so neither b nor c holds and the reading is the declared anything-else.
    result = _run(
        tmp_path,
        {7: (-0.0040, -0.0040), 11: (0.0300, 0.0320), 13: (0.0320, 0.0340), 17: (0.0330, 0.0350)},
    )
    assert result["outcome"] == "Y2-d"
    assert result["o1"]["mean"] == pytest.approx(0.02275, abs=1e-6)
    assert result["o1"]["positive_draws"] == 3


def test_the_treatment_and_control_spreads_are_both_reported(tmp_path: Path) -> None:
    # E2630's second question: the treatment's own seed-to-seed spread beside the control's, neither of which the
    # single-pair reader could state.
    result = _run(
        tmp_path,
        {7: (0.0229, 0.0286), 11: (0.0294, 0.0349), 13: (0.0250, 0.0300), 17: (0.0270, 0.0320)},
    )
    treatment = [draw["o1"]["treatment"] for draw in result["draw"]]
    assert result["control_spread"]["o1"]["values"] == pytest.approx([_CONTROL_O1] * 4)
    assert treatment == pytest.approx([1.8771, 1.8706, 1.8750, 1.8730], abs=1e-9)
    assert result["treatment_spread"]["o1"]["sample_stdev"] == pytest.approx(0.002778, abs=1e-5)
    assert result["treatment_spread"]["prose-"]["sample_stdev"] == pytest.approx(0.002778, abs=1e-5)
    assert result["control_spread"]["o1"]["sample_stdev"] == pytest.approx(0.0)


def test_a_treatment_whose_updates_differ_from_its_control_names_the_seed(tmp_path: Path) -> None:
    module = _script("analyse_tokenizer_seed_spread")
    control = _report(
        Path(tmp_path / "c11.json"), partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False
    )
    treatment = _report(
        Path(tmp_path / "t11.json"), partition=1.87, whole=2.02, treatment=True, steps=2200
    )
    good_control = _report(
        Path(tmp_path / "c13.json"), partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False
    )
    good_treatment = _report(
        Path(tmp_path / "t13.json"), partition=1.87, whole=2.02, treatment=True
    )
    with pytest.raises(
        module.AnalysisError, match="seed 11: updates: treatment 2200, control 2400"
    ):
        module.read(
            [(11, treatment, control), (13, good_treatment, good_control)],
            spread=_aggregate([11, 13]),
        )


def test_a_control_extract_that_disagrees_with_the_aggregate_is_refused(tmp_path: Path) -> None:
    module = _script("analyse_tokenizer_seed_spread")
    control = _report(
        Path(tmp_path / "c11.json"), partition=_CONTROL_O1, whole=2.0600, treatment=False
    )
    treatment = _report(Path(tmp_path / "t11.json"), partition=1.87, whole=2.02, treatment=True)
    other_control = _report(
        Path(tmp_path / "c13.json"), partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False
    )
    other_treatment = _report(
        Path(tmp_path / "t13.json"), partition=1.87, whole=2.02, treatment=True
    )
    # The extract says the seed-11 control scored 2.0600 on the pack; E2614's aggregate says 2.0554. One of the two
    # files is not the run the record cited, and the reader refuses to choose.
    with pytest.raises(module.AnalysisError, match=r"seed 11, whole pack: extract 2\.06"):
        module.read(
            [(11, treatment, control), (13, other_treatment, other_control)],
            spread=_aggregate([11, 13]),
        )


def test_a_seed_named_twice_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _script("analyse_tokenizer_seed_spread")
    spread = tmp_path / "spread.json"
    spread.write_text(json.dumps(_aggregate([11])), encoding="utf-8")
    status = module.main(
        [
            f"--spread={spread}",
            f"--treatment=11={_write(tmp_path, 'a.json', partition=1.87, whole=2.02, treatment=True)}",
            f"--treatment=11={_write(tmp_path, 'b.json', partition=1.88, whole=2.03, treatment=True)}",
            f"--control=11={_write(tmp_path, 'c.json', partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False)}",
        ]
    )
    assert status == 1
    assert "names seed 11 twice" in capsys.readouterr().err


def test_a_treatment_without_its_control_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script("analyse_tokenizer_seed_spread")
    spread = tmp_path / "spread.json"
    spread.write_text(json.dumps(_aggregate([7, 11])), encoding="utf-8")
    status = module.main(
        [
            f"--spread={spread}",
            f"--treatment=7={_write(tmp_path, 'a.json', partition=1.87, whole=2.02, treatment=True)}",
            f"--treatment=11={_write(tmp_path, 'b.json', partition=1.88, whole=2.03, treatment=True)}",
            f"--control=7={_write(tmp_path, 'c.json', partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False)}",
        ]
    )
    assert status == 1
    assert "treatments [7, 11], controls [7]" in capsys.readouterr().err


def test_one_draw_is_refused_because_a_spread_needs_two(tmp_path: Path) -> None:
    module = _script("analyse_tokenizer_seed_spread")
    control = _report(
        Path(tmp_path / "c.json"), partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False
    )
    treatment = _report(Path(tmp_path / "t.json"), partition=1.87, whole=2.02, treatment=True)
    with pytest.raises(module.AnalysisError, match="at least two draws"):
        module.read([(11, treatment, control)], spread=_aggregate([11]))


def test_a_missing_seed_in_the_aggregate_is_named_with_the_seeds_it_has(tmp_path: Path) -> None:
    module = _script("analyse_tokenizer_seed_spread")
    control = _report(
        Path(tmp_path / "c7.json"), partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False
    )
    treatment = _report(Path(tmp_path / "t7.json"), partition=1.87, whole=2.02, treatment=True)
    other_control = _report(
        Path(tmp_path / "c13.json"), partition=_CONTROL_O1, whole=_CONTROL_WHOLE, treatment=False
    )
    other_treatment = _report(
        Path(tmp_path / "t13.json"), partition=1.87, whole=2.02, treatment=True
    )
    with pytest.raises(module.AnalysisError, match=r"seed 7 is not in the control aggregate"):
        module.read(
            [(7, treatment, control), (13, other_treatment, other_control)],
            spread=_aggregate([11, 13]),
        )


def test_the_summary_prints_the_outcome_and_the_mean(tmp_path: Path) -> None:
    module = _script("analyse_tokenizer_seed_spread")
    result = _run(
        tmp_path,
        {7: (0.0229, 0.0286), 11: (0.0294, 0.0349), 13: (0.0250, 0.0300), 17: (0.0270, 0.0320)},
    )
    text = module.summarise(result)
    assert text.startswith("outcome Y2-a over 4 same-host draws")
    assert "O1: mean +0.026075" in text
    assert "4/4 positive" in text
