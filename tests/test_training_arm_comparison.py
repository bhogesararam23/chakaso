"""Tests for arm comparison: matching is proved before a difference is read, and the weighting is the scored one."""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from chakaso.training.arm_comparison import ArmComparisonError, compare_arms


def _report(
    *, prose: float, science: float, prefixes: Any = None, cap: int = 20_000_000
) -> dict[str, Any]:
    def block(bits: float, tokens: int, characters: int) -> dict[str, Any]:
        return {
            "tokens_scored": tokens,
            "metrics": {"bits_per_character": bits, "characters_scored": characters},
        }

    return {
        "corpus": {
            "fingerprint": "c" * 16,
            "split_fingerprint": "s" * 16,
            "split_seed": 7,
            "training_tokens_packed": cap,
        },
        "tokenizer": {"fingerprint": "t" * 16, "size": 4096},
        "transformer": {
            "parameter_count": 3_413_760,
            "architecture": {"d_model": 256, "num_layers": 4},
            "budget": {
                "seed": 7,
                "max_steps": 2400,
                "schedule": "cosine",
                "effective_batch_size": 32,
                "learning_rate": 0.0006,
                "training_group_prefixes": prefixes,
                "max_training_tokens": cap,
                "shuffle_training_units": True,
                "document_masking": False,
                "grad_accum_steps": 1,
            },
            "test": {
                "bits_per_character": (prose + science) / 2,
                "positions_scored": 1000,
                "characters_scored": 2000,
            },
            "test_partitions": {
                "prose-": block(prose, 600, 1200),
                "science-": block(science, 400, 800),
            },
        },
    }


def _three_arms() -> dict[str, dict[str, Any]]:
    return {
        "P": _report(prose=1.80, science=2.00, prefixes=["prose-"]),
        "T": _report(prose=1.84, science=1.90, prefixes=None),
        "H": _report(prose=1.82, science=2.10, prefixes=["prose-"], cap=10_000_000),
    }


def test_matched_arms_are_proved_matched_and_their_declared_differences_are_listed() -> None:
    result = compare_arms(_three_arms(), reference="P")

    assert result["all_conditions_matched"] is True
    assert set(result["declared_differences"]) == {
        "training_group_prefixes",
        "max_training_tokens",
        "training_tokens_packed",
    }
    assert result["declared_differences"]["training_group_prefixes"]["values"] == {
        "P": ["prose-"],
        "T": None,
        "H": ["prose-"],
    }


def test_a_condition_that_is_not_matched_is_named_and_flips_the_verdict() -> None:
    arms = _three_arms()
    arms["H"]["tokenizer"]["fingerprint"] = "x" * 16

    result = compare_arms(arms, reference="P")

    assert result["all_conditions_matched"] is False
    assert result["matched_conditions"]["tokenizer_fingerprint"]["equal"] is False
    assert result["matched_conditions"]["tokenizer_fingerprint"]["values"]["H"] == "x" * 16


def test_a_partition_scored_on_different_text_is_not_a_matched_condition() -> None:
    arms = _three_arms()
    arms["T"]["transformer"]["test_partitions"]["prose-"]["tokens_scored"] = 599

    assert compare_arms(arms, reference="P")["all_conditions_matched"] is False


def test_differences_are_arm_minus_reference_so_a_negative_number_is_a_better_arm() -> None:
    result = compare_arms(_three_arms(), reference="P")

    prose = result["partitions"]["prose-"]["difference_from_reference"]
    assert prose["T"] == pytest.approx(0.04)
    assert prose["H"] == pytest.approx(0.02)
    science = result["partitions"]["science-"]["difference_from_reference"]
    assert science["T"] == pytest.approx(-0.10)
    assert result["whole_pack_bits_per_character"]["difference_from_reference"][
        "T"
    ] == pytest.approx(-0.03)


def test_an_aggregate_is_weighted_by_scored_characters_not_by_an_equal_split() -> None:
    result = compare_arms(
        _three_arms(),
        reference="P",
        aggregate={"both": ["prose-", "science-"], "tech": ["science-"]},
    )

    both = result["aggregates"]["both"]
    expected_p = (1.80 * 1200 + 2.00 * 800) / 2000
    expected_t = (1.84 * 1200 + 1.90 * 800) / 2000
    assert both["values"]["P"] == pytest.approx(expected_p, abs=1e-6)
    assert both["difference_from_reference"]["T"] == pytest.approx(
        expected_t - expected_p, abs=1e-6
    )
    assert both["weighting"] == "metrics.characters_scored"
    assert result["aggregates"]["tech"]["values"]["T"] == pytest.approx(1.90)
    # An unweighted mean of the two partitions would be 1.90: the weights matter.
    assert both["values"]["P"] != pytest.approx((1.80 + 2.00) / 2)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda arms: {"P": arms["P"]}, "at least two arms"),
        (lambda arms: {**arms, "P": {"corpus": {}}}, "has no"),
    ],
)
def test_an_ill_posed_comparison_is_refused(mutate: Any, match: str) -> None:
    with pytest.raises(ArmComparisonError, match=match):
        compare_arms(mutate(_three_arms()), reference="P")


def test_an_unknown_reference_or_an_unscored_aggregate_partition_is_refused() -> None:
    arms = _three_arms()

    with pytest.raises(ArmComparisonError, match="not one of the arms"):
        compare_arms(arms, reference="Z")
    with pytest.raises(ArmComparisonError, match="did not score partition 'math-'"):
        compare_arms(arms, reference="P", aggregate={"g": ["math-"]})
    with pytest.raises(ArmComparisonError, match="names no partitions"):
        compare_arms(arms, reference="P", aggregate={"g": []})


def test_comparing_does_not_mutate_the_reports_it_reads() -> None:
    arms = _three_arms()
    before = copy.deepcopy(arms)

    compare_arms(arms, reference="P", aggregate={"g": ["prose-"]})

    assert arms == before
    assert math.isfinite(compare_arms(arms, reference="P")["partitions"]["prose-"]["values"]["P"])


def _script() -> Any:
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "compare_arms.py"
    spec = importlib.util.spec_from_file_location("compare_arms_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_script_rebuilds_the_comparison_from_report_files_and_signals_a_mismatch(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    arms = _three_arms()
    paths = {}
    for name, report in arms.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(report), encoding="utf-8")
    out = tmp_path / "results.json"
    arguments = [f"--arm={name}={path}" for name, path in paths.items()]

    status = _script().main(
        [*arguments, "--reference=P", "--aggregate=g=prose-,science-", f"--out={out}"]
    )

    assert status == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["all_conditions_matched"] is True
    assert written["aggregates"]["g"]["weighting"] == "metrics.characters_scored"
    assert "all conditions matched" in capsys.readouterr().err

    arms["H"]["tokenizer"]["fingerprint"] = "x" * 16
    paths["H"].write_text(json.dumps(arms["H"]), encoding="utf-8")
    assert _script().main([*arguments, "--reference=P"]) == 2
    assert "CONDITIONS NOT MATCHED" in capsys.readouterr().err
    assert _script().main(["--arm=bad", "--reference=P"]) == 1


def test_arms_that_differ_only_in_share_list_the_share_as_the_declared_difference() -> None:
    arms = {
        "low": _report(prose=1.90, science=1.95),
        "high": _report(prose=1.82, science=2.05),
    }
    for name, share in (("low", 0.25), ("high", 0.75)):
        budget = arms[name]["transformer"]["budget"]
        budget["share_of_tokens"] = share
        budget["share_group_prefix"] = "prose-"

    result = compare_arms(arms, reference="low")

    assert result["all_conditions_matched"] is True
    assert result["declared_differences"]["share_of_tokens"]["values"] == {
        "low": 0.25,
        "high": 0.75,
    }
    assert "share_group_prefix" not in result["declared_differences"]
