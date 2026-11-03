"""Tests for the curve-sensitivity analysis script.

The script is dependency-free until it opens a checkpoint — the statistics come from
:mod:`chakaso.training.probe_sensitivity`, and the artifact scoring is behind a function the tests do not call —
so these run in the core matrix and every fixture is a dict written here, not a trained model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyse_probe_sensitivity.py"
_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyse_probe_sensitivity_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _report(steps: tuple[int, ...] = (400, 800), endpoint_loss: float = 2.0) -> dict[str, Any]:
    return {
        "experiment": "E2626",
        "transformer": {
            "test": {
                "bits_per_character": endpoint_loss,
                "positions_scored": 254976,
                "characters_scored": 535282,
            },
            "curve_artifacts": [
                {"step": step, "path": f"E2626-step{step}-decoder.pt"} for step in steps
            ],
        },
    }


def _scored(**overrides: float) -> dict[str, dict[str, Any]]:
    """Three rungs and the untrained floor, one bits-per-character number each."""
    values = {"step0": 6.0, "step400": 2.4, "step800": overrides.get("step800", 2.0)}
    table: dict[str, dict[str, Any]] = {}
    for name, whole in values.items():
        table[name] = {
            "test": {
                "bits_per_character": whole,
                "positions_scored": 254976,
                "characters_scored": 535282,
            },
            "test_partitions": {
                prefix: {"metrics": {"bits_per_character": whole + index / 100}}
                for index, prefix in enumerate(_PARTITIONS)
            },
        }
    return table


def _pairs(passes: dict[str, int], total: int = 20) -> dict[str, Any]:
    """A minimal-pair result: each named model passes its first ``passes`` items, one phenomenon each."""
    models = {}
    for name, count in passes.items():
        models[name] = {
            "result": {
                "outcomes": [
                    [
                        f"item{index}",
                        "agreement" if index % 2 else "order",
                        1.0 if index < count else 0.0,
                    ]
                    for index in range(total)
                ]
            }
        }
    return {"models": models}


def _continuations(passes: dict[str, int], total: int = 20) -> dict[str, Any]:
    models = {
        name: {
            "result": {
                "outcomes": {
                    reading: [
                        [f"item{index}", 1.0 if index < count else 0.0] for index in range(total)
                    ]
                    for reading in (
                        "pmi_easy",
                        "pmi_hard",
                        "conditional_easy",
                        "conditional_hard",
                        "blind_easy",
                        "blind_hard",
                    )
                }
            }
        }
        for name, count in passes.items()
    }
    return {"models": models}


def test_the_curve_names_rung_zero_and_every_rung_the_run_wrote() -> None:
    module = _script()

    names = module.rung_names(_report((400, 800, 200)))

    assert names["step0"] == "untrained"
    assert list(names) == ["step0", "step200", "step400", "step800"]


def test_a_report_that_wrote_no_rungs_is_refused_rather_than_read_as_one() -> None:
    module = _script()
    report = _report()
    report["transformer"]["curve_artifacts"] = []

    with pytest.raises(module.AnalysisError, match="no rungs"):
        module.rung_names(report)


def test_the_criterion_table_carries_the_pack_and_the_equal_weight_mean() -> None:
    module = _script()

    table = module.criterion_table(_scored())

    assert table["step0"]["whole_pack_bits_per_character"] == 6.0
    # partitions are 6.00, 6.01, 6.02, 6.03, 6.04 -> mean 6.02
    assert table["step0"]["o1_equal_weight_mean"] == pytest.approx(6.02)


def test_a_partition_with_no_scored_metrics_is_refused() -> None:
    module = _script()
    scored = _scored()
    del scored["step400"]["test_partitions"]["medicine-"]

    with pytest.raises(module.AnalysisError, match="no scored partition"):
        module.criterion_table(scored)


def test_the_endpoint_must_be_the_model_the_run_reported() -> None:
    module = _script()
    scored = _scored()

    module.check_endpoint_matches_the_report(scored, _report(), endpoint="step800")
    scored["step800"]["test"]["bits_per_character"] = 2.0001

    with pytest.raises(module.AnalysisError, match="did not reproduce its own endpoint"):
        module.check_endpoint_matches_the_report(scored, _report(), endpoint="step800")


def test_the_probe_scripts_control_name_becomes_rung_zero() -> None:
    module = _script()

    table = module.pair_outcomes(_pairs({"untrained-control": 10, "step400": 18}), phenomena=None)

    assert sorted(table) == ["step0", "step400"]
    assert len(table["step400"]) == 20


def test_only_the_declared_usable_phenomena_enter_the_usable_table() -> None:
    module = _script()
    result = _pairs({"step400": 20})

    usable = module.pair_outcomes(result, phenomena=["agreement"])

    # Half the fixture's items are "agreement"; the others must not be counted as pairs tested.
    assert len(usable["step400"]) == 10
    assert all(value == 1.0 for value in usable["step400"].values())


def test_a_pair_result_without_per_item_outcomes_is_refused_not_read_as_a_mean() -> None:
    module = _script()
    result = _pairs({"step400": 20})
    result["models"]["step400"]["result"] = {"accuracy": 1.0}

    with pytest.raises(module.AnalysisError, match="before the probe recorded them"):
        module.pair_outcomes(result, phenomena=None)


def test_a_model_missing_from_one_instrument_stops_the_analysis() -> None:
    module = _script()

    with pytest.raises(module.AnalysisError, match="a rung without a score"):
        module.check_same_models(
            {"step0": 6.0, "step400": 2.4, "step800": 2.0},
            {"probe": {"step0": {}, "step400": {}}},
        )


def test_models_beyond_the_curve_are_reported_rather_than_silently_dropped() -> None:
    """E2627 scores a dozen further arms beside the rungs; the criterion table defines the population."""
    module = _script()

    extras = module.check_same_models(
        {"step0": 6.0, "step400": 2.4},
        {"probe": {"step0": {}, "step400": {}, "E2612-T": {}, "E2614-s11": {}}},
    )

    assert extras == ["E2612-T", "E2614-s11"]


def test_each_same_region_kind_becomes_its_own_instrument() -> None:
    module = _script()
    result = _continuations({"untrained-control": 10, "step400": 19, "step800": 20})
    # The same-region runner names its readings pmi_prior_span and siblings, not pmi_easy.
    for block in result["models"].values():
        block["result"]["outcomes"] = {
            f"pmi_{kind}": list(block["result"]["outcomes"]["pmi_easy"])
            for kind in ("prior_span", "next_span", "permuted", "reversed")
        }

    instruments = module.context_outcomes(result)

    assert set(instruments) == {
        "context_pmi_prior_span",
        "context_pmi_next_span",
        "context_pmi_permuted",
        "context_pmi_reversed",
    }
    assert sorted(instruments["context_pmi_permuted"]) == ["step0", "step400", "step800"]


def test_a_kind_the_runner_did_not_score_is_refused_not_skipped() -> None:
    module = _script()
    result = _continuations({"step400": 20})
    for block in result["models"].values():
        block["result"]["outcomes"] = {
            "pmi_prior_span": list(block["result"]["outcomes"]["pmi_easy"])
        }

    with pytest.raises(
        module.AnalysisError, match="no per-item outcomes for the same-region reading"
    ):
        module.context_outcomes(result)


def test_read_produces_a_verdict_per_instrument_from_the_assembled_tables() -> None:
    module = _script()

    result = module.read(
        report=_report(),
        scored=_scored(),
        pairs_result=_pairs({"untrained-control": 1, "step400": 12, "step800": 20}),
        continuations_result=_continuations(
            {"untrained-control": 10, "step400": 19, "step800": 20}
        ),
        usable_phenomena=["agreement", "order"],
        materiality=0.02,
        endpoint="step800",
    )

    assert sorted(result["instruments"]) == [
        "continuation_pmi_easy",
        "continuation_pmi_hard",
        "minimal_pairs_all_800",
        "minimal_pairs_usable",
    ]
    assert result["endpoint"]["rung"] == "step800"
    assert result["endpoint"]["bits_per_character"] == 2.0
    assert len(result["models"]) == 3
    # 3 models -> 3 pairs, and the untrained floor is 6.0 against 2.4/2.0: every pair is material here.
    assert all(row["material"] for row in result["pairs"])
    assert result["instruments"]["minimal_pairs_all_800"]["verdict"] in {"S-a", "S-b", "S-c", "S-d"}


def test_the_summary_names_the_gaps_rather_than_only_the_label() -> None:
    module = _script()
    result = module.read(
        report=_report(),
        scored=_scored(),
        pairs_result=_pairs({"untrained-control": 1, "step400": 12, "step800": 20}),
        continuations_result=_continuations(
            {"untrained-control": 10, "step400": 19, "step800": 20}
        ),
        usable_phenomena=["agreement", "order"],
        materiality=0.02,
        endpoint="step800",
    )

    text = module.summarise(result)

    assert "materiality 0.02" in text
    for name in result["instruments"]:
        assert name in text
