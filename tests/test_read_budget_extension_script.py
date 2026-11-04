"""A budget-extension pair, read offline: what it refuses, what it derives, and which letter is its own.

Every report here is a fixture built by :func:`_report`, so nothing in this file trains, scores or loads a checkpoint.
The properties that matter are the ones a run cannot be trusted without: a pair is refused unless it is the same run at
two budgets; the band a verdict is read on is derived from a committed record rather than typed into the tool; and a
declared outcome vocabulary is checked against the facts the tool actually computes, which is what turns a mis-typed
condition from a rule that never fires into a refusal (K-034).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "read_budget_extension.py"
_ROOT = Path(__file__).resolve().parent.parent
_VOCABULARY = _ROOT / "configs" / "analyses" / "budget-extension-e2636-v1.json"
_BANDS_RECORD = _ROOT / "experiments" / "E2630" / "results.json"
_BANDS = json.loads(_BANDS_RECORD.read_text(encoding="utf-8"))

_REGISTERS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("read_budget_extension_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _curve(points: list[tuple[int, float]]) -> list[dict[str, Any]]:
    return [
        {"step": step, "validation_loss": value, "train_loss": value - 0.2, "gradient_norm": 0.5}
        for step, value in points
    ]


def _falling(steps: int) -> list[dict[str, Any]]:
    """A curve that only goes down — what every arm this series has trained reports at its ceiling."""
    return _curve(
        [(step, 8.3 - 0.4 * index) for index, step in enumerate(range(0, steps + 1, 400))]
    )


def _turned(steps: int, bottom_at: int) -> list[dict[str, Any]]:
    return _curve(
        [
            (
                step,
                8.3 - 0.4 * index
                if step <= bottom_at
                else 4.0 + 0.05 * ((step - bottom_at) // 400),
            )
            for index, step in enumerate(range(0, steps + 1, 400))
        ]
    )


def _report(
    *,
    max_steps: int = 2400,
    bits: float = 2.0,
    curve: list[dict[str, Any]] | None = None,
    best_step: int | None = None,
    completion: str = "budget_ceiling",
) -> dict[str, Any]:
    """A report shaped like the ones `chakaso experiment model` writes, small enough to read in a test.

    Every field the reader compares is present and defaults to the same value across calls, so a test that wants a pair
    to differ names the one field it moves.
    """
    return {
        "corpus": {
            "fingerprint": "312571f9fa8351a8",
            "split_fingerprint": "adf650435c3a9068",
            "split_seed": 7,
            "training_tokens_packed": 14033424,
            "test_documents_packed": 34,
        },
        "evaluation_sample": {"documents": 34, "works": 25},
        "tokenizer": {
            "fingerprint": "4a47f0281c81a9e5",
            "size": 4096,
            "pre_tokenizer": "space-prefixed-word|space-prefixed-symbol|whitespace-run",
        },
        "transformer": {
            "parameter_count": 3413760,
            "architecture": {"d_model": 256, "num_layers": 4},
            "budget": {
                "seed": 7,
                "max_steps": max_steps,
                "schedule": "cosine",
                "decay_to": 0.1,
                "warmup_steps": 200,
                "learning_rate": 0.0006,
                "weight_decay": 0.01,
                "optimizer": {"name": "AdamW", "betas": [0.9, 0.999]},
                "effective_batch_size": 32,
                "eval_every": 400,
                "max_training_tokens": 14033424,
                "patience": 99,
                "gradient_clip_norm": 1.0,
                "best_step": max_steps if best_step is None else best_step,
                "completion": completion,
            },
            "curve": _falling(max_steps) if curve is None else curve,
            "test": {
                "bits_per_character": bits + 0.05,
                "positions_scored": 254976,
                "characters_scored": 535391,
            },
            "test_partitions": {
                name: {"metrics": {"bits_per_character": bits + index * 0.1}}
                for index, name in enumerate(_REGISTERS)
            },
        },
    }


def _written(tmp_path: Path, name: str, report: dict[str, Any]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _read(tmp_path: Path, pairs: dict[str, tuple[float, float]], **kwargs: Any) -> dict[str, Any]:
    """Read a declared pair set: seed to its (short-arm O1, long-arm O1), two budgets per seed."""
    module = _script()
    long_curve = kwargs.pop("long_curve", None)
    long_best_step = kwargs.pop("long_best_step", None)
    shorts = {
        seed: _written(tmp_path, f"short-{seed}.json", _report(max_steps=2400, bits=short))
        for seed, (short, _long) in pairs.items()
    }
    longs = {
        seed: _written(
            tmp_path,
            f"long-{seed}.json",
            _report(
                max_steps=7200,
                bits=long,
                curve=long_curve or _falling(7200),
                best_step=long_best_step,
            ),
        )
        for seed, (_short, long) in pairs.items()
    }
    return module.read(shorts, longs, bands_source=_BANDS, **kwargs)


# ------------------------------------------------------------------ the band


def test_the_band_comes_from_the_committed_record_and_not_from_a_constant() -> None:
    module = _script()
    bands = module.resolve_bands(_BANDS)
    assert bands["band_source_sigma_o1"] == pytest.approx(0.001595, abs=1e-6)
    # 1.96*sqrt(2)*0.001595 = 0.004418, rounded up to the next 0.005 by E2614's own replacement rule.
    assert bands["derived_band_unrounded"] == pytest.approx(0.00442, abs=1e-6)
    assert bands["o1_band"] == 0.005
    assert bands["inherited_o1_band"] == 0.02
    assert bands["inherited_o1_tie"] == pytest.approx(0.0062)


def test_the_derived_band_is_printed_beside_the_inherited_one_rather_than_replacing_it() -> None:
    module = _script()
    bands = module.resolve_bands(_BANDS)
    assert "0.250" in bands["relationship"]
    assert "whole pack" in bands["whole_pack_note"]


def test_a_bands_source_without_a_measured_spread_is_refused_by_name() -> None:
    module = _script()
    with pytest.raises(module.BudgetReadingError, match=r"no treatment_spread\.o1"):
        module.resolve_bands({"margins": {"o1_separable": 0.02}})
    with pytest.raises(module.BudgetReadingError, match=r"no margins\.o1_separable"):
        module.resolve_bands({"treatment_spread": {"o1": {"sample_stdev": 0.0016}}})


# ------------------------------------------------------------------ matchedness


def test_a_matched_pair_reads_a_difference_and_names_both_bands(tmp_path: Path) -> None:
    report = _read(tmp_path, {"7": (2.0, 1.95), "13": (2.0, 1.95)})
    row = report["per_seed"][0]
    assert row["delta_o1"] == pytest.approx(0.05)
    assert row["delta_whole_pack"] == pytest.approx(0.05)
    assert row["clears_derived_band"] is True
    assert row["clears_inherited_band"] is True
    assert row["partitions"]["prose-"]["delta"] == pytest.approx(0.05)
    assert report["design"]["bands"]["o1_band"] == 0.005
    assert report["outcome"] is None
    assert report["statistics_only"] is True


def test_a_pair_that_differs_in_a_matched_condition_is_refused_by_that_condition(
    tmp_path: Path,
) -> None:
    module = _script()
    long_report = _report(max_steps=7200)
    long_report["tokenizer"]["fingerprint"] = "0000000000000000"
    with pytest.raises(module.BudgetReadingError, match=r"tokenizer\.fingerprint"):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {"7": _written(tmp_path, "l.json", long_report)},
            bands_source=_BANDS,
        )


def test_a_split_that_moved_is_refused_as_a_protocol_change(tmp_path: Path) -> None:
    module = _script()
    long_report = _report(max_steps=7200)
    long_report["corpus"]["split_fingerprint"] = "0000000000000000"
    with pytest.raises(module.BudgetReadingError, match="split_fingerprint"):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {"7": _written(tmp_path, "l.json", long_report)},
            bands_source=_BANDS,
        )


def test_the_evaluation_pack_has_to_be_the_same_text_in_both_arms(tmp_path: Path) -> None:
    module = _script()
    long_report = _report(max_steps=7200)
    long_report["evaluation_sample"] = {"documents": 30, "works": 22}
    with pytest.raises(module.BudgetReadingError, match="evaluation_sample"):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {"7": _written(tmp_path, "l.json", long_report)},
            bands_source=_BANDS,
        )


@pytest.mark.parametrize(
    ("long_steps", "pattern"),
    [
        (2400, "ran 2400 updates against"),
        (4800, "not the declared"),
        (1200, "ran 1200 updates against"),
    ],
)
def test_a_budget_that_is_not_the_declared_extension_is_refused(
    tmp_path: Path, long_steps: int, pattern: str
) -> None:
    module = _script()
    with pytest.raises(module.BudgetReadingError, match=pattern):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {
                "7": _written(
                    tmp_path, "l.json", _report(max_steps=long_steps, curve=_falling(long_steps))
                )
            },
            bands_source=_BANDS,
        )


def test_an_arm_that_stopped_for_a_reason_other_than_its_ceiling_is_refused(tmp_path: Path) -> None:
    module = _script()
    points = _falling(7200)[:-1]
    with pytest.raises(module.BudgetReadingError, match="finished by 'converged'"):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {
                "7": _written(
                    tmp_path,
                    "l.json",
                    _report(
                        max_steps=7200,
                        curve=points,
                        completion="converged",
                        best_step=points[-1]["step"],
                    ),
                )
            },
            bands_source=_BANDS,
        )


def test_a_curve_that_stops_short_of_its_own_ceiling_is_refused(tmp_path: Path) -> None:
    module = _script()
    with pytest.raises(module.BudgetReadingError, match="ends at step 6800"):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {"7": _written(tmp_path, "l.json", _report(max_steps=7200, curve=_falling(6800)))},
            bands_source=_BANDS,
        )


def test_a_report_whose_best_step_disagrees_with_the_curves_own_argmin_is_refused(
    tmp_path: Path,
) -> None:
    module = _script()
    with pytest.raises(module.BudgetReadingError, match="best_step is 4800"):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {"7": _written(tmp_path, "l.json", _report(max_steps=7200, best_step=4800))},
            bands_source=_BANDS,
        )


def test_a_seed_present_on_one_side_only_is_refused_before_anything_is_read(tmp_path: Path) -> None:
    module = _script()
    with pytest.raises(module.BudgetReadingError, match="short arms cover"):
        module.read(
            {
                "7": _written(tmp_path, "s.json", _report(max_steps=2400)),
                "13": _written(tmp_path, "s13.json", _report(max_steps=2400)),
            },
            {"7": _written(tmp_path, "l.json", _report(max_steps=7200))},
            bands_source=_BANDS,
        )


def test_one_completed_pair_is_refused_because_a_mean_needs_two_draws(tmp_path: Path) -> None:
    """A shortfall is reported as a shortfall: the tool will not turn one draw into an averaged verdict."""
    module = _script()
    with pytest.raises(module.BudgetReadingError, match="one draw has no spread"):
        module.read(
            {"7": _written(tmp_path, "s.json", _report(max_steps=2400))},
            {"7": _written(tmp_path, "l.json", _report(max_steps=7200))},
            bands_source=_BANDS,
        )


# ------------------------------------------------------------------ the curve


def test_a_curve_still_falling_at_the_end_reports_no_turn() -> None:
    module = _script()
    curve = module.validation_curve_reading(_falling(7200), where="long arm")
    assert curve["turned_up_inside_horizon"] is False
    assert curve["validation_min_step"] == 7200
    assert curve["tail_slope_nats_per_100_updates"] < 0


def test_a_curve_that_bottomed_out_reports_where_and_that_it_ran_back_up() -> None:
    module = _script()
    curve = module.validation_curve_reading(_turned(7200, bottom_at=4800), where="long arm")
    assert curve["turned_up_inside_horizon"] is True
    assert curve["validation_min_step"] == 4800
    assert curve["final_above_min_nats_per_token"] > 0
    assert curve["tail_slope_nats_per_100_updates"] > 0


def test_two_points_cannot_show_a_minimum_inside_a_horizon() -> None:
    module = _script()
    with pytest.raises(module.BudgetReadingError, match="cannot show a minimum"):
        module.validation_curve_reading(_curve([(0, 8.3), (400, 4.6)]), where="short arm")


# ------------------------------------------------------------------ the aggregate and the letter


def test_the_aggregate_is_a_students_t_interval_over_the_draws_and_says_which_way_favours_what(
    tmp_path: Path,
) -> None:
    report = _read(tmp_path, {"7": (2.0, 1.95), "13": (2.01, 1.96), "17": (2.02, 1.97)})
    aggregate = report["aggregate"]
    assert aggregate["o1"]["mean"] == pytest.approx(0.05)
    assert aggregate["o1"]["degrees_of_freedom"] == 2
    assert aggregate["o1"]["t_critical"] == pytest.approx(4.303, abs=0.01)
    assert aggregate["o1"]["positive_draws"] == 3
    assert aggregate["o1"]["sign_test_p_value"] == pytest.approx(0.125)
    assert "short arm minus long arm" in aggregate["unit"]


def test_a_letter_is_printed_only_when_the_record_handed_one_over(tmp_path: Path) -> None:
    report = _read(tmp_path, {"7": (2.0, 1.95), "13": (2.0, 1.95)}, vocabulary_path=_VOCABULARY)
    verdict = report["outcome"]
    assert verdict is not None
    assert verdict["letter"] == "C-b"
    assert verdict["vocabulary"] == "E2636 budget-extension outcomes (C-a to C-d)"
    assert verdict["facts"]["every_long_arm_still_falling"] is True
    assert verdict["facts"]["mean_delta_o1"] == pytest.approx(0.05)
    assert "C-d" in verdict["also_matched"]


def test_the_declared_vocabulary_names_only_facts_the_tool_computes() -> None:
    """The K-034 property, checked against the committed declaration rather than a fixture of it."""
    module = _script()
    vocabulary = module.load_vocabulary(_VOCABULARY, known_facts=sorted(module.KNOWN_FACTS))
    assert set(vocabulary.letters) == {"C-a", "C-b", "C-c", "C-d"}
    assert vocabulary.exclusive is False


def test_a_vocabulary_that_names_a_fact_the_tool_does_not_compute_is_refused() -> None:
    from chakaso.training.outcome_rules import OutcomeRuleError

    module = _script()
    document = json.loads(_VOCABULARY.read_text(encoding="utf-8"))
    document["outcomes"][0]["when"].append({"fact": "mean_delta_o2", "at_least": 0.005})
    with pytest.raises(OutcomeRuleError, match="does not compute"):
        module.load_vocabulary(document, known_facts=sorted(module.KNOWN_FACTS))


@pytest.mark.parametrize(
    ("pairs", "long_curve", "expected"),
    [
        ({"7": (2.0, 1.95), "13": (2.0, 1.95)}, None, "C-b"),
        ({"7": (2.0, 1.9), "13": (2.0, 1.9)}, _turned(7200, bottom_at=4800), "C-a"),
        ({"7": (2.0, 2.06), "13": (2.0, 2.06)}, None, "C-c"),
        ({"7": (2.0, 1.998), "13": (2.0, 1.998)}, None, "C-d"),
    ],
)
def test_each_declared_letter_is_reachable_from_the_tools_own_readings(
    tmp_path: Path, pairs: dict[str, tuple[float, float]], long_curve: Any, expected: str
) -> None:
    report = _read(
        tmp_path,
        pairs,
        long_curve=long_curve,
        long_best_step=4800 if long_curve is not None else None,
        vocabulary_path=_VOCABULARY,
    )
    assert report["outcome"]["letter"] == expected


def test_a_declared_set_that_did_not_complete_cannot_claim_a_positive_outcome(
    tmp_path: Path,
) -> None:
    report = _read(
        tmp_path,
        {"7": (2.0, 1.9), "13": (2.0, 1.9)},
        long_curve=_turned(7200, bottom_at=4800),
        long_best_step=4800,
        vocabulary_path=_VOCABULARY,
        expected_pairs=3,
    )
    assert report["facts"]["all_declared_pairs_present"] is False
    assert report["outcome"]["letter"] == "C-d"


def test_the_verdict_records_which_band_it_was_read_on_and_that_it_is_not_a_capability_claim(
    tmp_path: Path,
) -> None:
    report = _read(tmp_path, {"7": (2.0, 1.95), "13": (2.0, 1.95)})
    assert report["facts"]["o1_band"] == 0.005
    assert report["facts"]["inherited_o1_band"] == 0.02
    assert "K-001" in report["claim_limit"]


def test_the_inputs_are_digested_so_a_reader_can_prove_which_reports_made_the_number(
    tmp_path: Path,
) -> None:
    report = _read(tmp_path, {"7": (2.0, 1.95), "13": (2.0, 1.95)})
    assert set(report["inputs"]) == {"short", "long"}
    for side in ("short", "long"):
        for seed in ("7", "13"):
            entry = report["inputs"][side][seed]
            assert len(entry["sha256"]) == 64
            assert entry["path"].endswith(f"{side}-{seed}.json")
            assert Path(entry["path"]).read_bytes().startswith(b"{")


# ------------------------------------------------------------------ the command


def test_the_command_refuses_to_replace_a_reading_already_written(tmp_path: Path) -> None:
    module = _script()
    out = tmp_path / "results.json"
    out.write_text("{}", encoding="utf-8")
    status = module.main(
        [f"--short=7={out}", f"--long=7={out}", f"--bands={_BANDS_RECORD}", f"--out={out}"]
    )
    assert status == 2


def test_the_command_writes_the_reading_it_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script()
    short = _written(tmp_path, "s.json", _report(max_steps=2400))
    long_ = _written(tmp_path, "l.json", _report(max_steps=7200, bits=1.95))
    out = tmp_path / "out" / "results.json"
    status = module.main(
        [
            f"--short=7={short}",
            f"--short=13={short}",
            f"--long=7={long_}",
            f"--long=13={long_}",
            f"--bands={_BANDS_RECORD}",
            f"--outcomes={_VOCABULARY}",
            f"--out={out}",
        ]
    )
    assert status == 0
    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["outcome"]["letter"] == "C-b"
    assert "band" in capsys.readouterr().err


def test_a_malformed_arm_entry_is_reported_rather_than_traced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _script()
    out = tmp_path / "o.json"
    status = module.main(
        ["--short=7", f"--long=7={out}", f"--bands={_BANDS_RECORD}", f"--out={out}"]
    )
    assert status == 1
    assert "must be SEED=report.json" in capsys.readouterr().err
