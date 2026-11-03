"""Tests for the arm-pair analysis script.

Nothing here trains or scores a model: the fixtures are reports and probe results written in this file, and every
expectation is derived by hand from McNemar's exact test and Holm's step-down. The property under test is the one
E2626 could not reach — an instrument ranking two independently trained arms — and the script's job is to refuse the
comparison when the arms were not matched, before any statistic is read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyse_arm_pairs.py"
_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_PHENOMENA = ("auxiliary_agreement", "determiner_noun_agreement", "reflexive_pronoun_agreement")
#: The three arms E2629 uses, at the whole-pack bits per character their records report.
_LOSSES = {"mixture": 2.05545, "prose": 2.16369, "half_prose": 2.16822}
_ITEMS = tuple(f"item-{index}" for index in range(1, 17))


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("analyse_arm_pairs_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _arm_report(name: str, *, whole: float, vocabulary: str = "t" * 16) -> dict[str, Any]:
    return {
        "experiment": "E2612",
        "corpus": {
            "fingerprint": "c" * 16,
            "split_fingerprint": "s" * 16,
            "split_seed": 7,
            "training_tokens_packed": 20_000_000,
        },
        "tokenizer": {"fingerprint": vocabulary, "size": 4096},
        "transformer": {
            "parameter_count": 3_413_760,
            "architecture": {"d_model": 256, "num_layers": 4},
            "budget": {
                "seed": 7,
                "max_steps": 2400,
                "schedule": "cosine",
                "effective_batch_size": 32,
                "learning_rate": 0.0006,
                "max_training_tokens": 20_000_000,
                "shuffle_training_units": True,
                "document_masking": False,
                "grad_accum_steps": 1,
                "training_group_prefixes": None if name == "mixture" else ["prose-"],
            },
            "test": {
                "bits_per_character": whole,
                "positions_scored": 250_000,
                "characters_scored": 500_000,
            },
            "test_partitions": {
                prefix: {"metrics": {"bits_per_character": whole + index / 100}}
                for index, prefix in enumerate(_PARTITIONS)
            },
        },
    }


def _reports() -> dict[str, dict[str, Any]]:
    return {name: _arm_report(name, whole=whole) for name, whole in _LOSSES.items()}


def _pair_result(values: dict[str, list[float]]) -> dict[str, Any]:
    """A minimal-pair result in the shape ``evaluate_minimal_pairs.py`` writes."""
    return {
        "models": {
            name: {
                "result": {
                    "outcomes": [
                        [item, _PHENOMENA[0], value]
                        for item, value in zip(_ITEMS, scores, strict=True)
                    ]
                }
            }
            for name, scores in values.items()
        }
    }


def _continuation_result(values: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "models": {
            name: {
                "result": {
                    "outcomes": {
                        "pmi_easy": [
                            [item, value] for item, value in zip(_ITEMS, scores, strict=True)
                        ],
                        "pmi_hard": [
                            [item, value] for item, value in zip(_ITEMS, scores, strict=True)
                        ],
                    }
                }
            }
            for name, scores in values.items()
        }
    }


def _read(**overrides: dict[str, list[float]]) -> dict[str, Any]:
    """Run the analysis over the fixture arms with one instrument's per-item scores overridden."""
    script = _script()
    scores = {name: list(values) for name, values in _BEST_FIRST.items()}
    scores.update(overrides)
    return script.read(
        reports=_reports(),
        reference="mixture",
        pairs_result=_pair_result(scores),
        continuations_result=_continuation_result(scores),
        usable_phenomena=_PHENOMENA,
        materiality=0.02,
        decision_bearing=("minimal_pairs_usable",),
    )


#: The best arm wins eight of the sixteen items against either of the two worse ones, which agree item for item.
#:
#: McNemar on eight discordant pairs and none the other way gives p = 2 * (1/2) ** 8 = 0.0078125, and Holm's first
#: step over three pairs is 0.05 / 3 = 0.016667, so both material pairs are separated. The pair inside the band has
#: no discordant item at all, so n = 0 and p = 1, and it is left alone.
_BEST_FIRST = {
    "mixture": [1.0] * 12 + [0.0] * 4,
    "prose": [1.0] * 4 + [0.0] * 12,
    "half_prose": [1.0] * 4 + [0.0] * 12,
}


def test_matched_arms_rank_every_material_pair_and_leave_the_band_alone() -> None:
    result = _read()

    assert result["outcome"]["letter"] == "Z-a"
    assert result["outcome"]["material_pairs"] == 2
    assert result["outcome"]["inside_band_pairs"] == 1
    for row in (row for row in result["pairs"] if row["material"]):
        cell = row["minimal_pairs_usable"]
        assert cell["discordant"] == [8, 0]
        assert cell["separated"] is True
        assert cell["direction"] == "agrees"
        assert cell["p_value"] == pytest.approx(0.0078125)
    inside = next(row for row in result["pairs"] if not row["material"])
    assert inside["minimal_pairs_usable"]["separated"] is False
    # The library's gap is ``criterion[first] - criterion[second]`` with lower better, so a negative number means
    # the first arm is the better one — the prose arm is, by 0.00453, and the band calls that equal.
    assert inside["criterion_gap"] == pytest.approx(-0.00453)


def test_the_criterion_and_the_arms_travel_with_the_result() -> None:
    result = _read()

    assert result["models"] == ["mixture", "prose", "half_prose"]
    assert result["arms"]["mixture"]["whole_pack_bits_per_character"] == 2.05545
    assert result["arms"]["mixture"]["o1_equal_weight_mean"] == pytest.approx(2.07545)
    assert result["arms"]["prose"]["vocabulary"] == "t" * 16
    assert result["matchedness"]["training_group_prefixes"]["equal"] is False
    assert result["usable_phenomena"] == list(_PHENOMENA)


def test_the_worse_arm_being_on_top_is_reported_as_a_contradiction() -> None:
    # The probe now prefers the arm the loss likes least, on eight discordant items.
    result = _read(
        mixture=[1.0] * 4 + [0.0] * 12,
        prose=[1.0] * 12 + [0.0] * 4,
    )

    assert result["outcome"]["letter"] == "Z-c"
    contradicted = result["outcome"]["per_instrument"]["minimal_pairs_usable"]["contradicted_pairs"]
    assert ["mixture", "prose"] in contradicted


def test_separating_the_pair_the_band_calls_equal_loses_the_strong_verdict() -> None:
    # The prose arm and the half-prose arm now disagree on eight items, inside the band.
    result = _read(prose=[1.0] * 12 + [0.0] * 4)

    inside = next(row for row in result["pairs"] if not row["material"])
    assert inside["minimal_pairs_usable"]["separated"] is True
    assert result["outcome"]["per_instrument"]["minimal_pairs_usable"][
        "inside_band_pairs_separated"
    ] == [["prose", "half_prose"]]
    # The best-versus-prose pair is a tie on this pattern, so not every material pair is ranked: Z-b, not Z-a.
    assert result["outcome"]["letter"] == "Z-b"


def test_an_instrument_that_sees_nothing_is_not_reported_as_a_pass() -> None:
    result = _read(prose=[1.0] * 8 + [0.0] * 8, half_prose=[1.0] * 8 + [0.0] * 8)

    # The best arm and the prose arm now split their eight discordant items 4:4, which is p = 1.0 exactly, and the
    # two worse arms disagree on nothing. Holm has nothing to reject.
    assert result["outcome"]["letter"] == "Z-d"
    assert all(row["minimal_pairs_usable"]["separated"] is False for row in result["pairs"])


def test_arms_that_differ_in_an_undeclared_condition_are_refused_before_any_statistic() -> None:
    script = _script()
    reports = _reports()
    reports["prose"] = json.loads(json.dumps(reports["prose"]))
    reports["prose"]["tokenizer"]["fingerprint"] = "f" * 16

    with pytest.raises(script.AnalysisError, match="tokenizer_fingerprint"):
        script.matchedness(reports, reference="mixture")


def test_an_instrument_that_missed_an_arm_is_refused_by_name() -> None:
    script = _script()
    pairs = _pair_result(_BEST_FIRST)
    del pairs["models"]["half_prose"]

    with pytest.raises(script.AnalysisError, match=r"minimal_pairs_usable.*half_prose"):
        script.read(
            reports=_reports(),
            reference="mixture",
            pairs_result=pairs,
            continuations_result=_continuation_result(_BEST_FIRST),
            usable_phenomena=_PHENOMENA,
            materiality=0.02,
            decision_bearing=("minimal_pairs_usable",),
        )


def test_a_model_outside_the_criterion_table_is_reported_and_not_compared() -> None:
    script = _script()
    scores = {**_BEST_FIRST, "untrained-control": [1.0] * 16}

    result = script.read(
        reports=_reports(),
        reference="mixture",
        pairs_result=_pair_result(scores),
        continuations_result=_continuation_result(scores),
        usable_phenomena=_PHENOMENA,
        materiality=0.02,
        decision_bearing=("minimal_pairs_usable",),
    )

    assert result["models_scored_but_not_analysed"] == ["untrained-control"]
    assert result["models"] == ["mixture", "prose", "half_prose"]
    assert all(row["first"] != "untrained-control" for row in result["pairs"])


def test_the_arm_entries_are_parsed_from_name_equals_path(tmp_path: Path) -> None:
    script = _script()
    first = tmp_path / "first.json"
    first.write_text(json.dumps(_arm_report("mixture", whole=2.05)), encoding="utf-8")

    loaded = script.arm_reports([f"mixture={first}"])
    assert loaded["mixture"]["experiment"] == "E2612"

    with pytest.raises(script.AnalysisError, match="NAME=path"):
        script.arm_reports(["mixture"])
    with pytest.raises(script.AnalysisError, match="given twice"):
        script.arm_reports([f"mixture={first}", f"mixture={first}"])
    with pytest.raises(script.AnalysisError, match="not an experiment report"):
        missing = tmp_path / "other.json"
        missing.write_text("{}", encoding="utf-8")
        script.arm_reports([f"other={missing}"])


def test_the_summary_names_each_pair_and_the_outcome() -> None:
    script = _script()
    result = _read()

    text = script.summarise(result)

    assert "mixture vs prose: gap 0.10824 material" in text
    assert "prose vs half_prose: gap 0.00453 inside the band" in text
    assert "outcome: Z-a" in text


def test_a_compact_extract_of_a_probe_result_is_read_as_the_arrays_it_stands_for(
    tmp_path: Path,
) -> None:
    """K-032's fix on the reader's side: the file that was refused can now be paired.

    E2629 handed this script E2627's committed extract and was refused because the extractor had stubbed the
    per-item array. A file written in the compact form carries the same rows, and the analysis must produce the
    same verdict from it as from the whole result — byte for byte, not approximately.
    """
    from chakaso.training.outcome_vectors import compact_probe_result

    script = _script()
    whole = _pair_result(_BEST_FIRST)
    path = tmp_path / "minimal-pairs.json"
    path.write_text(json.dumps(compact_probe_result(whole), indent=2), encoding="utf-8")

    loaded = script._load(path)
    assert loaded == whole
    result = script.read(
        reports=_reports(),
        reference="mixture",
        pairs_result=loaded,
        continuations_result=_continuation_result(_BEST_FIRST),
        usable_phenomena=_PHENOMENA,
        materiality=0.02,
        decision_bearing=("minimal_pairs_usable",),
    )
    assert result["outcome"] == _read()["outcome"]


def test_an_extract_whose_array_was_stubbed_is_still_refused_by_name(tmp_path: Path) -> None:
    """The other half of the same fix: a pre-K-032 extract must not silently read as an empty one."""
    script = _script()
    stubbed = _pair_result(_BEST_FIRST)
    for entry in stubbed["models"].values():
        entry["result"]["outcomes"] = {
            "omitted_from_extract": True,
            "canonical_json_sha256": "0" * 64,
            "json_bytes": 58_600,
        }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(stubbed), encoding="utf-8")

    # The refusal names the fact about the file — the rows were left out to fit the size bound — rather than the
    # misleading "no pair matched the usable phenomena" a reader would otherwise get and go debugging for.
    with pytest.raises(script.AnalysisError, match="omitted from this extract"):
        script.read(
            reports=_reports(),
            reference="mixture",
            pairs_result=script._load(path),
            continuations_result=_continuation_result(_BEST_FIRST),
            usable_phenomena=_PHENOMENA,
            materiality=0.02,
            decision_bearing=("minimal_pairs_usable",),
        )
