"""Tests for the paired held-out loss test's own machinery.

The statistics are tested in :mod:`tests.test_training_paired_loss`; this file tests the parts that decide whether
the numbers going *in* describe the same pack: the rebuild of the held-out side with document boundaries, the
comparison against a committed report, and the pairing of works between two arms.

Gated on ``[nn]`` because the script imports the checkpoint loader — a loss comparison that cannot load a model has
nothing to say.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

torch = pytest.importorskip("torch")

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

from chakaso.training.data import TextUnit  # noqa: E402
from chakaso.training.subword import train_subword_vocabulary  # noqa: E402

_VOCABULARY = train_subword_vocabulary(
    ["the quick red fox ran over the dry sand and the quiet owl slept"],
    target_size=280,
    min_frequency=2,
)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "paired_loss_test_under_test", _SCRIPTS / "paired_loss_test.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _units() -> tuple[TextUnit, ...]:
    return (
        TextUnit(identifier="a", text="the quick red fox ran over the dry sand", group="book-one"),
        TextUnit(identifier="b", text="the quiet owl slept peacefully", group="book-one"),
        TextUnit(identifier="c", text="a loud dog chased the red fox", group="book-two"),
    )


def test_the_rebuilt_pack_carries_the_document_boundaries_the_test_pairs_on() -> None:
    module = _script()
    batch, packed = module.build_pack(
        _units(), _VOCABULARY, context_length=8, max_tokens=None, selection_vocabulary=None
    )
    assert [unit.identifier for unit in packed] == ["a", "b", "c"]
    assert batch.document_count == 3
    assert all(example.segment_ids is not None for example in batch.examples)
    assert {segment for example in batch.examples for segment in example.segment_ids or ()} == {
        0,
        1,
        2,
    }


def test_a_selection_vocabulary_picks_the_documents_and_the_cap_moves_to_the_selection() -> None:
    module = _script()
    # Counting with a second vocabulary is how two arms with different tokenizers are scored on identical text
    # (ADR-0048). The cap applies to the selection, not to the packing, so no document is half taken.
    counts = [len(_VOCABULARY.encode(unit.text)) for unit in _units()]
    cap = counts[0] + counts[1]  # the walk stops after the whole unit that reaches the cap
    batch, packed = module.build_pack(
        _units(),
        _VOCABULARY,
        context_length=8,
        max_tokens=cap,
        selection_vocabulary=_VOCABULARY,
    )
    assert [unit.identifier for unit in packed] == ["a", "b"]
    assert batch.document_count == 2


def _scored(works: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {key: dict(value) for key, value in works.items()}
    nats = sum(row["nats"] for row in rows.values())
    characters = sum(row["characters"] for row in rows.values())
    return {
        "pooled_bits_per_character": nats / characters,
        "positions_scored": 100,
        "characters_scored": characters,
        "works": rows,
    }


def test_a_report_that_agrees_with_the_rebuilt_pack_raises_nothing(tmp_path: Path) -> None:
    module = _script()
    scored = {"a": _scored({"book-one": {"nats": 20.0, "characters": 10, "documents": 2}})}
    scored["a"]["pooled_bits_per_character"] = 2.0
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "transformer": {
                    "test": {
                        "positions_scored": 100,
                        "characters_scored": 10,
                        "bits_per_character": 2.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert module.check_against_reports(scored, {"a": str(report)}) == []


def test_a_denominator_that_moved_is_named_with_both_numbers(tmp_path: Path) -> None:
    module = _script()
    scored = {"a": _scored({"book-one": {"nats": 20.0, "characters": 12, "documents": 2}})}
    scored["a"]["pooled_bits_per_character"] = 2.0
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "transformer": {
                    "test": {
                        "positions_scored": 96,
                        "characters_scored": 12,
                        "bits_per_character": 1.9,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    problems = module.check_against_reports(scored, {"a": str(report)})
    assert len(problems) == 2
    assert "positions scored — the report says 96, this pack gives 100" in problems[0]
    assert "pooled bits per character" in problems[1]


def test_a_report_for_an_arm_that_was_not_scored_is_refused(tmp_path: Path) -> None:
    module = _script()
    problems = module.check_against_reports({}, {"ghost": str(tmp_path / "none.json")})
    assert problems == ["ghost: a report was given with no arm of that name"]


def test_two_are_scored_on_the_same_works_before_they_are_paired() -> None:
    module = _script()
    first = _scored(
        {
            "book-one": {"nats": 20.0, "characters": 10, "documents": 2},
            "book-two": {"nats": 12.0, "characters": 6, "documents": 1},
        }
    )
    second = _scored(
        {
            "book-one": {"nats": 22.0, "characters": 10, "documents": 2},
            "book-three": {"nats": 12.0, "characters": 6, "documents": 1},
        }
    )
    with pytest.raises(module.AnalysisError, match="different documents"):
        module.compare({"a": first, "b": second}, ["a/b"], seed=1, resamples=100)


def test_a_pair_naming_an_arm_that_was_not_scored_is_refused_by_name() -> None:
    module = _script()
    scored = {"a": _scored({"book-one": {"nats": 20.0, "characters": 10, "documents": 1}})}
    with pytest.raises(module.AnalysisError, match=r"names arms that were not scored"):
        module.compare(scored, ["a/missing"], seed=1, resamples=100)


def test_the_paired_reading_is_weighted_by_characters_and_reports_its_works() -> None:
    module = _script()
    # book-one is 20 nats over 10 characters against 22 over 10, and book-two 100 over 100 against 110 over 100,
    # so the first arm is better on both and every difference is negative: in bits per character the
    # character-weighted mean is log2(e) x ((20-22) + (100-110)) / 110 = 1.4427 x -0.10909 = -0.15738.
    first = _scored(
        {
            "book-one": {"nats": 20.0, "characters": 10, "documents": 2},
            "book-two": {"nats": 100.0, "characters": 100, "documents": 1},
        }
    )
    second = _scored(
        {
            "book-one": {"nats": 22.0, "characters": 10, "documents": 2},
            "book-two": {"nats": 110.0, "characters": 100, "documents": 1},
        }
    )
    reading = module.compare({"a": first, "b": second}, ["a/b"], seed=1, resamples=200)[0]
    test = reading["test"]
    assert test["mean_difference"] == pytest.approx(-0.1090909090909 * math.log2(math.e), abs=1e-9)
    assert test["clusters"] == 2
    assert test["documents"] == 2
    assert test["positive_documents"] == 0
    assert test["negative_documents"] == 2
    assert reading["published_difference"] == pytest.approx(
        first["pooled_bits_per_character"] - second["pooled_bits_per_character"]
    )


def test_name_equals_path_is_parsed_and_a_repeat_is_refused() -> None:
    module = _script()
    assert module._entries(["a=/x/y.pt", "b=/z/w.pt"], option="arm") == {
        "a": "/x/y.pt",
        "b": "/z/w.pt",
    }
    with pytest.raises(module.AnalysisError, match="wants NAME=path"):
        module._entries(["noequals"], option="arm")
    with pytest.raises(module.AnalysisError, match="names 'a' twice"):
        module._entries(["a=/x", "a=/y"], option="arm")
