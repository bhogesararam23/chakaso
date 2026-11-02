"""The E2620 reading applies V-a..V-d at their boundaries and refuses arms that are not scored on the same text."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "read_tokenizer_arm.py"
_PARTITIONS = ("prose-", "science-", "technology-", "medicine-", "mathematics-")
_RULE = "space-prefixed-word|space-prefixed-symbol|whitespace-run"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("read_tokenizer_arm_under_test", _SCRIPT)
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
    documents: int = 34,
    selection_fingerprint: str | None = None,
    corpus: str = "c" * 16,
    characters: int = 535_000,
    cpt: float = 2.1,
) -> str:
    tokenizer: dict[str, Any] = {"fingerprint": "t" * 16 if not treatment else "n" * 16}
    tokenizer["characters_per_token_test"] = cpt
    if treatment:
        tokenizer["pre_tokenizer"] = _RULE
    corpus_block: dict[str, Any] = {
        "fingerprint": corpus,
        "split_fingerprint": "s" * 16,
        "test_documents_packed": documents,
    }
    if treatment:
        corpus_block["evaluation_selection"] = {
            "tokenizer_fingerprint": selection_fingerprint or "t" * 16,
            "test_documents": documents,
            "validation_documents": 34,
        }
    report = {
        "corpus": corpus_block,
        "tokenizer": tokenizer,
        "transformer": {
            "parameter_count": 3_413_760,
            "budget": {"max_steps": 2400, "batch_size": 32},
            "test": {
                "bits_per_character": whole,
                "characters_scored": characters,
                "positions_scored": 254_976,
            },
            "test_partitions": {
                prefix: {
                    "documents_packed": documents,
                    "metrics": {"bits_per_character": partition, "characters_scored": 100_000},
                }
                for prefix in _PARTITIONS
            },
        },
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path)


def _read(tmp_path: Path, delta_o1: float, delta_whole: float, **kwargs: Any) -> dict[str, Any]:
    out = tmp_path / "out.json"
    control = _report(tmp_path / "c.json", partition=1.90, whole=2.0554, treatment=False)
    treatment = _report(
        tmp_path / "t.json",
        partition=1.90 - delta_o1,
        whole=2.0554 - delta_whole,
        treatment=True,
        **kwargs,
    )
    status = _script().main(  # type: ignore[attr-defined]
        [f"--treatment={treatment}", f"--control={control}", f"--out={out}"]
    )
    assert status == 0
    return json.loads(out.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("delta_o1", "delta_whole", "outcome"),
    [
        (0.0201, 0.0094, "V-a"),
        (0.0201, 0.0093, "V-d"),  # one measure short of its bar
        (0.0200, 0.0300, "V-d"),  # exactly at the separable bar is not over it
        (-0.0201, -0.0094, "V-b"),
        (-0.0201, -0.0090, "V-d"),
        (0.0062, 0.00933, "V-c"),
        (-0.0062, -0.00933, "V-c"),
        (0.0063, 0.0000, "V-d"),
        (0.0, 0.0, "V-c"),
    ],
)
def test_the_outcome_follows_the_records_boundaries(
    tmp_path: Path, delta_o1: float, delta_whole: float, outcome: str
) -> None:
    assert _read(tmp_path, delta_o1, delta_whole)["outcome"] == outcome


def test_the_reading_reports_characters_per_identifier_and_both_deltas(tmp_path: Path) -> None:
    result = _read(tmp_path, 0.03, 0.02)

    assert result["delta_o1"] == pytest.approx(0.03, abs=1e-9)
    assert result["delta_whole_pack"] == pytest.approx(0.02, abs=1e-9)
    assert result["characters_per_identifier"] == {"control": 2.1, "treatment": 2.1}


def test_a_treatment_scored_on_other_documents_than_the_control_is_refused(tmp_path: Path) -> None:
    control = _report(tmp_path / "c.json", partition=1.9, whole=2.0, treatment=False)
    treatment = _report(tmp_path / "t.json", partition=1.9, whole=2.0, treatment=True, documents=40)

    status = _script().main([f"--treatment={treatment}", f"--control={control}"])  # type: ignore[attr-defined]

    assert status == 2


def test_a_treatment_whose_documents_were_chosen_by_another_vocabulary_is_refused(
    tmp_path: Path,
) -> None:
    control = _report(tmp_path / "c.json", partition=1.9, whole=2.0, treatment=False)
    treatment = _report(
        tmp_path / "t.json",
        partition=1.9,
        whole=2.0,
        treatment=True,
        selection_fingerprint="x" * 16,
    )

    assert _script().main([f"--treatment={treatment}", f"--control={control}"]) == 2  # type: ignore[attr-defined]


def test_a_treatment_with_no_selection_block_is_refused(tmp_path: Path) -> None:
    control = _report(tmp_path / "c.json", partition=1.9, whole=2.0, treatment=False)
    treatment_path = tmp_path / "t.json"
    _report(treatment_path, partition=1.9, whole=2.0, treatment=True)
    document = json.loads(treatment_path.read_text(encoding="utf-8"))
    del document["corpus"]["evaluation_selection"]
    treatment_path.write_text(json.dumps(document), encoding="utf-8")

    assert _script().main([f"--treatment={treatment_path}", f"--control={control}"]) == 2  # type: ignore[attr-defined]


def test_a_treatment_on_another_corpus_is_refused(tmp_path: Path) -> None:
    control = _report(tmp_path / "c.json", partition=1.9, whole=2.0, treatment=False)
    treatment = _report(
        tmp_path / "t.json", partition=1.9, whole=2.0, treatment=True, corpus="d" * 16
    )

    assert _script().main([f"--treatment={treatment}", f"--control={control}"]) == 2  # type: ignore[attr-defined]


def test_a_treatment_that_still_uses_the_default_rule_is_refused(tmp_path: Path) -> None:
    control = _report(tmp_path / "c.json", partition=1.9, whole=2.0, treatment=False)
    treatment_path = tmp_path / "t.json"
    _report(treatment_path, partition=1.9, whole=2.0, treatment=True)
    document = json.loads(treatment_path.read_text(encoding="utf-8"))
    del document["tokenizer"]["pre_tokenizer"]
    treatment_path.write_text(json.dumps(document), encoding="utf-8")

    assert _script().main([f"--treatment={treatment_path}", f"--control={control}"]) == 2  # type: ignore[attr-defined]
