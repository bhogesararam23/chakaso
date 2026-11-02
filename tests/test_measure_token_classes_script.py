"""The token-class measurement counts what each identifier decodes to, and only that."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from chakaso.training.subword import train_subword_vocabulary

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "measure_token_classes.py"
_TEXT = "the quick red fox ran over the dry sand and slept, and then it woke; the end.\n\n" * 6


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_token_classes_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _vocabulary():  # type: ignore[no-untyped-def]
    return train_subword_vocabulary([_TEXT], target_size=300, min_frequency=2)


def test_the_classes_partition_every_identifier() -> None:
    result = _script().token_classes(_vocabulary(), _TEXT)  # type: ignore[attr-defined]

    assert sum(result["counts"].values()) == result["tokens"]
    assert result["characters"] == len(_TEXT)
    assert result["characters_per_token"] == pytest.approx(len(_TEXT) / result["tokens"])
    assert sum(result["shares"].values()) == pytest.approx(1.0)


def test_a_text_of_single_spaces_between_letters_has_whitespace_tokens_for_each_gap() -> None:
    vocabulary = train_subword_vocabulary(["a b c d e f g h"] * 4, target_size=270, min_frequency=2)

    result = _script().token_classes(vocabulary, "a b c d e f g h")  # type: ignore[attr-defined]

    assert result["counts"]["whitespace"] == 7
    assert result["single_space_tokens"] == 7
    assert result["counts"]["word_or_subword"] == 8


def test_punctuation_is_counted_apart_from_words_and_whitespace() -> None:
    result = _script().token_classes(_vocabulary(), "ab, cd; ef.")  # type: ignore[attr-defined]

    assert result["counts"]["punctuation_or_symbol"] >= 3


def test_the_command_line_reads_a_vocabulary_and_a_text_file(tmp_path: Path) -> None:
    vocabulary_path = tmp_path / "v.json"
    _vocabulary().save(vocabulary_path)
    text_path = tmp_path / "t.txt"
    text_path.write_text(_TEXT, encoding="utf-8")
    out = tmp_path / "out.json"

    status = _script().main(  # type: ignore[attr-defined]
        [f"--vocabulary={vocabulary_path}", f"--text-file={text_path}", f"--out={out}"]
    )

    assert status == 0
    assert json.loads(out.read_text(encoding="utf-8"))["characters"] == len(_TEXT)
