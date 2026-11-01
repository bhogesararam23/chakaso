"""Tests for loading an experiment plan from a committed protocol file.

A protocol file exists so that a record can cite one artifact instead of a command line, which only helps if the
loader is stricter than a command line: an unknown field, a value of the wrong type or a quantity declared twice
with two values must all stop the run before it starts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

from chakaso.training.protocol import ProtocolError, load_plan
from chakaso.training.subword_experiment import ExperimentPlan

_PROTOCOL = """
[plan]
model_family = "decoder"
vocab_size = 4096
learning_rate = 6e-4
max_steps = 2400
whole_work_splits = true
character_baseline = false
training_group_prefixes = ["prose-"]
test_partitions = ["prose-", "science-"]
tokenizer_file = "vocabulary.json"
max_training_tokens = 20000000
"""


def _write(tmp_path: Path, text: str = _PROTOCOL) -> Path:
    path = tmp_path / "protocol.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _load(path: Path, **fixed: object) -> ExperimentPlan:
    return load_plan(path, root=Path(), work_dir=Path("work"), fixed=fixed or None)


def test_a_protocol_file_declares_the_plan_with_every_type_restored(tmp_path: Path) -> None:
    plan = _load(_write(tmp_path))

    assert plan.model_family == "decoder"
    assert plan.vocab_size == 4096
    assert plan.learning_rate == pytest.approx(6e-4)
    assert plan.whole_work_splits is True and plan.character_baseline is False
    assert plan.training_group_prefixes == ("prose-",)
    assert plan.test_partitions == ("prose-", "science-")
    assert plan.tokenizer_file == Path("vocabulary.json")
    assert plan.max_training_tokens == 20_000_000
    assert plan.work_dir == Path("work")


def test_an_integer_is_accepted_where_the_plan_wants_a_float(tmp_path: Path) -> None:
    plan = _load(_write(tmp_path, "[plan]\nlearning_rate = 1\n"))

    assert plan.learning_rate == 1.0 and isinstance(plan.learning_rate, float)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[plan]\nlearning_rat = 0.1\n", "not experiment plan fields"),
        ('[plan]\nwork_dir = "elsewhere"\n', "where a run lives"),
        ("[plan]\noverwrite_artifacts = true\n", "where a run lives"),
        ('[plan]\nmax_steps = "many"\n', "wrong type"),
        ("[plan]\nwhole_work_splits = 1\n", "wrong type"),
        ("[plan]\nmax_steps = true\n", "wrong type"),
        ('[plan]\ntest_partitions = "prose-"\n', "wrong type"),
        ("[other]\nmax_steps = 1\n", "no \\[plan\\] table"),
        ("[plan\nmax_steps = 1\n", "cannot read"),
    ],
)
def test_a_protocol_that_cannot_mean_what_it_says_is_refused(
    tmp_path: Path, text: str, match: str
) -> None:
    with pytest.raises(ProtocolError, match=match):
        _load(_write(tmp_path, text))


def test_a_missing_protocol_file_is_a_named_error(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="cannot read the protocol file"):
        _load(tmp_path / "absent.toml")


def test_a_quantity_declared_in_both_places_must_agree(tmp_path: Path) -> None:
    path = _write(tmp_path, "[plan]\nmax_steps = 2400\nsplit_seed = 7\n")

    assert _load(path, split_seed=7, max_steps=2400).split_seed == 7
    with pytest.raises(ProtocolError, match="declares each quantity once"):
        _load(path, split_seed=11)
    with pytest.raises(ProtocolError, match="declares each quantity once"):
        _load(path, max_steps=100)


def test_a_value_the_command_line_supplies_alone_is_applied_and_an_absent_one_is_not(
    tmp_path: Path,
) -> None:
    plan = _load(_write(tmp_path), split_seed=7, expected_corpus_fingerprint=None)

    assert plan.split_seed == 7
    assert plan.expected_corpus_fingerprint is None


def test_the_plans_own_validation_still_runs_on_a_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected_tokenizer_fingerprint"):
        _load(_write(tmp_path, '[plan]\nexpected_tokenizer_fingerprint = "abc"\n'))
