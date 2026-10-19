"""Tests for the training configuration's validation and reproducibility surface.

Pure standard library, so these run in the dependency-free core CI matrix too.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from chakaso.training.config import TrainingConfig


def _config(**overrides: object) -> TrainingConfig:
    base: dict[str, object] = {
        "context_size": 4,
        "embed_dim": 8,
        "hidden_dim": 16,
        "learning_rate": 0.01,
        "seed": 0,
        "steps": 100,
        "eval_every": 10,
    }
    base.update(overrides)
    return TrainingConfig(**base)  # type: ignore[arg-type]


def test_a_valid_config_round_trips_through_as_dict() -> None:
    config = _config()
    mapping = config.as_dict()

    assert mapping["context_size"] == 4
    assert mapping["learning_rate"] == 0.01
    assert mapping["seed"] == 0


def test_a_frozen_config_cannot_be_mutated() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.seed = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("context_size", 0),
        ("embed_dim", -1),
        ("hidden_dim", True),
        ("steps", 0),
        ("eval_every", -2),
    ],
)
def test_a_non_positive_integer_dimension_is_refused(field: str, bad: object) -> None:
    with pytest.raises(ValueError, match=f"{field}"):
        _config(**{field: bad})


def test_a_non_positive_learning_rate_is_refused() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=0.0)
    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=-0.1)


def test_a_non_integer_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="seed"):
        _config(seed=1.5)


def test_a_boolean_seed_is_refused_rather_than_read_as_int() -> None:
    # bool is an int subclass; True must not silently mean seed == 1.
    with pytest.raises(ValueError, match="seed"):
        _config(seed=True)
