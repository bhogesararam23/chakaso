"""Tests for adapter selection.

The registry is the single place where configuration becomes a model (ADR-0002),
so the interesting behaviour is the failure paths: an unregistered name, a name
that could not have come from a configuration file, and a duplicate registration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chakaso.config import load_config
from chakaso.models import (
    DETERMINISTIC_MODEL_ID,
    DeterministicModel,
    LanguageModel,
    ModelRegistry,
    create_model,
    default_registry,
)
from chakaso.models.errors import UnknownModelError

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "default.toml"


def test_default_registry_contains_the_deterministic_double() -> None:
    assert default_registry().names() == (DETERMINISTIC_MODEL_ID,)


def test_default_registry_returns_a_fresh_instance_each_time() -> None:
    # A shared registry would let one test's registration leak into another.
    assert default_registry() is not default_registry()


def test_create_model_builds_the_configured_adapter() -> None:
    config = load_config(DEFAULT_CONFIG_FILE).config

    model = create_model(config)

    assert isinstance(model, DeterministicModel)
    assert model.metadata.model_id == DETERMINISTIC_MODEL_ID


def test_create_model_without_a_config_file_uses_the_built_in_default() -> None:
    assert isinstance(create_model(load_config().config), LanguageModel)


def test_create_model_uses_a_supplied_registry(tmp_path: Path) -> None:
    def factory(_config: object) -> LanguageModel:
        return DeterministicModel(model_id="from_custom_registry")

    registry = ModelRegistry()
    registry.register("custom", factory)  # type: ignore[arg-type]
    path = tmp_path / "custom.toml"
    path.write_text("[model]\nadapter = 'custom'\n", encoding="utf-8")

    model = create_model(load_config(path).config, registry=registry)

    assert model.metadata.model_id == "from_custom_registry"


def test_unknown_adapter_names_what_is_registered(tmp_path: Path) -> None:
    path = tmp_path / "unknown.toml"
    path.write_text("[model]\nadapter = 'nope'\n", encoding="utf-8")

    with pytest.raises(UnknownModelError) as info:
        create_model(load_config(path).config)

    message = str(info.value)
    assert "nope" in message
    assert DETERMINISTIC_MODEL_ID in message


def test_unknown_model_error_is_a_lookup_error() -> None:
    # Configuration-driven lookups should be catchable as the built-in type.
    assert issubclass(UnknownModelError, LookupError)


def test_registering_the_same_name_twice_is_rejected() -> None:
    # A silent overwrite is how two tests end up disagreeing about what a name
    # means, and how a mistyped registration hides a real one.
    def factory(_config: object) -> LanguageModel:
        return DeterministicModel()

    registry = ModelRegistry()
    registry.register("dup", factory)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="already registered"):
        registry.register("dup", factory)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["", "Upper", "with-dash", "9lives", "has space"])
def test_invalid_adapter_name_is_rejected(name: str) -> None:
    # The same string is written in a configuration file, and the schema accepts
    # only names matching this pattern, so a name that cannot be configured cannot
    # be registered either.
    with pytest.raises(ValueError, match="not a valid adapter name"):
        ModelRegistry().register(name, lambda _config: DeterministicModel())


def test_registry_reports_membership_and_size() -> None:
    registry = default_registry()

    assert DETERMINISTIC_MODEL_ID in registry
    assert "absent" not in registry
    assert len(registry) == 1


def test_registry_accepts_initial_factories() -> None:
    registry = ModelRegistry({"seeded": lambda _config: DeterministicModel()})

    assert registry.names() == ("seeded",)


def test_names_are_sorted() -> None:
    registry = ModelRegistry()
    for name in ("zebra", "alpha", "middle"):
        registry.register(name, lambda _config: DeterministicModel())

    assert registry.names() == ("alpha", "middle", "zebra")
