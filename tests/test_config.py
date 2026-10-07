"""Tests for configuration loading and validation.

Configuration failures are the ones a user meets first and understands least, so
the assertions here are mostly about error messages naming the offending field and
file rather than about the happy path.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from chakaso.config import (
    BUILT_IN_SOURCE,
    SPECS,
    Config,
    ConfigFileError,
    ConfigParseError,
    ConfigValidationError,
    FieldSpec,
    ModelConfig,
    load_config,
)
from chakaso.config.schema import SPECS_BY_KEY, model_config_field_names, validate_field

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = REPO_ROOT / "configs" / "default.toml"


def write_config(tmp_path: Path, text: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_built_in_defaults_are_used_when_no_file_is_given() -> None:
    loaded = load_config()

    assert loaded.config.model.adapter == "deterministic"
    assert loaded.config.model.max_new_tokens == 256
    assert loaded.config.model.temperature == 0.0
    assert loaded.sources == ()
    assert set(loaded.provenance.values()) == {BUILT_IN_SOURCE}


def test_shipped_default_config_matches_the_built_in_defaults() -> None:
    # The shipped file documents the defaults and the schema declares them. They
    # are two places saying the same thing, so a test keeps them from drifting.
    from_file = load_config(DEFAULT_CONFIG_FILE)

    assert from_file.config == load_config().config


def test_values_from_a_file_are_reported_with_their_source(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
        [model]
        adapter = "local_test"
        max_new_tokens = 64
        """,
    )

    loaded = load_config(path)

    assert loaded.config.model.adapter == "local_test"
    assert loaded.config.model.max_new_tokens == 64
    # Untouched fields keep coming from the built-in defaults, and the provenance
    # says so rather than attributing every value to the file.
    assert loaded.config.model.temperature == 0.0
    assert loaded.provenance["model.adapter"] == str(path)
    assert loaded.provenance["model.temperature"] == BUILT_IN_SOURCE
    assert loaded.sources == (path,)


def test_later_files_win_over_earlier_ones(tmp_path: Path) -> None:
    base = write_config(
        tmp_path,
        """
        [model]
        max_new_tokens = 100
        temperature = 0.5
        """,
        name="base.toml",
    )
    override = write_config(
        tmp_path,
        """
        [model]
        max_new_tokens = 200
        """,
        name="local.toml",
    )

    loaded = load_config(base, overrides=[override])

    assert loaded.config.model.max_new_tokens == 200
    assert loaded.config.model.temperature == 0.5
    assert loaded.provenance["model.max_new_tokens"] == str(override)
    assert loaded.provenance["model.temperature"] == str(base)
    assert loaded.sources == (base, override)


def test_a_toml_integer_is_accepted_for_a_float_field(tmp_path: Path) -> None:
    # `temperature = 0` is the natural thing to write and is unambiguous, so it is
    # accepted and normalised rather than rejected as a type error.
    path = write_config(
        tmp_path,
        """
        [model]
        temperature = 0
        """,
    )

    loaded = load_config(path)

    assert loaded.config.model.temperature == 0.0
    assert isinstance(loaded.config.model.temperature, float)


def test_unknown_key_is_rejected_and_names_the_key(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
        [model]
        adaptor = "typo"
        """,
    )

    with pytest.raises(ConfigValidationError) as info:
        load_config(path)

    message = str(info.value)
    assert "model.adaptor" in message
    assert str(path) in message
    # The message lists what is accepted, so the typo can be fixed from the error.
    assert "model.adapter" in message


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    # A section for a component that does not exist is a mistake, not forward
    # planning: silently ignoring it produces a run that does not use it.
    path = write_config(
        tmp_path,
        """
        [retrieval]
        top_k = 5
        """,
    )

    with pytest.raises(ConfigValidationError, match=r"retrieval\.top_k"):
        load_config(path)


def test_array_value_is_rejected(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
        [model]
        max_new_tokens = [1, 2, 3]
        """,
    )

    with pytest.raises(ConfigValidationError, match="array"):
        load_config(path)


def test_wrong_type_is_rejected_and_names_the_actual_type(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        """
        [model]
        max_new_tokens = "256"
        """,
    )

    with pytest.raises(ConfigValidationError) as info:
        load_config(path)

    assert "expected an integer" in str(info.value)
    assert "str" in str(info.value)


def test_boolean_is_not_accepted_where_an_integer_is_expected(tmp_path: Path) -> None:
    # bool is a subclass of int, so True would otherwise pass a range check as 1.
    path = write_config(
        tmp_path,
        """
        [model]
        max_new_tokens = true
        """,
    )

    with pytest.raises(ConfigValidationError, match="expected an integer"):
        load_config(path)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("max_new_tokens = 0", "minimum"),
        ("max_new_tokens = 1000001", "maximum"),
        ("temperature = -0.5", "minimum"),
        ("temperature = 11.0", "maximum"),
    ],
)
def test_out_of_range_values_are_rejected(tmp_path: Path, line: str, expected: str) -> None:
    path = write_config(tmp_path, f"[model]\n{line}\n")

    with pytest.raises(ConfigValidationError, match=expected):
        load_config(path)


@pytest.mark.parametrize("adapter", ["Deterministic", "with-dash", "9lives", "", "has space"])
def test_adapter_name_must_look_like_a_registry_key(tmp_path: Path, adapter: str) -> None:
    path = write_config(tmp_path, f'[model]\nadapter = "{adapter}"\n')

    with pytest.raises(ConfigValidationError):
        load_config(path)


def test_missing_file_is_reported_as_a_file_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigFileError, match="not found"):
        load_config(tmp_path / "absent.toml")


def test_directory_instead_of_file_is_reported_as_a_file_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigFileError, match="directory"):
        load_config(tmp_path)


def test_invalid_toml_is_reported_with_the_file_name(tmp_path: Path) -> None:
    path = write_config(tmp_path, "[model\nadapter = ")

    with pytest.raises(ConfigParseError) as info:
        load_config(path)

    assert str(path) in str(info.value)


def test_empty_file_yields_the_defaults(tmp_path: Path) -> None:
    # An empty file is a legitimate way to say "use the defaults", and it must not
    # be confused with a file that failed to load.
    path = write_config(tmp_path, "")

    loaded = load_config(path)

    assert loaded.config == load_config().config
    assert loaded.sources == (path,)


def test_config_is_frozen() -> None:
    config = load_config().config

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model.temperature = 1.0  # type: ignore[misc]


def test_provenance_mapping_cannot_be_mutated() -> None:
    loaded = load_config()

    with pytest.raises(TypeError):
        loaded.provenance["model.adapter"] = "something-else"  # type: ignore[index]


def test_schema_and_dataclasses_cover_the_same_fields() -> None:
    # A field added to a dataclass without a spec (or a spec left behind after a
    # field is removed) would otherwise be caught only at runtime, if at all.
    assert set(SPECS_BY_KEY) == model_config_field_names()


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.key)
def test_declared_defaults_satisfy_their_own_spec(spec: FieldSpec) -> None:
    # A default that violates its own range or pattern would make the schema
    # unsatisfiable and would only surface when the field was left unset.
    assert spec.description, spec.key
    assert spec.key.count(".") == 1, spec.key
    assert validate_field(spec, spec.default) == spec.default


def test_model_config_and_config_are_separate_types() -> None:
    # Guards against a refactor that flattens the sections, which would break the
    # dotted-key contract that error messages and provenance depend on.
    assert Config.__dataclass_fields__.keys() == {"model"}
    assert set(ModelConfig.__dataclass_fields__) == {"adapter", "max_new_tokens", "temperature"}


def test_relative_paths_are_reported_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Provenance should name the file the caller asked for, not an absolute path
    # that depends on the working directory.
    write_config(tmp_path, "[model]\nmax_new_tokens = 32\n")
    monkeypatch.chdir(tmp_path)

    loaded = load_config("config.toml")

    assert loaded.sources == (Path("config.toml"),)
    assert loaded.provenance["model.max_new_tokens"] == "config.toml"
