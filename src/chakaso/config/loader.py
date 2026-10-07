"""Loading and validating configuration.

The loader's job is to answer, for every value, both *what is it* and *where did it
come from*. The second question is what makes a run reproducible: "temperature was
0.0" is not useful without knowing whether that came from the built-in default or
from a file that has since been edited.

Loading is explicit. Nothing scans the filesystem for a configuration file, and no
environment variable is substituted for a file value. A caller passes the files it
wants, in the order it wants them applied.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from chakaso.config.errors import ConfigFileError, ConfigParseError, ConfigValidationError
from chakaso.config.schema import (
    SPECS,
    SPECS_BY_KEY,
    Config,
    FieldValue,
    build_config,
    validate_field,
)

__all__ = ["BUILT_IN_SOURCE", "LoadedConfig", "ResolvedValue", "load_config"]

# The provenance label for a value that came from the schema's default rather than
# from any file.
BUILT_IN_SOURCE = "built-in default"


@dataclass(frozen=True, slots=True)
class ResolvedValue:
    """One configuration value, with the source it was taken from."""

    key: str
    value: FieldValue
    source: str
    description: str


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    """A resolved configuration together with how it was resolved."""

    config: Config
    resolved: tuple[ResolvedValue, ...]
    sources: tuple[Path, ...]

    @property
    def provenance(self) -> Mapping[str, str]:
        """Map every configuration key to the source that supplied its value."""
        return MappingProxyType({item.key: item.source for item in self.resolved})


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Iterable[str | Path] = (),
) -> LoadedConfig:
    """Load configuration from files layered over the built-in defaults.

    Args:
        path: The primary configuration file. ``None`` uses the built-in defaults
            only, which is useful for tests and for checking what the defaults are.
        overrides: Additional files applied after ``path``, later files winning.
            This is how a machine-local file is applied: explicitly, by name, so
            that the resolved configuration always reports what was read.

    Returns:
        The typed configuration, the value-by-value provenance, and the files read.

    Raises:
        ConfigFileError: A file does not exist or could not be read.
        ConfigParseError: A file is not valid TOML.
        ConfigValidationError: A key is unknown, or a value is of the wrong type or
            outside its permitted range.
    """
    values: dict[str, FieldValue] = {spec.key: spec.default for spec in SPECS}
    provenance: dict[str, str] = {spec.key: BUILT_IN_SOURCE for spec in SPECS}
    sources: list[Path] = []

    for candidate in _resolve_paths(path, overrides):
        raw = _read_toml(candidate)
        flat = _flatten(raw, candidate)
        for key, value in flat.items():
            spec = SPECS_BY_KEY.get(key)
            if spec is None:
                known = ", ".join(sorted(SPECS_BY_KEY))
                message = f"{candidate}: unknown configuration key {key!r}. Known keys: {known}."
                raise ConfigValidationError(message)
            values[key] = validate_field(spec, value)
            provenance[key] = str(candidate)
        sources.append(candidate)

    resolved = tuple(
        ResolvedValue(
            key=spec.key,
            value=values[spec.key],
            source=provenance[spec.key],
            description=spec.description,
        )
        for spec in SPECS
    )
    return LoadedConfig(config=build_config(values), resolved=resolved, sources=tuple(sources))


def _resolve_paths(path: str | Path | None, overrides: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    if path is not None:
        paths.append(Path(path))
    paths.extend(Path(item) for item in overrides)
    return paths


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            # tomllib requires binary mode and raises TOMLDecodeError, which is a
            # ValueError and says nothing about which file it came from.
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        message = f"configuration file not found: {path}"
        raise ConfigFileError(message) from exc
    except IsADirectoryError as exc:
        message = f"configuration path is a directory, not a file: {path}"
        raise ConfigFileError(message) from exc
    except PermissionError as exc:
        message = f"configuration file is not readable: {path}"
        raise ConfigFileError(message) from exc
    except tomllib.TOMLDecodeError as exc:
        message = f"{path}: not valid TOML: {exc}"
        raise ConfigParseError(message) from exc


def _flatten(raw: Mapping[str, object], path: Path) -> dict[str, FieldValue]:
    """Turn nested TOML tables into dotted keys.

    A table nested more deeply than the schema allows becomes an unknown key rather
    than being silently dropped, so a misplaced section in a configuration file is
    an error instead of a surprise.

    TOML value types outside the schema's set — dates carry their own types in
    TOML, for example — are rejected here rather than reaching validation as an
    object nothing downstream can interpret.
    """
    flat: dict[str, FieldValue] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for nested_key, nested_value in _flatten(value, path).items():
                flat[f"{key}.{nested_key}"] = nested_value
        elif isinstance(value, list):
            message = (
                f"{path}: {key!r} is an array. Chakaso configuration has no array "
                "fields; a list-valued setting needs a schema field and a reason."
            )
            raise ConfigValidationError(message)
        elif isinstance(value, str | int | float | bool):
            flat[key] = value
        else:
            message = (
                f"{path}: {key!r} has TOML type {type(value).__name__}, which Chakaso "
                "configuration does not support. Expected a string, integer, float or "
                "boolean."
            )
            raise ConfigValidationError(message)
    return flat
