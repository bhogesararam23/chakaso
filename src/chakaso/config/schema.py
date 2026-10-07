"""The configuration schema.

Configuration is small on purpose. Every value here is required by something that
already exists or by a decision already recorded, and nothing is here because the
roadmap mentions a component that will need it. Configuration that anticipates
requirements is configuration that has to be maintained against guesses.

The schema is declared as an explicit list of fields rather than derived from the
dataclasses. Explicit specs make two things possible that reflection alone does not:
a field can carry a range or a pattern, and an unknown key in a file can be
rejected rather than ignored. A typo in a configuration key is a defect, and
silently ignoring it produces a run that does not use the settings the author
believes it does.

`tests/test_config.py` asserts that the specs and the dataclasses cover exactly the
same fields, so the two cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import cast

from chakaso.config.errors import ConfigValidationError

__all__ = [
    "ADAPTER_NAME_PATTERN",
    "SPECS",
    "Config",
    "FieldSpec",
    "FieldValue",
    "ModelConfig",
    "build_config",
]

# The value types a configuration field may hold. TOML produces exactly these, so
# no coercion from strings is ever needed.
FieldValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Declares one configuration field: its key, type, default and limits.

    Attributes:
        key: Dotted path, for example ``model.adapter``.
        kind: The declared Python type. ``int`` and ``float`` both accept a TOML
            integer for a float field, because ``temperature = 0`` is a natural
            thing to write and is unambiguous.
        default: The built-in value, used when no file sets the field.
        description: One line, shown by ``chakaso config show``.
        minimum: Inclusive lower bound for numeric fields.
        maximum: Inclusive upper bound for numeric fields.
        pattern: A regular expression that a string field must match entirely.
    """

    key: str
    kind: type
    default: FieldValue
    description: str
    minimum: int | float | None = None
    maximum: int | float | None = None
    pattern: str | None = None


# Names are used to look adapters up in the model registry, so they are restricted
# to a form that is stable in configuration files, log lines and file names. The
# model registry validates against this same pattern, so a name that cannot be
# configured cannot be registered either.
ADAPTER_NAME_PATTERN = r"[a-z][a-z0-9_]*"

SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        key="model.adapter",
        kind=str,
        # The only adapter that exists is the development double in
        # chakaso.models.deterministic. There is no trained Chakaso model and no
        # local inference adapter yet; when one lands, this default and
        # configs/default.toml change together, and a test enforces that.
        default="deterministic",
        description="Registered language-model adapter to use (ADR-0002)",
        pattern=ADAPTER_NAME_PATTERN,
    ),
    FieldSpec(
        key="model.max_new_tokens",
        kind=int,
        default=256,
        description="Default ceiling on generated tokens for one response",
        minimum=1,
        maximum=1_000_000,
    ),
    FieldSpec(
        key="model.temperature",
        kind=float,
        default=0.0,
        description="Default sampling temperature; 0.0 means greedy",
        minimum=0.0,
        maximum=10.0,
    ),
)

SPECS_BY_KEY: dict[str, FieldSpec] = {spec.key: spec for spec in SPECS}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings that select and constrain the language-model boundary."""

    adapter: str
    max_new_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class Config:
    """The resolved configuration.

    Frozen, so that a component cannot quietly change the configuration it was
    handed. A component that needs different settings takes them as arguments.
    """

    model: ModelConfig


def validate_field(spec: FieldSpec, value: object) -> FieldValue:
    """Return ``value`` if it satisfies ``spec``, or raise :class:`ConfigValidationError`.

    The error message names the field, the offending value and the expectation, so
    that a bad configuration file can be fixed without reading this module.
    """
    if spec.kind is bool:
        if not isinstance(value, bool):
            raise _type_error(spec, value, "a boolean")
        return value

    if spec.kind is int:
        # bool is a subclass of int in Python, and True would otherwise pass a
        # range check for max_new_tokens as the value 1.
        if isinstance(value, bool) or not isinstance(value, int):
            raise _type_error(spec, value, "an integer")
        _check_range(spec, value)
        return value

    if spec.kind is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _type_error(spec, value, "a number")
        number = float(value)
        _check_range(spec, number)
        return number

    if spec.kind is str:
        if not isinstance(value, str):
            raise _type_error(spec, value, "a string")
        if spec.pattern is not None and re.fullmatch(spec.pattern, value) is None:
            message = (
                f"{spec.key}: {value!r} does not match the required form "
                f"{spec.pattern!r}. {spec.description}."
            )
            raise ConfigValidationError(message)
        return value

    message = f"{spec.key}: unsupported field type {spec.kind!r}"
    raise ConfigValidationError(message)


def _type_error(spec: FieldSpec, value: object, expected: str) -> ConfigValidationError:
    return ConfigValidationError(
        f"{spec.key}: expected {expected}, got {type(value).__name__} ({value!r}). "
        f"{spec.description}."
    )


def _check_range(spec: FieldSpec, value: int | float) -> None:
    if spec.minimum is not None and value < spec.minimum:
        message = f"{spec.key}: {value!r} is below the minimum {spec.minimum!r}."
        raise ConfigValidationError(message)
    if spec.maximum is not None and value > spec.maximum:
        message = f"{spec.key}: {value!r} is above the maximum {spec.maximum!r}."
        raise ConfigValidationError(message)


def build_config(values: dict[str, FieldValue]) -> Config:
    """Build the typed configuration from flat, already-validated values.

    The construction is written out rather than driven by reflection: there are
    three fields, and an explicit mapping is easier to read than the machinery that
    would replace it. The consistency check between this function and :data:`SPECS`
    is a test, not a runtime cost.
    """
    missing = sorted(set(SPECS_BY_KEY) - set(values))
    if missing:
        message = f"missing configuration fields: {', '.join(missing)}"
        raise ConfigValidationError(message)

    return Config(
        model=ModelConfig(
            adapter=cast(str, values["model.adapter"]),
            max_new_tokens=cast(int, values["model.max_new_tokens"]),
            temperature=cast(float, values["model.temperature"]),
        )
    )


def model_config_field_names() -> set[str]:
    """Field names of :class:`ModelConfig`, prefixed with their section.

    Used by the schema-consistency test so that a field added to a dataclass
    without a spec, or removed from one without removing the spec, fails CI.
    """
    return {f"model.{field.name}" for field in fields(ModelConfig)}
