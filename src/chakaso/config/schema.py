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
    "RETRIEVAL_STRATEGY_PATTERN",
    "SPECS",
    "Config",
    "FieldSpec",
    "FieldValue",
    "ModelConfig",
    "RetrievalConfig",
    "StorageConfig",
    "build_config",
    "config_field_names",
    "model_config_field_names",
    "retrieval_config_field_names",
    "storage_config_field_names",
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

# The two answer-store backends, and the SQLite journal modes the durable store will
# accept. Both are restricted here so a typo ('sqllite', 'wa1') is a configuration
# error at load time rather than a surprise when a store is first opened. The journal
# set mirrors SQLiteAnswerStore._SUPPORTED_JOURNAL_MODES.
STORAGE_BACKEND_PATTERN = r"(memory|sqlite)"
STORAGE_JOURNAL_MODE_PATTERN = r"(delete|truncate|persist|memory|wal|off)"

# The retrieval strategies a query planner and the retrieval-aware runtime can select.
# All three are backed by a retriever today (lexical BM25, the exact dense index over the
# non-semantic fixture embedding, and reciprocal-rank fusion — ADR-0023/ADR-0024), so a
# configured strategy always has a real retriever behind it and never silently falls back
# to another. The set mirrors `chakaso.planning.QueryMode`'s retrieval modes.
RETRIEVAL_STRATEGY_PATTERN = r"(lexical|dense|hybrid)"

SPECS: tuple[FieldSpec, ...] = (
    FieldSpec(
        key="model.adapter",
        kind=str,
        # The default is the development double in chakaso.models.deterministic. Two trained
        # adapters are registered — trained_char behind [train] and subword_transformer behind
        # [nn] — and neither is ever the default: both are measured to generate looping text, and
        # selecting one without its extra or its checkpoint is a typed error, not a fallback.
        # Changing this default is a product decision, not a configuration convenience.
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
    FieldSpec(
        key="model.max_context_turns",
        kind=int,
        # A stopgap, and chosen as one. The model adapter clips its own window, but the
        # conversation layer holds no tokenizer, so a turn ceiling is the only bound this
        # configuration can express. The value is not a measured optimum and is not presented
        # as one.
        default=40,
        description="Maximum recent turns projected into one model request",
        minimum=1,
        maximum=10_000,
    ),
    FieldSpec(
        key="model.checkpoint_path",
        kind=str,
        # Path to a trained model checkpoint, read by the two trained adapters: the
        # character checkpoint (trained_char, [train]) and the subword Transformer
        # artifact (subword_transformer, [nn]) — the format decides the adapter when
        # load_model() is used. Empty by default so nothing is loaded and the base
        # configuration never requires an optional extra; an adapter selected without a
        # usable path fails explicitly rather than silently falling back.
        default="",
        description="Checkpoint path for a trained model adapter; unused by the double",
    ),
    FieldSpec(
        key="storage.backend",
        kind=str,
        # The answer store is selected here, in one place, exactly as the model adapter
        # is (ADR-0002's discipline applied to persistence). "memory" is the default:
        # nothing is written to disk unless a caller chooses the durable SQLite backend
        # (ADR-0018 kept in-memory the default; ADR-0021 added the durable option).
        default="memory",
        description="Answer-store backend to use: 'memory' or 'sqlite' (ADR-0021)",
        pattern=STORAGE_BACKEND_PATTERN,
    ),
    FieldSpec(
        key="storage.path",
        kind=str,
        # Empty by default so the built-in configuration writes nothing anywhere. A
        # durable path is set explicitly, and only then does backend='sqlite' make
        # sense; requiring a non-empty path when sqlite is chosen is a composition-time
        # check, not a schema one, because it depends on the backend value.
        default="",
        description="File path for the durable SQLite answer store; unused for 'memory'",
    ),
    FieldSpec(
        key="storage.journal_mode",
        kind=str,
        # SQLite's own default. The set matches SQLiteAnswerStore._SUPPORTED_JOURNAL_MODES;
        # 'wal' is available for readers concurrent with a writer but is not the default,
        # because it adds side-car files and no tested workload needs it yet.
        default="delete",
        description="SQLite journal mode for the durable store (ADR-0021)",
        pattern=STORAGE_JOURNAL_MODE_PATTERN,
    ),
    FieldSpec(
        key="retrieval.enabled",
        kind=bool,
        # The retrieval-aware runtime is what makes a turn grounded in evidence (ADR-0025).
        # It is on by default so `chakaso chat` runs through the real composition rather
        # than the bare conversation; with no documents supplied the planner still runs and
        # retrieval honestly returns an empty pack. Turning it off restores a retrieval-free
        # conversation, which is a distinct, deliberate mode rather than a degraded one.
        default=True,
        description="Whether the conversation runtime grounds turns in retrieved evidence (ADR-0025)",
    ),
    FieldSpec(
        key="retrieval.strategy",
        kind=str,
        # Which retriever the planner selects when it decides to retrieve. Every value is
        # backed by a retriever, so an explicit choice is honoured rather than substituted.
        # "dense"/"hybrid" run over the non-semantic fixture embedding (ADR-0023/ADR-0024).
        default="lexical",
        description="Retrieval strategy the planner selects: 'lexical', 'dense' or 'hybrid'",
        pattern=RETRIEVAL_STRATEGY_PATTERN,
    ),
    FieldSpec(
        key="retrieval.top_k",
        kind=int,
        # How many chunks a planned retrieval asks for. It matches the retrieval service's
        # own default so an unset configuration behaves exactly as the library does today.
        default=5,
        description="How many chunks a retrieval turn returns",
        minimum=1,
        maximum=1_000,
    ),
    FieldSpec(
        key="retrieval.candidate_k",
        kind=int,
        # The pool each hybrid component draws from before fusion (ADR-0024). It is used
        # only when the strategy is 'hybrid'; over-retrieving here trades latency for a
        # better merged ranking and has no effect on the lexical or dense paths.
        default=10,
        description="Per-component candidate pool for hybrid fusion (used when strategy is 'hybrid')",
        minimum=1,
        maximum=10_000,
    ),
    FieldSpec(
        key="retrieval.rrf_k",
        kind=int,
        # The reciprocal-rank-fusion smoothing constant (Cormack et al.; published default
        # 60). Used only for the 'hybrid' strategy. A larger value flattens rank influence.
        default=60,
        description="RRF smoothing constant for hybrid fusion (used when strategy is 'hybrid')",
        minimum=1,
        maximum=10_000,
    ),
    FieldSpec(
        key="retrieval.max_ingest_bytes",
        kind=int,
        # A bound on how large a document the runtime will read into the corpus, checked
        # against file size before bytes are loaded. It mirrors the ingestion layer's own
        # default so the built-in configuration matches today's behaviour exactly.
        default=10_485_760,
        description="Maximum size in bytes of a document the runtime will ingest",
        minimum=1,
        maximum=1_073_741_824,
    ),
)

SPECS_BY_KEY: dict[str, FieldSpec] = {spec.key: spec for spec in SPECS}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings that select and constrain the language-model boundary."""

    adapter: str
    max_new_tokens: int
    temperature: float
    max_context_turns: int
    checkpoint_path: str


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Settings that select and configure the durable answer-store backend (ADR-0021)."""

    backend: str
    path: str
    journal_mode: str


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Settings that construct the retrieval-aware runtime (ADR-0025).

    These are the values the application composition reads to decide whether a turn is
    grounded, which retriever the planner selects, and what limits bound the corpus. They
    cover only what genuinely exists: enabling retrieval, choosing among the strategies that
    have a retriever behind them, and the fusion/limit knobs those retrievers actually take.
    """

    enabled: bool
    strategy: str
    top_k: int
    candidate_k: int
    rrf_k: int
    max_ingest_bytes: int


@dataclass(frozen=True, slots=True)
class Config:
    """The resolved configuration.

    Frozen, so that a component cannot quietly change the configuration it was
    handed. A component that needs different settings takes them as arguments.
    """

    model: ModelConfig
    storage: StorageConfig
    retrieval: RetrievalConfig


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
            max_context_turns=cast(int, values["model.max_context_turns"]),
            checkpoint_path=cast(str, values["model.checkpoint_path"]),
        ),
        storage=StorageConfig(
            backend=cast(str, values["storage.backend"]),
            path=cast(str, values["storage.path"]),
            journal_mode=cast(str, values["storage.journal_mode"]),
        ),
        retrieval=RetrievalConfig(
            enabled=cast(bool, values["retrieval.enabled"]),
            strategy=cast(str, values["retrieval.strategy"]),
            top_k=cast(int, values["retrieval.top_k"]),
            candidate_k=cast(int, values["retrieval.candidate_k"]),
            rrf_k=cast(int, values["retrieval.rrf_k"]),
            max_ingest_bytes=cast(int, values["retrieval.max_ingest_bytes"]),
        ),
    )


def model_config_field_names() -> set[str]:
    """Field names of :class:`ModelConfig`, prefixed with their section.

    Used by the schema-consistency test so that a field added to a dataclass
    without a spec, or removed from one without removing the spec, fails CI.
    """
    return {f"model.{field.name}" for field in fields(ModelConfig)}


def storage_config_field_names() -> set[str]:
    """Field names of :class:`StorageConfig`, prefixed with their section."""
    return {f"storage.{field.name}" for field in fields(StorageConfig)}


def retrieval_config_field_names() -> set[str]:
    """Field names of :class:`RetrievalConfig`, prefixed with their section."""
    return {f"retrieval.{field.name}" for field in fields(RetrievalConfig)}


def config_field_names() -> set[str]:
    """Every dotted configuration key derived from the section dataclasses.

    The parity test compares this against the flat spec keys, so adding a section
    dataclass without its specs (or the reverse) fails CI rather than surfacing when
    a value is unexpectedly missing. Deriving the expected set from the dataclasses —
    rather than restating it in the test — is what keeps adding a field a one-place edit.
    """
    return (
        model_config_field_names() | storage_config_field_names() | retrieval_config_field_names()
    )
