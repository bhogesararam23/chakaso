"""The language-model boundary and adapter selection.

Application code imports types from here and never a concrete implementation
(ADR-0002). :func:`create_model` is the single place where configuration becomes a
model.

There is no trained Chakaso model *in the runtime*: the default adapter
:data:`~chakaso.models.deterministic.DETERMINISTIC_MODEL_ID` is a development double that produces
fixed text, is not a language model, and its metadata says so. A second adapter,
:data:`~chakaso.models.registry.TRAINED_CHAR_MODEL_ID`, can be selected to load a *trained* character
checkpoint (P25, ADR-0030), but it lives behind the optional ``[train]`` dependency, is not the default,
and its model is a tiny toy — not a useful language model.
"""

from __future__ import annotations

from chakaso.models.base import (
    Capability,
    FinishReason,
    GenerationParams,
    GenerationResult,
    LanguageModel,
    Message,
    ModelMetadata,
    Role,
    StructuredGenerator,
    StructuredResult,
    Tokenizer,
    generate_structured,
    require_capability,
    require_messages,
    tokenize,
)
from chakaso.models.deterministic import DETERMINISTIC_MODEL_ID, DeterministicModel
from chakaso.models.errors import ModelError, UnknownModelError, UnsupportedCapabilityError
from chakaso.models.registry import (
    TRAINED_CHAR_MODEL_ID,
    ModelFactory,
    ModelRegistry,
    create_model,
    default_registry,
)

__all__ = [
    "DETERMINISTIC_MODEL_ID",
    "TRAINED_CHAR_MODEL_ID",
    "Capability",
    "DeterministicModel",
    "FinishReason",
    "GenerationParams",
    "GenerationResult",
    "LanguageModel",
    "Message",
    "ModelError",
    "ModelFactory",
    "ModelMetadata",
    "ModelRegistry",
    "Role",
    "StructuredGenerator",
    "StructuredResult",
    "Tokenizer",
    "UnknownModelError",
    "UnsupportedCapabilityError",
    "create_model",
    "default_registry",
    "generate_structured",
    "require_capability",
    "require_messages",
    "tokenize",
]
