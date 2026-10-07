"""The language-model boundary and adapter selection.

Application code imports types from here and never a concrete implementation
(ADR-0002). :func:`create_model` is the single place where configuration becomes a
model.

There is no trained Chakaso model and no local inference adapter yet. The adapter
:data:`~chakaso.models.deterministic.DETERMINISTIC_MODEL_ID` is a development
double that produces fixed text; it is not a language model, and its metadata says
so.
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
from chakaso.models.registry import ModelFactory, ModelRegistry, create_model, default_registry

__all__ = [
    "DETERMINISTIC_MODEL_ID",
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
