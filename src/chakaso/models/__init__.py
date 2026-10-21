"""The language-model boundary and adapter selection.

Application code imports types from here and never a concrete implementation
(ADR-0002). :func:`create_model` is the single place where configuration becomes a
model.

The primary interface to a trained checkpoint is :func:`load_model`: a Python caller with a checkpoint on
disk needs no terminal, no configuration file and no conversation runtime
([ADR-0033](../../../docs/decisions/ADR-0033-model-centred-product-contract.md)).

There is no trained Chakaso model *in the runtime's default path*: the default adapter
:data:`~chakaso.models.deterministic.DETERMINISTIC_MODEL_ID` is a development double that produces fixed
text, is not a language model, and its metadata says so. Two adapters can load a *trained* checkpoint when
selected: :data:`~chakaso.models.registry.TRAINED_CHAR_MODEL_ID` (a character toy, P25, behind the
``[train]`` extra) and :data:`~chakaso.models.registry.SUBWORD_TRANSFORMER_MODEL_ID` (a subword causal
Transformer, P31, behind the ``[nn]`` extra). Neither is the default, and neither is a useful language
model on a real workload — what they establish is that a checkpoint this project trained is loadable and
generatable.
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
from chakaso.models.load import load_model
from chakaso.models.registry import (
    SUBWORD_TRANSFORMER_MODEL_ID,
    TRAINED_CHAR_MODEL_ID,
    ModelFactory,
    ModelRegistry,
    create_adapter,
    create_model,
    default_registry,
)

__all__ = [
    "DETERMINISTIC_MODEL_ID",
    "SUBWORD_TRANSFORMER_MODEL_ID",
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
    "create_adapter",
    "create_model",
    "default_registry",
    "generate_structured",
    "load_model",
    "require_capability",
    "require_messages",
    "tokenize",
]
