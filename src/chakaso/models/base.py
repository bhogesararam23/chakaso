"""The language-model boundary.

Everything the application needs from a model, and nothing that belongs to one
particular model. Business logic depends on this module; it never imports a
concrete implementation (ADR-0002). Configuration selects an implementation in
exactly one place, :func:`chakaso.models.create_model`.

The boundary is deliberately narrow:

* :class:`LanguageModel` — the only thing every implementation must provide:
  metadata and generation from a message list.
* :class:`Tokenizer` and :class:`StructuredGenerator` — optional protocols for
  implementations that can do more.
* :class:`Capability` — how an implementation *declares* what it supports, so a
  caller can find out before asking rather than discovering it from a confusing
  failure.

Optional behaviour is expressed twice on purpose. The declaration lives in
metadata, where it can be inspected, logged and checked without touching the
implementation; the method lives on a separate protocol, so an implementation that
cannot do something is not forced to define a method that raises. The module-level
:func:`tokenize` and :func:`generate_structured` helpers are the enforcement point
that keeps the two from disagreeing.

The interface is not a place to expose a feature of one model. A parameter that
only one backend understands does not belong here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from chakaso.models.errors import ModelError, UnsupportedCapabilityError

__all__ = [
    "Capability",
    "FinishReason",
    "GenerationParams",
    "GenerationResult",
    "LanguageModel",
    "Message",
    "ModelMetadata",
    "Role",
    "StructuredGenerator",
    "StructuredResult",
    "Tokenizer",
    "generate_structured",
    "require_capability",
    "require_messages",
    "tokenize",
]


class Role(StrEnum):
    """Who produced a message."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    """One message handed to a model.

    This is the model's input type, not the application's conversation type.
    Conversation state carries identifiers, timestamps and provenance that a model
    has no use for, and is mapped onto these messages at the call site. Keeping the
    two apart means a change to conversation state cannot change what a model sees
    by accident.
    """

    role: Role
    content: str


class Capability(StrEnum):
    """Something a model may be able to do beyond generating text."""

    TOKENIZE = "tokenize"
    STRUCTURED_GENERATION = "structured_generation"


class FinishReason(StrEnum):
    """Why generation stopped.

    ``LENGTH`` means the output was cut off by the token ceiling, which is a
    different situation from a model that finished its answer, and a caller that
    cannot tell them apart will present a truncated answer as a complete one.
    """

    STOP = "stop"
    LENGTH = "length"


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Explicit generation parameters.

    Passed as one object rather than as keyword arguments so that adding a
    parameter later does not change the signature of every implementation, and so
    that validation happens once here instead of once per adapter.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0

    def __post_init__(self) -> None:
        # bool is a subclass of int, so True would otherwise be accepted as 1.
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int):
            message = (
                f"max_new_tokens must be an integer, got {type(self.max_new_tokens).__name__} "
                f"({self.max_new_tokens!r})"
            )
            raise ValueError(message)
        if self.max_new_tokens < 1:
            message = f"max_new_tokens must be at least 1, got {self.max_new_tokens}"
            raise ValueError(message)
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, int | float):
            message = (
                f"temperature must be a number, got {type(self.temperature).__name__} "
                f"({self.temperature!r})"
            )
            raise ValueError(message)
        if self.temperature < 0:
            message = f"temperature must not be negative, got {self.temperature}"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """What a model is, and what it can do.

    ``development_double`` marks an implementation that exists to exercise the
    application without being a language model. It is part of the metadata rather
    than a comment because documentation and reports must be able to say plainly
    that no model was involved in a result.
    """

    model_id: str
    display_name: str
    context_window: int
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    development_double: bool = False

    def __post_init__(self) -> None:
        if not self.model_id:
            message = "model_id must not be empty"
            raise ValueError(message)
        if self.context_window < 1:
            message = f"context_window must be at least 1, got {self.context_window}"
            raise ValueError(message)

    def supports(self, capability: Capability) -> bool:
        """Whether this model declares ``capability``."""
        return capability in self.capabilities

    def describe_capabilities(self) -> str:
        """Capabilities as a readable list, for error messages."""
        if not self.capabilities:
            return "none"
        return ", ".join(sorted(capability.value for capability in self.capabilities))


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Generated text and the little that must be known about how it was produced."""

    text: str
    model_id: str
    finish_reason: FinishReason = FinishReason.STOP


@dataclass(frozen=True, slots=True)
class StructuredResult:
    """A validated structured response.

    ``data`` is a mapping rather than a string because the caller asked for a
    structure, and re-parsing text at the call site would put the same parse in
    every caller.
    """

    data: Mapping[str, object]
    model_id: str


@runtime_checkable
class LanguageModel(Protocol):
    """The contract every model implementation satisfies.

    Only two members, because everything else is conditional on a capability.
    """

    @property
    def metadata(self) -> ModelMetadata:
        """Identity, context window and declared capabilities."""
        ...

    def generate(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
    ) -> GenerationResult:
        """Generate a continuation for ``messages``.

        Implementations must be deterministic for identical inputs when
        ``params.temperature`` is zero, because evaluation depends on a run being
        replayable.

        Raises:
            ModelError: the request cannot be served, for example because no
                messages were supplied.
        """
        ...


@runtime_checkable
class Tokenizer(Protocol):
    """Implemented by models that can tokenize.

    Declaring :attr:`Capability.TOKENIZE` without implementing this is caught by
    :func:`tokenize`.
    """

    def tokenize(self, text: str) -> list[int]:
        """Return the token identifiers for ``text``."""
        ...


@runtime_checkable
class StructuredGenerator(Protocol):
    """Implemented by models that can be constrained to a schema."""

    def generate_structured(
        self,
        messages: Sequence[Message],
        schema: Mapping[str, object],
        params: GenerationParams,
    ) -> StructuredResult:
        """Generate a response conforming to ``schema``."""
        ...


def require_messages(messages: Sequence[Message]) -> Sequence[Message]:
    """Return ``messages``, or raise if it is empty.

    Shared so that every implementation rejects the same request with the same
    error, rather than each inventing its own.
    """
    if not messages:
        message = "generation requires at least one message; the request contained none"
        raise ModelError(message)
    return messages


def require_capability(metadata: ModelMetadata, capability: Capability) -> None:
    """Raise unless ``metadata`` declares ``capability``.

    The message names the model and the capability, per ADR-0002, so the failure
    identifies what is missing rather than only that something is.
    """
    if not metadata.supports(capability):
        message = (
            f"model {metadata.model_id!r} ({metadata.display_name}) does not support "
            f"{capability.value!r}. Declared capabilities: {metadata.describe_capabilities()}."
        )
        raise UnsupportedCapabilityError(message)


def tokenize(model: LanguageModel, text: str) -> list[int]:
    """Tokenize ``text`` using ``model``, or fail clearly.

    Raises:
        UnsupportedCapabilityError: the model does not declare
            :attr:`Capability.TOKENIZE`, or declares it without implementing
            :class:`Tokenizer`. The second case is a defect in the implementation,
            and is reported as one rather than as a caller error.
    """
    require_capability(model.metadata, Capability.TOKENIZE)
    if not isinstance(model, Tokenizer):
        message = (
            f"model {model.metadata.model_id!r} declares the "
            f"{Capability.TOKENIZE.value!r} capability but does not implement tokenize(). "
            "Its metadata and its implementation disagree."
        )
        raise UnsupportedCapabilityError(message)
    return model.tokenize(text)


def generate_structured(
    model: LanguageModel,
    messages: Sequence[Message],
    schema: Mapping[str, object],
    params: GenerationParams | None = None,
) -> StructuredResult:
    """Generate a schema-constrained response using ``model``, or fail clearly.

    Raises:
        UnsupportedCapabilityError: the model does not declare
            :attr:`Capability.STRUCTURED_GENERATION`, or declares it without
            implementing :class:`StructuredGenerator`.
    """
    require_capability(model.metadata, Capability.STRUCTURED_GENERATION)
    if not isinstance(model, StructuredGenerator):
        message = (
            f"model {model.metadata.model_id!r} declares the "
            f"{Capability.STRUCTURED_GENERATION.value!r} capability but does not implement "
            "generate_structured(). Its metadata and its implementation disagree."
        )
        raise UnsupportedCapabilityError(message)
    return model.generate_structured(messages, schema, params or GenerationParams())
