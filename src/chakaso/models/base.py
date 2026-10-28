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

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from chakaso.models.errors import ModelError, UnsupportedCapabilityError

__all__ = [
    "Capability",
    "FinishReason",
    "GenerationChunk",
    "GenerationParams",
    "GenerationResult",
    "LanguageModel",
    "Message",
    "ModelMetadata",
    "Role",
    "StreamingGenerator",
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
    STREAMING = "streaming"
    STRUCTURED_GENERATION = "structured_generation"


class FinishReason(StrEnum):
    """Why generation stopped.

    ``LENGTH`` means the output was cut off by the token ceiling, which is a
    different situation from a model that finished its answer, and a caller that
    cannot tell them apart will present a truncated answer as a complete one.
    ``CONTEXT`` is the third case that matters: the model's own window filled, so
    the run cannot continue no matter what the caller asked for. ``CANCELLED``
    means the caller stopped a stream.
    """

    STOP = "stop"
    LENGTH = "length"
    CONTEXT = "context"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class GenerationParams:
    """Explicit generation parameters.

    Passed as one object rather than as keyword arguments so that adding a
    parameter later does not change the signature of every implementation, and so
    that validation happens once here instead of once per adapter.

    Every field except the first two defaults to what the measured runs used — greedy, no truncation of the
    candidate set, no stop sequences beyond the model's own separator — so a caller that changes nothing gets
    the behaviour an existing report describes.

    Attributes:
        max_new_tokens: Ceiling on tokens this call may produce.
        temperature: 0.0 is greedy. Above it, candidates are drawn from the scaled distribution; the value
            is not a knob that makes a model better, and a repetition rate is measured, not implied by it.
        top_k: Keep only the k most likely candidates before sampling.
        top_p: Nucleus threshold — keep the smallest set whose cumulative probability reaches it.
        seed: Makes a sampling run reproducible. Left as ``None`` the draw is reproducible per request
            rather than per process, which is what the boundary has always promised.
        stop_sequences: Text that ends generation when it appears *in the output*.
        stop_token_ids: Identifier that ends generation when the model emits it, in addition to whatever
            the checkpoint's own separator means.
        repeat_penalty: Multiplicative penalty on logits of tokens already produced. Offered because a
            degenerate model needs it to be measured, not because it fixes one: a repetition penalty that
            makes looping text look varied is a disguise, so the diagnostics stay reported either way.
    """

    max_new_tokens: int = 256
    temperature: float = 0.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop_sequences: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    repeat_penalty: float = 1.0

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
        if self.temperature < 0 or self.temperature > 10.0:
            # The same bound the configuration schema states for ``model.temperature``, so a request and a
            # configuration cannot disagree about what a temperature is allowed to be.
            message = f"temperature must be in [0, 10], got {self.temperature}"
            raise ValueError(message)
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1
        ):
            message = f"top_k must be an integer >= 1 or None, got {self.top_k!r}"
            raise ValueError(message)
        if self.top_p is not None:
            if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
                message = f"top_p must be a number or None, got {self.top_p!r}"
                raise ValueError(message)
            if not 0.0 < float(self.top_p) <= 1.0:
                message = f"top_p must be in (0, 1], got {self.top_p}"
                raise ValueError(message)
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            message = f"seed must be an integer or None, got {self.seed!r}"
            raise ValueError(message)
        for name, value in (
            ("stop_sequences", self.stop_sequences),
            ("stop_token_ids", self.stop_token_ids),
        ):
            if not isinstance(value, tuple):
                message = f"{name} must be a tuple, got {type(value).__name__}"
                raise ValueError(message)
        if any(not isinstance(item, str) or not item for item in self.stop_sequences):
            message = f"stop_sequences must be non-empty strings, got {self.stop_sequences!r}"
            raise ValueError(message)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.stop_token_ids
        ):
            message = f"stop_token_ids must be non-negative integers, got {self.stop_token_ids!r}"
            raise ValueError(message)
        if isinstance(self.repeat_penalty, bool) or not isinstance(
            self.repeat_penalty, (int, float)
        ):
            message = f"repeat_penalty must be a number, got {self.repeat_penalty!r}"
            raise ValueError(message)
        if not 0.5 <= float(self.repeat_penalty) <= 3.0:
            message = f"repeat_penalty must be in [0.5, 3.0], got {self.repeat_penalty}"
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
    """Generated text and what must be known about how it was produced.

    The token counts and identifiers are part of the result rather than a bonus because a caller building a
    product needs to charge, budget and truncate in tokens, and because diagnostics on a degenerate model
    ("it produced 64 tokens, 61 of them repeats of one block") are only possible from the ids. ``truncated``
    says whether the *prompt* was cut to fit the window: a caller that cannot tell will present an answer to
    a question the model never saw.
    """

    text: str
    model_id: str
    finish_reason: FinishReason = FinishReason.STOP
    token_ids: tuple[int, ...] = ()
    new_tokens: int = 0
    prompt_tokens: int = 0
    truncated: bool = False
    stopped_by: str | None = None

    def __post_init__(self) -> None:
        if self.new_tokens != len(self.token_ids) and self.token_ids:
            message = (
                f"new_tokens ({self.new_tokens}) disagrees with the {len(self.token_ids)} identifiers "
                "the result carries"
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    """One streamed piece of a generation.

    Attributes:
        text: The text this step completes. A token that only begins a multi-byte character contributes an
            empty string and the character arrives whole with the token that finishes it; a run that ends
            inside a character delivers one replacement character on the final chunk.
        token_id: The identifier just produced.
        final: Set on the last chunk, which also carries the finish reason.
        finish_reason: Only meaningful when ``final`` is true.
    """

    text: str
    token_id: int
    final: bool = False
    finish_reason: FinishReason | None = None


@runtime_checkable
class StreamingGenerator(Protocol):
    """Implemented by models that can yield a generation as it is produced."""

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        signal: object | None = None,
    ) -> Iterator[GenerationChunk]:
        """Yield chunks until the run stops, a ceiling is reached, or ``signal`` is aborted.

        Raises:
            ModelError: the request cannot be served.
        """
        ...


@runtime_checkable
class TextScorer(Protocol):
    """Implemented by models that can return the boundary's own held-out measurement for a text.

    Separate from :class:`LanguageModel` because scoring is not something every adapter can do: the
    development double has no distribution to take a cross-entropy over, and a JSON character checkpoint
    predates the arithmetic. A caller that needs the number asks for this shape instead of assuming the
    adapter it was handed has it.
    """

    def score(self, text: str) -> float:
        """Mean next-token cross-entropy, in nats, over ``text`` at the model's own context length.

        Raises:
            ModelError: the text is empty or too short to fill one prediction window.
        """
        ...


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
