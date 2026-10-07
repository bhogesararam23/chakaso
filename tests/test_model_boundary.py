"""Tests for the model boundary: capabilities, helpers and parameter validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from chakaso.models import (
    Capability,
    FinishReason,
    GenerationParams,
    GenerationResult,
    LanguageModel,
    Message,
    ModelMetadata,
    Role,
    StructuredResult,
    generate_structured,
    require_capability,
    require_messages,
    tokenize,
)
from chakaso.models.errors import ModelError, UnsupportedCapabilityError


class _MinimalModel:
    """A model that generates, and nothing else."""

    def __init__(self, capabilities: frozenset[Capability] = frozenset()) -> None:
        self._metadata = ModelMetadata(
            model_id="minimal",
            display_name="Minimal test model",
            context_window=128,
            capabilities=capabilities,
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def generate(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,  # noqa: ARG002 - protocol shape
    ) -> GenerationResult:
        require_messages(messages)
        return GenerationResult(text="ok", model_id=self._metadata.model_id)


class _LyingTokenizer:
    """Declares the tokenize capability without implementing it.

    This is a defect in an implementation, and the boundary must report it as one
    rather than as a caller error.
    """

    def __init__(self) -> None:
        self._metadata = ModelMetadata(
            model_id="liar",
            display_name="Declares more than it does",
            context_window=128,
            capabilities=frozenset({Capability.TOKENIZE}),
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def generate(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,  # noqa: ARG002 - protocol shape
    ) -> GenerationResult:
        require_messages(messages)
        return GenerationResult(text="ok", model_id=self._metadata.model_id)


class _FullModel(_MinimalModel):
    """A model that generates, tokenizes and produces structured output."""

    def __init__(self) -> None:
        super().__init__(frozenset({Capability.TOKENIZE, Capability.STRUCTURED_GENERATION}))

    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def generate_structured(
        self,
        messages: Sequence[Message],
        schema: Mapping[str, object],  # noqa: ARG002 - protocol shape
        params: GenerationParams | None = None,  # noqa: ARG002 - protocol shape
    ) -> StructuredResult:
        require_messages(messages)
        return StructuredResult(data={"echo": len(messages)}, model_id=self.metadata.model_id)


def test_protocol_is_satisfied_structurally() -> None:
    # ADR-0002 chose a structural interface over inheritance, so an implementation
    # must not have to declare anything to be accepted.
    assert isinstance(_MinimalModel(), LanguageModel)


def test_supports_reports_declared_capabilities() -> None:
    assert _MinimalModel().metadata.supports(Capability.TOKENIZE) is False
    assert _FullModel().metadata.supports(Capability.TOKENIZE) is True


def test_describe_capabilities_is_readable() -> None:
    assert _MinimalModel().metadata.describe_capabilities() == "none"
    assert _FullModel().metadata.describe_capabilities() == "structured_generation, tokenize"


def test_require_capability_names_the_model_and_the_capability() -> None:
    metadata = _MinimalModel().metadata

    with pytest.raises(UnsupportedCapabilityError) as info:
        require_capability(metadata, Capability.TOKENIZE)

    message = str(info.value)
    assert "minimal" in message
    assert "tokenize" in message
    assert "none" in message


def test_require_capability_passes_when_declared() -> None:
    require_capability(_FullModel().metadata, Capability.TOKENIZE)


def test_tokenize_delegates_when_implemented() -> None:
    assert tokenize(_FullModel(), "ab") == [97, 98]


def test_tokenize_rejects_a_model_without_the_capability() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="does not support"):
        tokenize(_MinimalModel(), "text")


def test_tokenize_reports_metadata_and_implementation_disagreeing() -> None:
    # A model declaring a capability it does not implement is a defect, and the
    # message must say so rather than blaming the caller.
    with pytest.raises(UnsupportedCapabilityError, match="disagree"):
        tokenize(_LyingTokenizer(), "text")


def test_generate_structured_delegates_when_implemented() -> None:
    result = generate_structured(
        _FullModel(), [Message(role=Role.USER, content="hi")], {"type": "object"}
    )

    assert result.data == {"echo": 1}
    assert result.model_id == "minimal"


def test_generate_structured_rejects_a_model_without_the_capability() -> None:
    with pytest.raises(UnsupportedCapabilityError, match="structured_generation"):
        generate_structured(_MinimalModel(), [], {"type": "object"})


def test_generate_structured_reports_metadata_and_implementation_disagreeing() -> None:
    class _Liar(_MinimalModel):
        def __init__(self) -> None:
            super().__init__(frozenset({Capability.STRUCTURED_GENERATION}))

    with pytest.raises(UnsupportedCapabilityError, match="disagree"):
        generate_structured(_Liar(), [], {"type": "object"})


def test_require_messages_rejects_an_empty_request() -> None:
    with pytest.raises(ModelError, match="at least one message"):
        require_messages([])


def test_require_messages_returns_what_it_was_given() -> None:
    messages = [Message(role=Role.USER, content="hi")]

    assert require_messages(messages) is messages


@pytest.mark.parametrize("value", [0, -1])
def test_max_new_tokens_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        GenerationParams(max_new_tokens=value)


def test_max_new_tokens_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        GenerationParams(max_new_tokens="many")  # type: ignore[arg-type]


def test_boolean_is_not_accepted_as_max_new_tokens() -> None:
    with pytest.raises(ValueError, match="must be an integer"):
        GenerationParams(max_new_tokens=True)


def test_negative_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        GenerationParams(temperature=-0.1)


def test_boolean_is_not_accepted_as_temperature() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        GenerationParams(temperature=True)


def test_integer_temperature_is_accepted() -> None:
    assert GenerationParams(temperature=1) == GenerationParams(temperature=1.0)


def test_generation_params_are_frozen_and_comparable() -> None:
    assert GenerationParams() == GenerationParams(max_new_tokens=256, temperature=0.0)
    assert GenerationParams() != GenerationParams(max_new_tokens=257)


def test_metadata_rejects_an_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        ModelMetadata(model_id="", display_name="x", context_window=1)


def test_metadata_rejects_a_non_positive_context_window() -> None:
    with pytest.raises(ValueError, match="context_window"):
        ModelMetadata(model_id="x", display_name="x", context_window=0)


def test_default_finish_reason_is_stop() -> None:
    assert GenerationResult(text="x", model_id="m").finish_reason is FinishReason.STOP
