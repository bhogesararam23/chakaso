"""The contract every language-model implementation must satisfy.

This module is not collected as tests: its name does not match pytest's default
file pattern, and its class is not named ``Test*``. Implementations subclass
:class:`LanguageModelContract` and inherit these assertions, so that a new adapter
cannot pass its own tests while violating the boundary.

Subclasses must provide the ``model`` fixture and declare what the implementation
claims, via ``expected_capabilities``.
"""

from __future__ import annotations

import pytest

from chakaso.models import (
    Capability,
    GenerationParams,
    LanguageModel,
    Message,
    Role,
    generate_structured,
    tokenize,
)


class LanguageModelContract:
    """Assertions that hold for any :class:`~chakaso.models.LanguageModel`."""

    #: Capabilities the implementation under test is expected to declare.
    expected_capabilities: frozenset[Capability] = frozenset()

    @pytest.fixture
    def model(self) -> LanguageModel:
        """The implementation under test."""
        raise NotImplementedError

    @pytest.fixture
    def messages(self) -> list[Message]:
        return [Message(role=Role.USER, content="What does the specification say?")]

    def test_metadata_is_populated(self, model: LanguageModel) -> None:
        metadata = model.metadata

        assert metadata.model_id
        assert metadata.display_name
        assert metadata.context_window >= 1

    def test_declared_capabilities_are_what_the_implementation_claims(
        self, model: LanguageModel
    ) -> None:
        assert model.metadata.capabilities == self.expected_capabilities

    def test_generated_result_names_the_model_that_produced_it(
        self, model: LanguageModel, messages: list[Message]
    ) -> None:
        # Provenance: an answer's record must identify the model that produced it,
        # so the result cannot be a bare string.
        result = model.generate(messages, GenerationParams())

        assert result.model_id == model.metadata.model_id

    def test_generate_returns_text(self, model: LanguageModel, messages: list[Message]) -> None:
        result = model.generate(messages, GenerationParams())

        assert isinstance(result.text, str)
        assert result.text

    def test_generate_is_repeatable_at_zero_temperature(
        self, model: LanguageModel, messages: list[Message]
    ) -> None:
        # Evaluation depends on a run being replayable.
        first = model.generate(messages, GenerationParams(temperature=0.0))
        second = model.generate(messages, GenerationParams(temperature=0.0))

        assert first == second

    def test_generate_rejects_an_empty_message_list(self, model: LanguageModel) -> None:
        with pytest.raises(Exception, match="at least one message"):
            model.generate([], GenerationParams())

    def test_generate_honours_the_length_limit(
        self, model: LanguageModel, messages: list[Message]
    ) -> None:
        # Either the output fits, or it was truncated and says so. Silently
        # returning more than the ceiling would make context budgeting a fiction.
        params = GenerationParams(max_new_tokens=8)
        result = model.generate(messages, params)

        assert len(result.text) <= params.max_new_tokens or result.finish_reason.name == "LENGTH"

    def test_unsupported_capabilities_fail_at_the_boundary(self, model: LanguageModel) -> None:
        # Asking for something the model does not declare must fail with a message
        # naming the model, not silently return something plausible.
        for capability in Capability:
            if capability in self.expected_capabilities:
                continue
            with pytest.raises(Exception, match=model.metadata.model_id):
                if capability is Capability.TOKENIZE:
                    tokenize(model, "text")
                else:
                    generate_structured(model, [], {})

    def test_declared_capabilities_are_actually_implemented(self, model: LanguageModel) -> None:
        # A model that declares a capability it cannot perform is worse than one
        # that does not declare it, because a caller has no way to plan around it.
        if Capability.TOKENIZE in self.expected_capabilities:
            assert tokenize(model, "text")
        if Capability.STRUCTURED_GENERATION in self.expected_capabilities:
            assert generate_structured(model, [], {}) is not None
