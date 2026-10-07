"""Tests for the deterministic development double.

The double satisfies the model contract (inherited below) plus the specific
behaviour it documents: fixed output, character-based truncation, and no
capabilities. Those details matter because a double that quietly behaves like a
model would let application tests pass against something that is not there.
"""

from __future__ import annotations

import pytest

from chakaso.models import (
    DETERMINISTIC_MODEL_ID,
    Capability,
    DeterministicModel,
    FinishReason,
    GenerationParams,
    Message,
    Role,
)
from model_contracts import LanguageModelContract


class TestDeterministicModel(LanguageModelContract):
    expected_capabilities = frozenset()

    @pytest.fixture
    def model(self) -> DeterministicModel:
        return DeterministicModel()


def test_metadata_marks_it_as_a_development_double() -> None:
    # This flag is what lets documentation and reports say plainly that no model
    # was involved. It is metadata rather than a comment for that reason.
    metadata = DeterministicModel().metadata

    assert metadata.development_double is True
    assert metadata.model_id == DETERMINISTIC_MODEL_ID


def test_display_name_says_it_is_not_a_language_model() -> None:
    assert "not a language model" in DeterministicModel().metadata.display_name.lower()


def test_it_declares_no_capabilities() -> None:
    assert DeterministicModel().metadata.capabilities == frozenset()


def test_output_reports_the_message_count_and_last_length() -> None:
    model = DeterministicModel()
    messages = [
        Message(role=Role.SYSTEM, content="Be brief."),
        Message(role=Role.USER, content="0123456789"),
    ]

    result = model.generate(messages, GenerationParams())

    assert "2 message(s)" in result.text
    assert "10 characters" in result.text
    assert "'user'" in result.text


def test_output_does_not_depend_on_message_content_beyond_length() -> None:
    # State it as an assertion rather than a comment: if the double ever starts
    # looking at content, a test that assumes it does not would be misleading.
    model = DeterministicModel()
    first = model.generate([Message(role=Role.USER, content="aaaa")], GenerationParams())
    second = model.generate([Message(role=Role.USER, content="bbbb")], GenerationParams())

    assert first.text == second.text


def test_truncation_is_reported_as_length() -> None:
    model = DeterministicModel()
    params = GenerationParams(max_new_tokens=12)

    result = model.generate([Message(role=Role.USER, content="hello")], params)

    assert len(result.text) == 12
    assert result.finish_reason is FinishReason.LENGTH


def test_no_truncation_is_reported_as_stop() -> None:
    model = DeterministicModel()
    result = model.generate([Message(role=Role.USER, content="hello")], GenerationParams())

    assert result.finish_reason is FinishReason.STOP


def test_temperature_is_ignored_and_output_stays_identical() -> None:
    # There is no sampling to make stochastic. Pretending otherwise would be a lie
    # about determinism, which evaluation relies on.
    model = DeterministicModel()
    messages = [Message(role=Role.USER, content="hello")]

    greedy = model.generate(messages, GenerationParams(temperature=0.0))
    hot = model.generate(messages, GenerationParams(temperature=2.0))

    assert greedy.text == hot.text


def test_default_params_are_used_when_none_are_given() -> None:
    model = DeterministicModel()
    explicit = model.generate([Message(role=Role.USER, content="hi")], GenerationParams())
    implicit = model.generate([Message(role=Role.USER, content="hi")])

    assert explicit == implicit


def test_it_cannot_tokenize() -> None:
    from chakaso.models import tokenize

    with pytest.raises(Exception, match="deterministic"):
        tokenize(DeterministicModel(), "text")


def test_it_cannot_produce_structured_output() -> None:
    from chakaso.models import generate_structured

    with pytest.raises(Exception, match=Capability.STRUCTURED_GENERATION.value):
        generate_structured(DeterministicModel(), [], {"type": "object"})


def test_model_id_can_be_overridden_for_distinguishing_instances() -> None:
    assert DeterministicModel(model_id="double_a").metadata.model_id == "double_a"
