"""Tests for the versioned prompt contract and for what it refuses to claim.

The property that matters here is that a template is *named*: two records that cite different prompt shapes
are not comparable, and a model's sample text means nothing without the shape that produced it. The other
property is the honest one — a chat-shaped transcript is not instruction tuning, so the contract exposes the
turn convention it assumes as stop sequences and nothing more.
"""

from __future__ import annotations

import pytest

from chakaso.models.base import Message, Role
from chakaso.models.errors import ModelError
from chakaso.models.formatting import (
    PLAIN_CONTINUATION,
    PROMPT_TEMPLATES,
    ROLE_LABELLED_CHAT,
    render,
    template_for,
)


def _thread() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="Answer briefly."),
        Message(role=Role.USER, content="What is Chakaso?"),
        Message(role=Role.ASSISTANT, content="Chakaso is a language model."),
    ]


def test_the_default_template_is_the_one_every_measured_sample_used() -> None:
    messages = _thread()

    rendered = render(messages)

    assert rendered == "\n".join(item.content for item in messages)
    assert rendered == render(messages, template=PLAIN_CONTINUATION)


def test_the_plain_template_carries_no_role_information_and_no_stops() -> None:
    rendered = render(_thread(), template=PLAIN_CONTINUATION)

    assert "system" not in rendered
    assert "user" not in rendered
    assert PLAIN_CONTINUATION.stop_sequences == ()


def test_a_role_labelled_transcript_labels_each_turn_and_opens_the_reply() -> None:
    rendered = render(_thread(), template=ROLE_LABELLED_CHAT)

    assert rendered.startswith("system: Answer briefly.\n")
    assert "user: What is Chakaso?" in rendered
    assert rendered.endswith("assistant: ")


def test_a_chat_shaped_template_states_where_an_assistant_turn_would_end() -> None:
    assert ROLE_LABELLED_CHAT.stop_sequences == ("\nuser:", "\nsystem:", "\n\n")
    # The stop markers must be findable in the shape the template produces, or they are decoration.
    rendered = render(
        [Message(role=Role.USER, content="one"), Message(role=Role.USER, content="two")],
        template=ROLE_LABELLED_CHAT,
    )

    assert any(marker in rendered for marker in ROLE_LABELLED_CHAT.stop_sequences)


def test_templates_are_addressed_by_version_and_the_registry_names_both() -> None:
    assert set(PROMPT_TEMPLATES) == {"chakaso-plain-v1", "chakaso-role-labelled-v1"}
    assert template_for("chakaso-role-labelled-v1") is ROLE_LABELLED_CHAT


def test_an_unknown_template_version_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ModelError, match="unknown prompt template"):
        template_for("chakaso-instruct-v9")


def test_blank_turns_are_dropped_rather_than_rendered_as_empty_lines() -> None:
    rendered = render(
        [
            Message(role=Role.SYSTEM, content="   "),
            Message(role=Role.USER, content="real question"),
        ]
    )

    assert rendered == "real question"


def test_a_request_with_nothing_to_say_is_refused() -> None:
    with pytest.raises(ModelError, match="something to continue"):
        render([Message(role=Role.USER, content="")])

    with pytest.raises(ModelError, match="something to continue"):
        render([], template=ROLE_LABELLED_CHAT)


def test_the_two_templates_render_different_text_from_the_same_messages() -> None:
    messages = _thread()

    plain = render(messages)
    labelled = render(messages, template=ROLE_LABELLED_CHAT)

    assert plain != labelled
    assert labelled != plain
    # Both are deterministic: the same messages and template never drift between calls.
    assert render(messages, template=ROLE_LABELLED_CHAT) == labelled


def test_a_template_records_what_it_is_for() -> None:
    assert "E2605" in PLAIN_CONTINUATION.description
    assert "not training" in ROLE_LABELLED_CHAT.description.lower() or (
        "not train" in ROLE_LABELLED_CHAT.description.lower()
    )
