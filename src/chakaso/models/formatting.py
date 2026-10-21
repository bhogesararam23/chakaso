"""Prompt formatting: the versioned contract between messages and the text a model is given.

A ``LanguageModel`` receives messages and produces text, and something has to decide how the first become
the second. Until now that decision was a line of code inside one adapter — ``"\\n".join(contents)`` — which
had three problems: it was not named, so a record could not say which prompts produced which samples; it was
not shared, so a second adapter could quietly choose differently; and it carried no role information at all,
so a system message and a user message reached the model identically.

This module names the decision and versions it.

**The default is the measured behaviour.** :data:`PLAIN_CONTINUATION` joins message contents with a newline
and nothing else, because every sample in every experiment record committed so far — E2605 through E2608 —
was generated that way. Making the new shape the default would silently change what those records' sample
texts mean, which is the same error as changing a metric after publishing it.

**Message formatting is not instruction tuning.** A template can tell a model "the user's turn is over,
yours begins", and a base next-token model trained on novels has no idea what that means. Rendering a
transcript in a chat shape does not make the model an assistant, and :func:`render` is documented to return
the text a *continuation* model can consume, not a prompt a chat model was trained to answer. The stop
sequences a chat-shaped template implies are reported alongside it so a caller can stop where a real
assistant's turn would end — a convention, not a capability.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from chakaso.models.base import Message, Role
from chakaso.models.errors import ModelError

__all__ = [
    "PLAIN_CONTINUATION",
    "PROMPT_TEMPLATES",
    "ROLE_LABELLED_CHAT",
    "PromptTemplate",
    "render",
    "template_for",
]


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """One named, versioned way to turn messages into model input.

    Attributes:
        version: Recorded with every generated sample, because a prompt shape is part of a result.
        label_message: Whether the template tells the model whose words each turn is.
        open_assistant_turn: Text appended to the end so the model's turn is the next thing in the stream.
        stop_sequences: Text that conventionally ends an assistant turn, for a caller to pass as stops.
    """

    version: str
    label_message: bool
    open_assistant_turn: bool
    stop_sequences: tuple[str, ...] = ()
    description: str = ""


#: The shape every measured Chakaso sample was produced with: contents joined by newlines, no roles.
PLAIN_CONTINUATION = PromptTemplate(
    version="chakaso-plain-v1",
    label_message=False,
    open_assistant_turn=False,
    stop_sequences=(),
    description=(
        "Message contents joined by newlines, oldest first. This is what E2605 through E2608 generated "
        "with, and it is the default so that choosing it is never a silent change to a recorded result."
    ),
)

#: A role-labelled transcript. Shaped for a future instruction-tuned checkpoint; honest for a base model.
ROLE_LABELLED_CHAT = PromptTemplate(
    version="chakaso-role-labelled-v1",
    label_message=True,
    open_assistant_turn=True,
    stop_sequences=("\nuser:", "\nsystem:", "\n\n"),
    description=(
        "Labels each turn with its role and ends with 'assistant: ' so the model's reply is the next text "
        "in the stream. This is a convention, not training: a base next-token model has been shown no "
        "instruction data, so nothing here makes it an assistant."
    ),
)

PROMPT_TEMPLATES: dict[str, PromptTemplate] = {
    template.version: template for template in (PLAIN_CONTINUATION, ROLE_LABELLED_CHAT)
}


def template_for(version: str) -> PromptTemplate:
    """Look a template up by its version string.

    Raises:
        ModelError: the version is not one this code defines. A configuration naming a template that does
            not exist must fail at startup rather than render something unexpected.
    """
    try:
        return PROMPT_TEMPLATES[version]
    except KeyError as exc:
        message = (
            f"unknown prompt template {version!r}; this build defines {sorted(PROMPT_TEMPLATES)}"
        )
        raise ModelError(message) from exc


def render(
    messages: Sequence[Message],
    *,
    template: PromptTemplate = PLAIN_CONTINUATION,
) -> str:
    """Render messages into one text for a model, under the named template.

    Empty-content messages are dropped rather than rendered as blank turns, so an empty system message does
    not become a stray newline the model reads as an answer. A request with nothing to say is refused: an
    empty prompt gives a continuation model nothing to continue, and generating from nothing is how a
    placeholder answer gets mistaken for a response.

    Raises:
        ModelError: no message survives rendering, or the template asks for a role the message does not have.
    """
    usable = [message for message in messages if message.content and message.content.strip()]
    if not usable:
        message = (
            "prompt formatting received no message with content; a model must be given something to "
            "continue"
        )
        raise ModelError(message)
    if not template.label_message:
        return "\n".join(item.content for item in usable)
    lines = [f"{_role_label(item.role)}: {item.content}" for item in usable]
    if template.open_assistant_turn:
        lines.append("assistant:")
    return "\n".join(lines) + " "


def _role_label(role: Role) -> str:
    try:
        return role.value
    except AttributeError as exc:
        message = f"a message carries something that is not a role: {role!r}"
        raise ModelError(message) from exc
