"""The development double.

This is not a language model. It produces a fixed, deterministic string so that
conversation, evidence and citation logic can be exercised — in tests and in a
local run — without weights, a GPU, or a download.

It exists because the alternative is worse. Without something behind the model
boundary, none of the application above it can be written or tested, and the
temptation is to reach for a hosted API, which ADR-0001 forbids. A double that is
obviously not a model, and says so in its own metadata, keeps the boundary
exercisable while keeping every claim about the system honest.

It declares no capabilities. It cannot tokenize and it cannot produce structured
output, so asking it for either fails at the boundary with a clear message rather
than returning something that looks like an answer.
"""

from __future__ import annotations

from collections.abc import Sequence

from chakaso.models.base import (
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
    require_messages,
)

__all__ = ["DETERMINISTIC_MODEL_ID", "DeterministicModel"]

#: The registry name this double is registered under, and the value
#: ``configs/default.toml`` selects until a real local adapter exists.
DETERMINISTIC_MODEL_ID = "deterministic"

_DISPLAY_NAME = "Deterministic double (not a language model)"

#: A conservative context window. The double has no real one, but callers budget
#: against this value, and a value that is too large would let a caller build a
#: request no real model could accept.
_CONTEXT_WINDOW = 2048


class DeterministicModel:
    """A stand-in for a language model that produces fixed text.

    Behaviour, stated precisely so that nothing here is mistaken for generation:

    * The output is a template naming the number of messages received and the
      length of the last one. It does not depend on the content of the messages
      beyond that length.
    * ``max_new_tokens`` is applied as a **character** limit, not a token limit,
      because this class does not tokenize. Truncation is reported as
      :attr:`FinishReason.LENGTH`.
    * ``temperature`` is ignored: there is no sampling to make stochastic, and
      pretending otherwise would be a lie about determinism.

    It is safe to use in tests that assert on plumbing and unsafe to use in
    anything that asserts on answer quality, because it has none.
    """

    def __init__(self, model_id: str = DETERMINISTIC_MODEL_ID) -> None:
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=_DISPLAY_NAME,
            context_window=_CONTEXT_WINDOW,
            capabilities=frozenset(),
            development_double=True,
        )

    @property
    def metadata(self) -> ModelMetadata:
        """Identity and capabilities of the double."""
        return self._metadata

    def generate(
        self,
        messages: Sequence[Message],
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        """Return the fixed double output, truncated to ``max_new_tokens`` characters."""
        require_messages(messages)
        resolved = params or GenerationParams()

        last = messages[-1]
        text = (
            f"{_DISPLAY_NAME}. It received {len(messages)} message(s); "
            f"the last was {len(last.content)} characters of {last.role.value!r} text."
        )

        if len(text) > resolved.max_new_tokens:
            return GenerationResult(
                text=text[: resolved.max_new_tokens],
                model_id=self._metadata.model_id,
                finish_reason=FinishReason.LENGTH,
            )
        return GenerationResult(text=text, model_id=self._metadata.model_id)
