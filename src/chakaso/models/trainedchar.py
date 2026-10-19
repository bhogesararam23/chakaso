"""A `LanguageModel` adapter over a trained character checkpoint (P25, ADR-0030).

This wires a *real* trained model — the artifact experiment E2601 produces — behind the boundary
(ADR-0002), so a trained Chakaso model can be selected through configuration rather than only measured
in a script. It loads a checkpoint, rebuilds the exact vocabulary bound to those weights, and generates
greedily, one character at a time, within the configured ceiling.

Honesty first: this is a tiny character model trained on a hand-written toy corpus. Its output is not
useful language — it is likely incoherent — and it must never be presented as a capable assistant. Its
value is that the training-and-inference path is real: the weights are this project's own, the
tokenizer is the one they were trained with, and generation is deterministic and bounded.

It depends on `numpy` and `chakaso.training` (the optional `[train]` extra). The model registry imports
*this module* lazily, so `import chakaso.models` never requires numpy; only selecting the trained-char
adapter does, and it then fails explicitly — never silently falling back to the development double — if
the dependency, the path, or the checkpoint is missing or invalid.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chakaso.models.base import (
    Capability,
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
    require_messages,
)
from chakaso.models.errors import ModelError
from chakaso.training.checkpoint import Checkpoint
from chakaso.training.errors import CheckpointError
from chakaso.training.model import CharCausalMLP
from chakaso.training.tokenizer import UNK_ID, CharVocabulary

__all__ = ["TRAINED_CHAR_MODEL_ID", "TrainedCharacterModel"]

#: The registry name for this adapter; the value ``model.adapter`` takes to select a trained model.
TRAINED_CHAR_MODEL_ID = "trained_char"

_DISPLAY_NAME = "Trained character model (E2601-class toy; not a useful language model)"


class TrainedCharacterModel:
    """A greedy character-level model backed by a trained checkpoint on disk."""

    def __init__(self, checkpoint_path: str, *, model_id: str = TRAINED_CHAR_MODEL_ID) -> None:
        if not checkpoint_path:
            message = (
                "the 'trained-char' adapter requires model.checkpoint_path to name a trained "
                "checkpoint, but none was configured; it does not fall back to the development double"
            )
            raise ModelError(message)

        path = Path(checkpoint_path)
        if not path.is_file():
            message = f"the 'trained-char' adapter found no checkpoint at {checkpoint_path!r}"
            raise ModelError(message)

        try:
            self._checkpoint = Checkpoint.load(path)
        except CheckpointError as exc:
            message = f"the 'trained-char' adapter could not load {checkpoint_path!r}: {exc}"
            raise ModelError(message) from exc

        self._vocabulary: CharVocabulary = self._checkpoint.vocabulary()
        self._context_size = self._checkpoint.config.context_size
        self._model = CharCausalMLP(
            vocab_size=self._checkpoint.vocab_size,
            context_size=self._context_size,
            embed_dim=self._checkpoint.config.embed_dim,
            hidden_dim=self._checkpoint.config.hidden_dim,
        )
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=_DISPLAY_NAME,
            # This model's real conditioning window is the character context it was trained with —
            # a genuine bound, unlike the double's placeholder, and the largest a request can use.
            context_window=self._context_size,
            capabilities=frozenset({Capability.TOKENIZE}),
            development_double=False,
        )

    @property
    def metadata(self) -> ModelMetadata:
        """Identity and the tokenizing capability this adapter really implements."""
        return self._metadata

    def tokenize(self, text: str) -> list[int]:
        """Return the identifier sequence for ``text`` using the checkpoint's own vocabulary."""
        return list(self._vocabulary.encode(text))

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        """Generate greedily from the last message, bounded by ``params.max_new_tokens`` characters.

        Generation is deterministic (argmax) at any temperature: this toy model implements no sampling,
        and it learns no stop token, so it always runs to the ceiling and reports
        :attr:`FinishReason.LENGTH` truthfully rather than pretending a natural stop.
        """
        require_messages(messages)

        window = list(self._vocabulary.encode(messages[-1].content))
        produced: list[int] = []
        for _ in range(params.max_new_tokens):
            predicted = self._next_identifier(window)
            produced.append(predicted)
            window.append(predicted)

        text = self._vocabulary.decode(tuple(produced))
        return GenerationResult(
            text=text, model_id=self._metadata.model_id, finish_reason=FinishReason.LENGTH
        )

    def _next_identifier(self, window: list[int]) -> int:
        recent = window[-self._context_size :] if window else []
        if len(recent) < self._context_size:
            recent = [UNK_ID] * (self._context_size - len(recent)) + recent
        contexts = np.array([recent], dtype=np.int64)
        probabilities = self._model.probabilities(self._checkpoint.params, contexts)
        return int(probabilities[0].argmax())
