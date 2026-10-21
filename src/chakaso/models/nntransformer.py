"""A `LanguageModel` adapter over a trained subword Transformer checkpoint.

This is what a trained Chakaso model looks like from the application side: weights this project trained,
its own tokenizer, greedy or sampled decoding, and a declaration of what it is. It is the second trained
adapter and the first with a subword vocabulary and attention.

Honesty about the artifact, stated in the adapter rather than only in the docs: the model behind this
class is trained on Chakaso's own documentation, on one machine, on CPU, at a scale measured in hundreds
of thousands of tokens. Its `display_name` says so, and its output must never be presented as a capable
assistant. What is real here is the path — a checkpoint on disk becomes text through a Python call — not a
claim of usefulness.

Decoding is a base completion, not a chat turn. A model trained on prose has never seen a role; the
adapter therefore conditions on the *concatenated* message contents and states that plainly, rather than
pretending a template it was not trained with.

Imports `torch` and `chakaso.training`, so it requires the optional ``[nn]`` extra. The registry imports
this module lazily: selecting the adapter needs the dependency, reading the registry does not, and a
missing dependency or checkpoint is a named `ModelError` — never a silent fall back to the development
double.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch

from chakaso.models.base import (
    Capability,
    FinishReason,
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
    Role,
    require_messages,
)
from chakaso.models.errors import ModelError
from chakaso.training.errors import CheckpointError, TrainingError
from chakaso.training.nn_checkpoint import ModelArtifact, load_artifact
from chakaso.training.subword import DOCUMENT_END

__all__ = ["SUBWORD_TRANSFORMER_MODEL_ID", "SubwordTransformerModel"]

#: The registry name for this adapter, and the value ``model.adapter`` takes to select it.
SUBWORD_TRANSFORMER_MODEL_ID = "subword_transformer"

_DISPLAY_NAME = (
    "Subword Transformer (Chakaso-trained character model successor; trained on the project's own "
    "documentation on CPU — a measured small model, not a capable assistant)"
)


def _replayable_seed(prompt: str, params: GenerationParams, top_k: int | None) -> int:
    """A seed derived from the request, so a sampled call repeats itself when asked identically.

    The boundary promises a run is replayable; sampling from process-global RNG state would break that for
    every adapter at once. Deriving the seed from the inputs keeps two identical requests identical without
    asking callers to manage a generator, and a *different* request still gets a different stream.
    """
    payload = f"{prompt}\x1f{params.temperature}\x1f{params.max_new_tokens}\x1f{top_k}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12], 16)


class SubwordTransformerModel:
    """Greedy or temperature sampling over a trained Transformer checkpoint."""

    def __init__(
        self,
        checkpoint_path: str,
        *,
        model_id: str = SUBWORD_TRANSFORMER_MODEL_ID,
        top_k: int | None = None,
    ) -> None:
        if not checkpoint_path:
            message = (
                "the 'subword_transformer' adapter requires a checkpoint path and none was given; it does "
                "not fall back to the development double"
            )
            raise ModelError(message)
        path = Path(checkpoint_path)
        if not path.is_file():
            message = (
                f"the 'subword_transformer' adapter found no checkpoint at {checkpoint_path!r}"
            )
            raise ModelError(message)
        try:
            self._artifact: ModelArtifact = load_artifact(path)
        except CheckpointError as exc:
            message = f"the 'subword_transformer' adapter could not load {checkpoint_path!r}: {exc}"
            raise ModelError(message) from exc

        self._model = self._artifact.to_model()
        self._context_length = self._artifact.config.context_length
        self._top_k = top_k
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=_DISPLAY_NAME,
            context_window=self._context_length,
            capabilities=frozenset({Capability.TOKENIZE}),
            development_double=False,
        )

    @property
    def metadata(self) -> ModelMetadata:
        """Identity, the real context window, and the tokenizing capability this adapter implements."""
        return self._metadata

    @property
    def artifact(self) -> ModelArtifact:
        """The loaded artifact: architecture, tokenizer, corpus fingerprint, training metadata."""
        return self._artifact

    def tokenize(self, text: str) -> list[int]:
        """Encode ``text`` with the vocabulary bound inside the checkpoint."""
        return list(self._artifact.vocabulary.encode(text))

    def generate(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
    ) -> GenerationResult:
        """Complete the conversation's text, bounded by ``params.max_new_tokens``.

        Deterministic at temperature zero (argmax). Above zero it samples, seeded from the request itself
        so an identical request reproduces and a different one does not.
        """
        return self._complete(messages, params, top_k=self._top_k)

    def sample(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Complete one bare prompt string — the shape an inference caller usually wants.

        Equivalent to :meth:`generate` with a single user message, and it shares the same decoding code, so
        there is one generation implementation rather than two that can disagree.
        """
        if not prompt:
            message = (
                "generation requires a non-empty prompt; an empty prompt has nothing to continue"
            )
            raise ModelError(message)
        return self._complete(
            [Message(role=Role.USER, content=prompt)],
            GenerationParams(max_new_tokens=max_new_tokens, temperature=temperature),
            top_k=top_k if top_k is not None else self._top_k,
        )

    def _complete(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        top_k: int | None,
    ) -> GenerationResult:
        require_messages(messages)
        prompt = "\n".join(message.content for message in messages if message.content)
        identifiers = list(self._artifact.vocabulary.encode(prompt))
        if not identifiers:
            message = (
                f"model {self._metadata.model_id!r} produced no input identifiers from the supplied text; "
                "the request contains nothing the tokenizer can encode"
            )
            raise ModelError(message)

        generator = torch.Generator().manual_seed(_replayable_seed(prompt, params, top_k))
        window = identifiers[-self._context_length :]
        produced: list[int] = []
        reason = FinishReason.LENGTH

        with torch.no_grad():
            for _ in range(params.max_new_tokens):
                input_ids = torch.tensor([window], dtype=torch.int64)
                logits = self._model.next_token_logits(input_ids)[0]
                if params.temperature <= 0.0:
                    nxt = int(logits.argmax())
                else:
                    probabilities = torch.softmax(logits / params.temperature, dim=-1)
                    if top_k is not None:
                        probabilities = _keep_top_k(probabilities, top_k)
                    nxt = int(torch.multinomial(probabilities, 1, generator=generator))
                if nxt == DOCUMENT_END:
                    # The separator is a boundary, not content: the model has signalled the end of a
                    # document, so generation stops and reports STOP rather than emitting it as text.
                    reason = FinishReason.STOP
                    break
                produced.append(nxt)
                window.append(nxt)
                if len(window) > self._context_length:
                    del window[0]

        text = self._artifact.vocabulary.decode(produced)
        return GenerationResult(text=text, model_id=self._metadata.model_id, finish_reason=reason)

    def score(self, text: str) -> float:
        """Mean next-token cross-entropy, in nats, over ``text`` packed at the model's own context.

        The measurement an evaluation caller wants from the boundary rather than from a training script:
        the same arithmetic the experiment uses, reachable from a loaded model.

        Raises:
            ModelError: the text is empty or too short to fill one prediction window.
        """
        from chakaso.training.data import TextUnit
        from chakaso.training.model_metrics import metrics_from_scores
        from chakaso.training.packing import pack_units

        if not text.strip():
            message = "cannot score an empty text"
            raise ModelError(message)
        identifiers = self._artifact.vocabulary.encode(text)
        if len(identifiers) <= self._context_length:
            message = (
                f"the supplied text encodes to {len(identifiers)} token(s), which is shorter than one "
                f"{self._context_length}-token window, so nothing can be scored"
            )
            raise ModelError(message)
        try:
            examples = pack_units(
                (TextUnit(identifier="score", text=text),),
                self._artifact.vocabulary,
                context_length=self._context_length,
            )
        except TrainingError as exc:
            # The packing layer's refusal is a specific, correct failure; the boundary reports its own
            # family, so the message is carried rather than replaced.
            message = f"model {self._metadata.model_id!r} cannot score the supplied text: {exc}"
            raise ModelError(message) from exc
        model = self._model
        nats: list[float] = []
        targets: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for example in examples.examples:
                input_ids = torch.tensor([list(example.input_ids)], dtype=torch.int64)
                logits, _ = model(input_ids)
                log_probabilities = torch.log_softmax(logits[0], dim=-1)
                for position, target in enumerate(example.targets):
                    token = int(target)
                    nats.append(-float(log_probabilities[position, token]))
                    targets.append(token)
                    predictions.append(int(log_probabilities[position].argmax()))
        return metrics_from_scores(nats, targets, predictions, self._artifact.vocabulary).loss_nats


def _keep_top_k(probabilities: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero every identifier outside the ``top_k`` most likely, then renormalize.

    Raises:
        ModelError: ``top_k`` is not a positive integer, or exceeds the model's vocabulary.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        message = f"top_k must be an integer >= 1, got {top_k!r}"
        raise ModelError(message)
    if top_k >= probabilities.numel():
        return probabilities
    kth = torch.topk(probabilities, top_k).values[-1]
    masked = torch.where(probabilities >= kth, probabilities, torch.zeros_like(probabilities))
    total = masked.sum()
    if float(total) <= 0.0:
        message = f"top_k={top_k} left no probability mass to sample from"
        raise ModelError(message)
    return masked / total
