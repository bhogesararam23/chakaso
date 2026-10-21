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
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import torch

from chakaso.models.base import (
    Capability,
    FinishReason,
    GenerationChunk,
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
    Role,
    require_messages,
)
from chakaso.models.decoding import AbortSignal, StopScanner, TokenSelector, truncate_prompt
from chakaso.models.errors import ModelError
from chakaso.training.decoder import ChakasoDecoder, KeyValueCache
from chakaso.training.errors import CheckpointError, TrainingError
from chakaso.training.nn_checkpoint import ModelArtifact, family_of, load_artifact
from chakaso.training.subword import DOCUMENT_END

__all__ = ["SUBWORD_TRANSFORMER_MODEL_ID", "SubwordTransformerModel"]

#: The registry name for this adapter, and the value ``model.adapter`` takes to select it.
SUBWORD_TRANSFORMER_MODEL_ID = "subword_transformer"

_DISPLAY_NAME = (
    "Subword Transformer (Chakaso-trained character model successor; trained on the project's own "
    "documentation on CPU — a measured small model, not a capable assistant)"
)


def _request_seed(prompt: str, params: GenerationParams) -> int:
    """The seed a sampled request draws from when the caller did not name one.

    Derived from the request rather than taken from process-global RNG state, because the boundary promises
    a replayable run: two identical calls must produce two identical samples, while a different prompt or a
    different ceiling must not be pinned to the same stream. An explicit ``params.seed`` overrides this and
    is the form a recorded experiment uses.
    """
    payload = "".join(
        (
            prompt,
            repr(params.temperature),
            str(params.max_new_tokens),
            str(params.top_k),
            str(params.top_p),
            str(params.stop_sequences),
            str(params.stop_token_ids),
            repr(params.repeat_penalty),
        )
    )
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
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
        ):
            message = (
                f"the 'subword_transformer' adapter was configured with top_k={top_k!r}; it must be an "
                "integer >= 1 or None"
            )
            raise ModelError(message)
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
        self._default_top_k = top_k
        #: Rows scored per forward pass when this model scores text. A loaded model scores in batches for the
        #: same reason a trained one does: one pass per block makes evaluation the expensive part of a call.
        self._scoring_batch = 32
        self._metadata = ModelMetadata(
            model_id=model_id,
            display_name=_DISPLAY_NAME,
            context_window=self._context_length,
            capabilities=frozenset({Capability.TOKENIZE, Capability.STREAMING}),
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
        """Complete the conversation's text under the parameters the caller declared.

        Deterministic at temperature zero (argmax over the penalized logits). Above zero it samples from the
        truncated candidate set, seeded from the request itself when ``params.seed`` is unset, so an identical
        request reproduces and a different one does not.
        """
        return self._decode(messages, params)

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        signal: AbortSignal | None = None,
    ) -> Iterator[GenerationChunk]:
        """Yield each generated step as it happens, and stop when the caller aborts.

        A chunk is one token's decoded text, which may be an incomplete multi-byte sequence: a caller
        rendering live output should accumulate and re-render, not trust one chunk. The final chunk carries
        ``finish_reason``, so a consumer can tell a complete answer from one cut off by the ceiling or the
        window.
        """
        collected: list[GenerationChunk] = []

        def emit(identifier: int, text: str) -> None:
            collected.append(GenerationChunk(text=text, token_id=identifier))

        result = self._decode(messages, params, on_token=emit, signal=signal)
        for index, chunk in enumerate(collected):
            last = index == len(collected) - 1
            yield (
                replace(chunk, final=True, finish_reason=result.finish_reason) if last else chunk
            )

    def sample(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        params: GenerationParams | None = None,
    ) -> GenerationResult:
        """Complete one bare prompt string — the shape an inference caller usually wants.

        Equivalent to :meth:`generate` with a single user message, and it shares the same decoding code, so
        there is one generation implementation rather than two that can disagree. Pass ``params`` for the
        full control set; the loose arguments are then refused rather than silently overridden.
        """
        if not prompt:
            message = (
                "generation requires a non-empty prompt; an empty prompt has nothing to continue"
            )
            raise ModelError(message)
        if params is not None:
            supplied = [
                name
                for name, value in (
                    ("max_new_tokens", max_new_tokens != 64),
                    ("temperature", temperature != 0.0),
                    ("top_k", top_k is not None),
                )
                if value
            ]
            if supplied:
                message = (
                    "sample() was given both a GenerationParams and "
                    f"{supplied}; the parameters object is the authority, so pass one or the other"
                )
                raise ModelError(message)
            chosen = params
        else:
            chosen = GenerationParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k if top_k is not None else self._default_top_k,
            )
        return self._decode([Message(role=Role.USER, content=prompt)], chosen)

    def _cache(self) -> KeyValueCache | None:
        """A fresh key/value cache when the loaded architecture supports one, otherwise nothing.

        The measured transformer has no incremental path — its ``forward`` re-runs the prefix — so a
        generation from its checkpoints keeps the older sliding-window behaviour instead of pretending to be
        cached. Which of the two ran is part of ``describe()``, because the difference is the difference
        between linear and quadratic generation cost.
        """
        if isinstance(self._model, ChakasoDecoder):
            return KeyValueCache(self._model.config.num_layers)
        return None

    def _decode(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        *,
        on_token: Callable[[int, str], None] | None = None,
        signal: AbortSignal | None = None,
    ) -> GenerationResult:
        require_messages(messages)
        prompt = "\n".join(message.content for message in messages if message.content)
        encoded = list(self._artifact.vocabulary.encode(prompt))
        if not encoded:
            message = (
                f"model {self._metadata.model_id!r} produced no input identifiers from the supplied text; "
                "the request contains nothing the tokenizer can encode"
            )
            raise ModelError(message)

        window, truncated = truncate_prompt(encoded, self._context_length)
        vocabulary = self._artifact.vocabulary
        selector = TokenSelector(
            params,
            vocabulary_size=vocabulary.size,
            rng=random.Random(
                params.seed if params.seed is not None else _request_seed(prompt, params)
            ),
        )
        scanner = StopScanner(
            sequences=params.stop_sequences,
            token_ids=params.stop_token_ids,
            document_end_id=DOCUMENT_END,
        )
        cache = self._cache()
        produced: list[int] = []
        texts: list[str] = []
        reason = FinishReason.LENGTH
        stopped_by: str | None = None

        with torch.no_grad():
            # The first step consumes the whole prompt; a cached step then consumes only the token it just
            # produced, which is the entire point of carrying the cache.
            step_input: Sequence[int] = window
            for _ in range(params.max_new_tokens):
                if signal is not None and signal.cancelled:
                    reason = FinishReason.CANCELLED
                    break
                logits = self._logits(step_input, cache)
                identifier = selector.select(logits)
                if identifier == DOCUMENT_END:
                    # The separator is a boundary, not content: the model signalled the end of a document,
                    # so generation stops and reports STOP rather than emitting it as text.
                    reason = FinishReason.STOP
                    stopped_by = "document-end"
                    break
                verdict = scanner.feed(vocabulary.decode([identifier]), identifier)
                selector.note(identifier)
                produced.append(identifier)
                chunk = vocabulary.decode([identifier])
                texts.append(chunk)
                if on_token is not None:
                    on_token(identifier, chunk)
                if cache is None:
                    step_input = (*step_input, identifier)[-self._context_length :]
                else:
                    step_input = (identifier,)
                    if cache.length >= self._context_length:
                        # A cached run cannot slide its window: the positions already computed are what the
                        # attention is conditioned on. Saying so is better than silently truncating the
                        # beginning of a long answer.
                        reason = FinishReason.CONTEXT
                        break
                if verdict.stopped:
                    reason = FinishReason.STOP
                    stopped_by = verdict.marker
                    break

        return GenerationResult(
            text="".join(texts),
            model_id=self._metadata.model_id,
            finish_reason=reason,
            token_ids=tuple(produced),
            new_tokens=len(produced),
            prompt_tokens=len(window),
            truncated=truncated,
            stopped_by=stopped_by,
        )

    def _logits(self, identifiers: Sequence[int], cache: KeyValueCache | None) -> list[float]:
        """One row of logits for the next token, from either architecture family."""
        row = torch.tensor([list(identifiers)], dtype=torch.int64)
        if cache is None:
            return [float(value) for value in self._model.next_token_logits(row)[0]]
        logits, _ = self._model(row, cache=cache)
        return [float(value) for value in logits[0, -1]]

    def describe(self) -> dict[str, object]:
        """Runtime facts a caller needs before choosing parameters, read from the artifact."""
        return {
            "model_id": self._metadata.model_id,
            "family": family_of(self._artifact.config),
            "parameter_count": self._artifact.parameter_count,
            "context_window": self._context_length,
            "vocabulary_size": self._artifact.vocabulary.size,
            "corpus_fingerprint": self._artifact.corpus_fingerprint,
            "tokenizer_fingerprint": self._artifact.vocabulary.fingerprint,
            "checkpoint": self._artifact.checksum()[:16],
            "cached_decoding": self._cache() is not None,
            "development_double": False,
        }

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
        from chakaso.training.nn_evaluation import score_model_on_examples

        # One scoring implementation, shared with the experiment's held-out numbers: a boundary that scored
        # text its own way would let a report and a caller disagree about the same checkpoint.
        with torch.no_grad():
            nats, targets, predictions = score_model_on_examples(self._model, examples.examples)
        return metrics_from_scores(nats, targets, predictions, self._artifact.vocabulary).loss_nats
