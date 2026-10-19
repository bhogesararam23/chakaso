"""A deliberately tiny causal character model, with hand-derived gradients.

The model is a shallow feed-forward network over a fixed window of previous characters:

* each context character is looked up in an embedding table and the window's embeddings are
  concatenated into one vector;
* a single hidden layer applies ``tanh``;
* an output layer produces one logit per vocabulary identifier, read as a distribution over the next
  character with a softmax, trained with cross-entropy.

This is not a Transformer and does not pretend to be one. It is the smallest architecture that has an
embedding, a non-linear hidden layer and a softmax over a learned vocabulary — enough that a real
optimization procedure has something non-trivial to fit, and small enough that *every* gradient below
is derived by hand and can be checked against a finite-difference estimate by test. That last property
is the reason this file computes gradients explicitly instead of delegating to an autograd engine: for
the first measured result, the project wants the update rule to be a verified fact of its own code, not
a behaviour inherited from a third-party library (ADR-0030).

Like the rest of ``chakaso.training`` this module depends on ``numpy``, which is an optional
``[train]`` extra; importing it requires that extra, but nothing above the model boundary imports
*this*, so ``import chakaso`` stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from chakaso.training.data import Example

__all__ = ["CharCausalMLP", "Params", "batches_from_examples"]

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class Params:
    """Every trainable array of the model, as one immutable bundle.

    Shapes are fixed by the architecture: ``embedding`` is ``(V, d)``, ``hidden_weight`` is
    ``(c*d, h)``, ``hidden_bias`` is ``(h,)``, ``output_weight`` is ``(h, V)`` and ``output_bias`` is
    ``(V,)`` — for vocabulary size ``V``, context size ``c``, embedding dimension ``d`` and hidden
    dimension ``h``.
    """

    embedding: F64
    hidden_weight: F64
    hidden_bias: F64
    output_weight: F64
    output_bias: F64

    def zeros_like(self) -> Params:
        """A parameter bundle of the same shapes filled with zeros, for accumulating gradients."""
        return Params(
            embedding=np.zeros_like(self.embedding),
            hidden_weight=np.zeros_like(self.hidden_weight),
            hidden_bias=np.zeros_like(self.hidden_bias),
            output_weight=np.zeros_like(self.output_weight),
            output_bias=np.zeros_like(self.output_bias),
        )

    def items(self) -> tuple[tuple[str, F64], ...]:
        """The arrays paired with stable names, for checkpointing and gradient checks."""
        return (
            ("embedding", self.embedding),
            ("hidden_weight", self.hidden_weight),
            ("hidden_bias", self.hidden_bias),
            ("output_weight", self.output_weight),
            ("output_bias", self.output_bias),
        )


def batches_from_examples(examples: tuple[Example, ...], context_size: int) -> tuple[I64, I64]:
    """Turn windowed examples into ``(contexts, targets)`` index arrays for the model.

    ``contexts`` is an integer matrix of shape ``(n, context_size)`` and ``targets`` a vector of
    ``(n,)``. The rows are ordered exactly as supplied, so a caller controls the example order and a
    run stays reproducible.
    """
    if not examples:
        message = "cannot build a batch from zero examples"
        raise ValueError(message)

    contexts = np.array([ex.context for ex in examples], dtype=np.int64)
    targets = np.array([ex.target for ex in examples], dtype=np.int64)
    if contexts.shape[1] != context_size:
        message = f"expected context width {context_size}, got {contexts.shape[1]}"
        raise ValueError(message)
    return contexts, targets


@dataclass(frozen=True, slots=True)
class _ForwardCache:
    """Intermediate activations kept between the forward pass and its matching backward pass."""

    contexts: I64
    targets: I64
    window: F64
    hidden: F64
    probabilities: F64


class CharCausalMLP:
    """The tiny character model: initialization, forward loss, and hand-derived gradients."""

    __slots__ = ("context_size", "embed_dim", "hidden_dim", "vocab_size")

    def __init__(
        self, *, vocab_size: int, context_size: int, embed_dim: int, hidden_dim: int
    ) -> None:
        for name, value in (
            ("vocab_size", vocab_size),
            ("context_size", context_size),
            ("embed_dim", embed_dim),
            ("hidden_dim", hidden_dim),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                message = f"{name} must be an integer >= 1, got {value!r}"
                raise ValueError(message)

        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    @property
    def parameter_count(self) -> int:
        """Total number of trainable scalars — recorded so the experiment can state model size."""
        input_dim = self.context_size * self.embed_dim
        return (
            self.vocab_size * self.embed_dim  # embedding
            + input_dim * self.hidden_dim  # hidden weight
            + self.hidden_dim  # hidden bias
            + self.hidden_dim * self.vocab_size  # output weight
            + self.vocab_size  # output bias
        )

    def initialize(self, rng: np.random.Generator) -> Params:
        """Draw parameters from ``rng`` at random initialization.

        Embeddings and weights use a bounded uniform scaled to the fan-in, the standard small-random
        init for this kind of net; biases start at zero. Every draw comes from the passed generator, so
        the caller's seed fully determines the starting point.
        """
        input_dim = self.context_size * self.embed_dim
        embedding_limit = 0.5
        hidden_limit = 1.0 / input_dim**0.5
        output_limit = 1.0 / self.hidden_dim**0.5
        return Params(
            embedding=rng.uniform(
                -embedding_limit, embedding_limit, (self.vocab_size, self.embed_dim)
            ),
            hidden_weight=rng.uniform(-hidden_limit, hidden_limit, (input_dim, self.hidden_dim)),
            hidden_bias=np.zeros(self.hidden_dim),
            output_weight=rng.uniform(
                -output_limit, output_limit, (self.hidden_dim, self.vocab_size)
            ),
            output_bias=np.zeros(self.vocab_size),
        )

    def forward(self, params: Params, contexts: I64, targets: I64) -> float:
        """Return the mean cross-entropy loss for ``contexts`` predicting ``targets``."""
        _, loss = self._forward_with_cache(params, contexts, targets)
        return loss

    def probabilities(self, params: Params, contexts: I64) -> F64:
        """Return the next-token distribution for each row of ``contexts``.

        The forward pass up to the softmax, without needing targets — used for accuracy and, later, for
        generation. Each row sums to one.
        """
        self._validate_contexts(contexts)
        window = np.concatenate(
            [params.embedding[contexts[:, j]] for j in range(self.context_size)], axis=1
        )
        hidden = np.tanh(window @ params.hidden_weight + params.hidden_bias)
        logits = hidden @ params.output_weight + params.output_bias
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probabilities: F64 = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
        return probabilities

    def _forward_with_cache(
        self, params: Params, contexts: I64, targets: I64
    ) -> tuple[_ForwardCache, float]:
        self._validate_batch(contexts, targets)

        # Concatenate the window embeddings: for each row, the embeddings of its context_size ids.
        rows = contexts.shape[0]
        window = np.concatenate(
            [params.embedding[contexts[:, j]] for j in range(self.context_size)], axis=1
        )
        pre_hidden = window @ params.hidden_weight + params.hidden_bias
        hidden = np.tanh(pre_hidden)
        logits = hidden @ params.output_weight + params.output_bias

        # log-softmax with the max subtracted for numerical stability; never exponentiating raw logits.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        normalizer = exp_shifted.sum(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(normalizer)
        probabilities = exp_shifted / normalizer

        target_log_probs = log_probabilities[np.arange(rows), targets]
        loss = float(-np.mean(target_log_probs))

        cache = _ForwardCache(
            contexts=contexts,
            targets=targets,
            window=window,
            hidden=hidden,
            probabilities=probabilities,
        )
        return cache, loss

    def gradient(self, params: Params, contexts: I64, targets: I64) -> Params:
        """Return the gradient of the mean cross-entropy loss with respect to every parameter.

        The backward pass is derived by hand, term by term, and is validated by finite differences in
        the test suite. A softmax cross-entropy at the output collapses to ``probs - onehot(target)``,
        scaled by ``1 / batch``; the rest is the chain rule through the two linear layers and the
        ``tanh``.
        """
        cache, _ = self._forward_with_cache(params, contexts, targets)
        rows = cache.contexts.shape[0]
        scale = 1.0 / rows

        # dL/dlogits = (probs - onehot(target)) / rows
        d_logits = cache.probabilities.copy()
        d_logits[np.arange(rows), cache.targets] -= 1.0
        d_logits *= scale

        d_output_weight = cache.hidden.T @ d_logits
        d_output_bias = d_logits.sum(axis=0)

        d_hidden = d_logits @ params.output_weight.T
        d_pre_hidden = d_hidden * (1.0 - cache.hidden**2)

        d_hidden_weight = cache.window.T @ d_pre_hidden
        d_hidden_bias = d_pre_hidden.sum(axis=0)

        d_window = d_pre_hidden @ params.hidden_weight.T

        # Scatter the concatenated-window gradient back into the embedding rows it came from, summed
        # over every position that used a given identifier. np.add.at is required, not `+=` on a fancy
        # index: identifiers repeat within a batch, and buffered fancy `+=` would apply a repeated row
        # only once and silently undercount the gradient — the finite-difference test catches exactly
        # this class of error.
        d_embedding = np.zeros_like(params.embedding)
        for j in range(self.context_size):
            ids = cache.contexts[:, j]
            d_slice = d_window[:, j * self.embed_dim : (j + 1) * self.embed_dim]
            np.add.at(d_embedding, ids, d_slice)

        return Params(
            embedding=d_embedding,
            hidden_weight=d_hidden_weight,
            hidden_bias=d_hidden_bias,
            output_weight=d_output_weight,
            output_bias=d_output_bias,
        )

    def _validate_batch(self, contexts: I64, targets: I64) -> None:
        self._validate_contexts(contexts)
        if targets.ndim != 1 or targets.shape[0] != contexts.shape[0]:
            message = f"targets must be a vector of length {contexts.shape[0]}"
            raise ValueError(message)
        if targets.min() < 0 or targets.max() >= self.vocab_size:
            message = "a target identifier is outside the vocabulary range"
            raise ValueError(message)

    def _validate_contexts(self, contexts: I64) -> None:
        if contexts.ndim != 2 or contexts.shape[1] != self.context_size:
            message = f"contexts must have shape (n, {self.context_size}), got {contexts.shape}"
            raise ValueError(message)
        if contexts.shape[0] == 0:
            message = "a batch must contain at least one example"
            raise ValueError(message)
        if contexts.min() < 0 or contexts.max() >= self.vocab_size:
            message = "a context identifier is outside the vocabulary range"
            raise ValueError(message)
