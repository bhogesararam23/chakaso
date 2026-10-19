"""Baselines that make the trained model's held-out loss mean something.

A loss curve with no baseline is not evidence of anything (roadmap, first model-training sprint;
ADR-0030). Two trivial next-token predictors give the comparison a lower bound the trained model must
beat to claim it learned more than it memorized the token frequency:

* the **uniform** predictor assigns every vocabulary identifier equal probability, so its cross-entropy
  is exactly ``ln(V)`` — the loss of knowing nothing but the alphabet;
* the **unigram** predictor assigns each identifier its frequency in the *training* targets and nothing
  else, so its loss is the empirical entropy of the training distribution evaluated on held-out data —
  the loss of knowing only how often characters occur, ignoring their order.

Both are computed the same way the model's held-out loss is — mean cross-entropy over the *validation*
tokens — so the comparison is mathematically valid: the same tokens, the same identifier space, the
same units (nats). This module is pure standard library; it needs no numerical dependency to state what
a model must beat.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

__all__ = ["UnigramBaseline", "uniform_next_token_loss"]


def uniform_next_token_loss(vocab_size: int) -> float:
    """The cross-entropy of a uniform distribution over ``vocab_size`` identifiers, in nats.

    This is a constant — it depends only on the size of the identifier space, not on the data — and it
    is the floor a model must fall below to be predicting better than a coin flip over the alphabet.
    """
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 1:
        message = f"vocab_size must be an integer >= 1, got {vocab_size!r}"
        raise ValueError(message)
    return math.log(vocab_size)


class UnigramBaseline:
    """A next-token predictor that ignores context and uses only training token frequencies.

    Probabilities are smoothed with ``alpha`` (Laplace by default) so an identifier unseen in training
    still gets a small, finite probability — otherwise a held-out token the baseline never saw would
    contribute infinite loss, which is an artefact of the estimator rather than a meaningful result. The
    smoothing is stated rather than hidden.
    """

    __slots__ = ("_log_probabilities", "alpha", "vocab_size")

    def __init__(self, counts: Sequence[int], vocab_size: int, *, alpha: float) -> None:
        if vocab_size < 1:
            message = f"vocab_size must be >= 1, got {vocab_size!r}"
            raise ValueError(message)
        if alpha <= 0:
            message = f"smoothing alpha must be positive, got {alpha!r}"
            raise ValueError(message)

        observed = Counter(counts)
        total = sum(observed.values()) + alpha * vocab_size
        self._log_probabilities: list[float] = [
            math.log((observed.get(identifier, 0) + alpha) / total)
            for identifier in range(vocab_size)
        ]
        self.vocab_size = vocab_size
        self.alpha = alpha

    @classmethod
    def fit(
        cls, train_targets: Sequence[int], vocab_size: int, *, alpha: float = 1.0
    ) -> UnigramBaseline:
        """Estimate the unigram distribution from training target identifiers."""
        if vocab_size < 1:
            message = f"vocab_size must be >= 1, got {vocab_size!r}"
            raise ValueError(message)
        for identifier in train_targets:
            _check_in_range(identifier, vocab_size)
        return cls(list(train_targets), vocab_size, alpha=alpha)

    def cross_entropy(self, targets: Sequence[int]) -> float:
        """Mean cross-entropy, in nats, of this baseline evaluated on ``targets``.

        Raises:
            ValueError: ``targets`` is empty, so there is nothing to average over and the result would
                be undefined.
        """
        if not targets:
            message = "cannot compute a cross-entropy over zero targets"
            raise ValueError(message)
        total = 0.0
        for identifier in targets:
            _check_in_range(identifier, self.vocab_size)
            total += self._log_probabilities[identifier]
        return -total / len(targets)


def _check_in_range(identifier: int, vocab_size: int) -> None:
    if identifier < 0 or identifier >= vocab_size:
        message = f"identifier {identifier!r} is outside the vocabulary range [0, {vocab_size})"
        raise ValueError(message)
