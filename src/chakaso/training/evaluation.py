"""Held-out evaluation of a trained model: the measurements the record will report.

This is where a loss number becomes a *result*. It reports the model's cross-entropy on data it never
trained on, in the same nats the baselines use, and the two derived quantities a reader needs to
interpret it — perplexity and bits-per-character — plus next-character accuracy. The units are chosen so
the model and the baselines are directly comparable: everything is mean cross-entropy over the same
held-out tokens.

It evaluates exactly what it is handed. It is up to the caller to pass the *validation* arrays; nothing
here would quietly score the training set and call it held-out performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from chakaso.training.model import CharCausalMLP, Params

__all__ = ["HeldOutMetrics", "evaluate", "per_position_cross_entropies"]

I64 = npt.NDArray[np.int64]
_LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class HeldOutMetrics:
    """The measured behavior of a model on one held-out set.

    Attributes:
        loss: Mean cross-entropy in nats — directly comparable to the uniform/unigram baselines.
        perplexity: ``exp(loss)``, the effective branching factor of the predictions.
        accuracy: Fraction of positions whose most-likely character equals the true next character.
        bits_per_char: The loss expressed in bits (``loss / ln 2``), the conventional LM scale.
        token_count: How many held-out positions were scored.
    """

    loss: float
    perplexity: float
    accuracy: float
    bits_per_char: float
    token_count: int


def evaluate(params: Params, model: CharCausalMLP, contexts: I64, targets: I64) -> HeldOutMetrics:
    """Score ``params`` on the given held-out ``contexts``/``targets`` arrays.

    Raises:
        ValueError: the batch is empty, so accuracy and loss would be computed over nothing. (The model
            validates the identifiers and shapes on the forward pass.)
    """
    if contexts.shape[0] == 0:
        message = "cannot evaluate an empty held-out set"
        raise ValueError(message)

    loss = model.forward(params, contexts, targets)
    probabilities = model.probabilities(params, contexts)
    predicted = probabilities.argmax(axis=1)
    accuracy = float(np.mean(predicted == targets))

    return HeldOutMetrics(
        loss=loss,
        perplexity=math.exp(loss),
        accuracy=accuracy,
        bits_per_char=loss / _LN2,
        token_count=int(contexts.shape[0]),
    )


def per_position_cross_entropies(
    params: Params, model: CharCausalMLP, contexts: I64, targets: I64
) -> tuple[float, ...]:
    """``-ln p(target)`` for each row, so the character model is scored by the shared metric arithmetic.

    :func:`evaluate` reports aggregated numbers; the comparison in E2605 needs the per-position series
    because bits per character is a *total* description length divided by a *total* character count. One
    arithmetic for both tokenizations is the honesty requirement, and returning the same quantity torch
    returns is how that is met without a second implementation.
    """
    probabilities = model.probabilities(params, contexts)
    return tuple(
        float(-math.log(probabilities[index, int(targets[index])])) for index in range(len(targets))
    )
