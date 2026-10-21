"""Decoding policy: what one generated token is, and when a generation is over.

Kept apart from the model adapters because these are the two decisions an inference runtime gets wrong in
ways a benchmark never shows. A nucleus that keeps the whole distribution is not top-p; a stop sequence
searched across the whole context matches the prompt and ends the run at token one; a repetition penalty
applied to a set rather than a count silences a model that legitimately uses a common word twice. Each rule
here is stated over plain sequences of numbers, so it can be tested without a framework, without weights and
without the four-second cold import that makes a torch-dependent policy invisible in the default matrix.

The interface is deliberately narrow: :class:`TokenSelector` turns one row of logits into one identifier, and
:class:`StopScanner` turns one decoded step into either "continue" or "stop, and here is what stopped it".
The adapter owns the loop, the cache and the tensor arithmetic; this module owns the policy, and neither can
silently change the other's meaning.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from chakaso.models.base import GenerationParams
from chakaso.models.errors import ModelError

__all__ = [
    "AbortSignal",
    "StopScanner",
    "StopVerdict",
    "TokenSelector",
    "truncate_prompt",
]


class AbortSignal:
    """A caller's way to stop a stream that is already running.

    Deliberately a mutable flag rather than a cancellation token with a scheduler behind it: the loop checks
    it between steps, which is the only granularity a synchronous CPU runtime can honestly offer. A caller
    that needs a hard deadline should set one on the transport, not assume this interrupts arithmetic.
    """

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        """Whether the caller has asked for the run to stop."""
        return self._cancelled

    def abort(self) -> None:
        """Ask the running generation to stop at its next step boundary."""
        self._cancelled = True


def truncate_prompt(prompt_ids: Sequence[int], context_length: int) -> tuple[tuple[int, ...], bool]:
    """Fit a prompt into the window by keeping its most recent identifiers.

    Returns:
        The identifiers to prefill with, and whether anything was dropped. Right-clipping is the only
        policy offered because the alternative — silently shortening the middle of a question — produces an
        answer that looks like a response to the original.

    Raises:
        ModelError: the window is not a positive integer, or the prompt is empty.
    """
    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        message = f"a context window must be an integer >= 1, got {context_length!r}"
        raise ModelError(message)
    if not prompt_ids:
        message = (
            "a generation needs at least one prompt identifier; an empty prompt has no context"
        )
        raise ModelError(message)
    if len(prompt_ids) <= context_length:
        return tuple(prompt_ids), False
    return tuple(prompt_ids[-context_length:]), True


@dataclass(frozen=True, slots=True)
class StopVerdict:
    """Why a generation ended: nothing, a token identifier, or a piece of text."""

    stopped: bool
    marker: str | None = None

    @classmethod
    def continue_generation(cls) -> StopVerdict:
        """The answer a scanner gives when the run should carry on."""
        return cls(False, None)


class StopScanner:
    """Watches a generation for the things that mean it is finished.

    Attributes:
        sequences: Text that ends the run when it appears in the *produced* text, never in the prompt.
        token_ids: Identifiers that end the run when emitted, in addition to the checkpoint's separator.
        document_end_id: The separator identifier the trained vocabulary reserves, when the caller has one.
    """

    def __init__(
        self,
        *,
        sequences: Sequence[str] = (),
        token_ids: Sequence[int] = (),
        document_end_id: int | None = None,
    ) -> None:
        longest = max((len(item) for item in sequences), default=0)
        if longest > 64:
            message = (
                f"a stop sequence longer than 64 characters ({longest}) is not a stop sequence"
            )
            raise ModelError(message)
        self._sequences = tuple(sequences)
        self._token_ids = frozenset(token_ids)
        self._document_end_id = document_end_id
        self._window = max(longest - 1, 0)
        self._tail = ""

    def scan_text(self, text: str) -> StopVerdict:
        """Check a candidate stop marker against the text produced so far.

        A caller with the whole generated string uses this; :meth:`feed` is the incremental form, which
        bounds its own work to the last few characters so a long generation does not re-scan its history
        every step.
        """
        for sequence in self._sequences:
            if sequence in text:
                return StopVerdict(True, sequence)
        return StopVerdict.continue_generation()

    def feed(self, produced: str, identifier: int) -> StopVerdict:
        """Record one produced step and report whether the run is over.

        The text test looks only at a window no longer than the longest sequence plus the newest chunk, so a
        marker spanning more text than that is not found — which is why the window is derived from the
        declared sequences rather than fixed, and why the whole-string form above exists for final checks.
        """
        if identifier in self._token_ids:
            return StopVerdict(True, f"token:{identifier}")
        if self._document_end_id is not None and identifier == self._document_end_id:
            return StopVerdict(True, "document-end")
        if not self._sequences:
            return StopVerdict.continue_generation()
        self._tail = (self._tail + produced)[-max(self._window, 0) - 256 :]
        for sequence in self._sequences:
            if sequence in self._tail:
                return StopVerdict(True, sequence)
        return StopVerdict.continue_generation()


class TokenSelector:
    """One row of logits in, one identifier out, under the parameters a caller declared.

    The order of operations is the standard one and is stated because it changes the answer: the repetition
    penalty is applied to the raw logits, the temperature rescales what remains, ``top_k`` truncates to the
    k best, ``top_p`` then keeps the smallest prefix whose cumulative mass reaches it, and the draw — or the
    argmax, when the temperature is zero — happens inside what survives.
    """

    def __init__(
        self,
        params: GenerationParams,
        *,
        vocabulary_size: int,
        rng: random.Random | None = None,
    ) -> None:
        if vocabulary_size < 1:
            message = f"a vocabulary must hold at least one identifier, got {vocabulary_size}"
            raise ModelError(message)
        if params.top_k is not None and params.top_k > vocabulary_size:
            message = (
                f"top_k ({params.top_k}) is wider than the vocabulary ({vocabulary_size}); the request "
                "asks to keep identifiers that do not exist"
            )
            raise ModelError(message)
        self._params = params
        self._vocabulary_size = vocabulary_size
        self._rng = rng if rng is not None else random.Random(params.seed)
        self._counts: dict[int, int] = {}

    @property
    def greedy(self) -> bool:
        """Whether this selector will always take the best remaining candidate."""
        return self._params.temperature == 0.0

    def note(self, identifier: int) -> None:
        """Record a produced identifier, so the repetition penalty sees real counts."""
        self._counts[identifier] = self._counts.get(identifier, 0) + 1

    def _penalized(self, logits: Sequence[float]) -> list[float]:
        penalty = float(self._params.repeat_penalty)
        if penalty == 1.0 or not self._counts:
            return list(logits)
        adjusted = list(logits)
        for identifier, count in self._counts.items():
            if identifier >= len(adjusted):
                continue
            value = adjusted[identifier]
            # A positive logit is divided and a negative one multiplied, which is the form that always
            # lowers the token's rank; applying the same operation to both signs would promote a negative
            # logit, so a penalty could make a used token *more* likely. Scaled by how many times the
            # token has appeared, because a token used once has not yet shown the loop a caller is
            # trying to break.
            adjusted[identifier] = (
                value / (penalty**count) if value > 0 else value * (penalty**count)
            )
        return adjusted

    def select(self, logits: Sequence[float]) -> int:
        """Choose the next identifier from one row of model logits.

        Raises:
            ModelError: the row is wider or narrower than the vocabulary, or carries a value that is not
                finite — a non-finite logit is a broken run, not a distribution, and guessing an argument
                would present a numerical failure as a word.
        """
        if len(logits) != self._vocabulary_size:
            message = (
                f"the model returned {len(logits)} logits for a vocabulary of "
                f"{self._vocabulary_size}"
            )
            raise ModelError(message)
        if not logits:
            message = "the model returned no logits at all"
            raise ModelError(message)
        if any(not math.isfinite(value) for value in logits):
            message = (
                "the model produced a non-finite logit, so no distribution can be drawn from it"
            )
            raise ModelError(message)

        scores = self._penalized(logits)
        if self._params.temperature == 0.0:
            # Ties go to the lowest identifier, so greedy output is a function of the logits alone.
            return max(range(len(scores)), key=lambda index: (scores[index], -index))

        scaled = [value / self._params.temperature for value in scores]
        order = sorted(range(len(scaled)), key=lambda index: scaled[index], reverse=True)
        if self._params.top_k is not None:
            order = order[: int(self._params.top_k)]
        maximum = scaled[order[0]]
        weights = [math.exp(scaled[index] - maximum) for index in order]
        total = math.fsum(weights)

        if self._params.top_p is not None:
            limit = float(self._params.top_p) * total
            cumulative = 0.0
            allowed = len(order)
            for position, weight in enumerate(weights):
                cumulative += weight
                if cumulative >= limit:
                    allowed = position + 1
                    break
            order = order[:allowed]
            weights = weights[:allowed]
            total = math.fsum(weights)

        draw = self._rng.random() * total
        cumulative = 0.0
        for index, weight in zip(order, weights, strict=True):
            cumulative += weight
            if draw < cumulative:
                return index
        return order[-1]
