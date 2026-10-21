"""Model metrics with one arithmetic, so two tokenizations can be compared at all.

Cross-entropy per token is not comparable across tokenizations: a 1,024-entry subword vocabulary and a
90-character vocabulary assign different amounts of text to each predicted identifier, so "nats per token"
measures different things. **Bits per character** is the tokenizer-independent quantity — the total code
length of the model's description of a text, divided by the characters of that text — and it is what
Chakaso reports when comparing models that segment differently.

The definition, stated because per-position attribution of a character is misleading:

```text
nats_total = sum over scored positions of -ln p(target at that position)
chars      = number of characters in decode(concatenation of all scored targets)
bpc        = (nats_total / ln 2) / chars
```

The denominator is the *decoded whole sequence*, not a per-token character count. A byte token that
continues a multi-byte character contributes no character at its own position and the completion contributes
one, so only the total is meaningful; summing over the whole scored stream is exactly how much text those
positions describe. The document separator decodes to nothing and so adds no characters while still
costing its prediction — a model that cannot place boundaries is charged for it.

Loss per token and perplexity are reported alongside and are *not* merged into bpc, and nothing here
combines the two into a single quality number ([`docs/evaluation.md`](../../../docs/evaluation.md)).

Standard library only: the arithmetic is the part that must be checkable by hand, and it is written
against an identifier-decoding shape rather than one tokenizer class, so the character model and the
subword model are scored by the same function.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from chakaso.training.errors import TrainingError

__all__ = ["IdentifierDecoder", "ModelMetrics", "bits_per_character", "metrics_from_scores"]

_LN2 = math.log(2.0)


class IdentifierDecoder(Protocol):
    """Whatever can turn scored identifiers back into text, plus the identifier that carries none.

    Both tokenizations satisfy this: the subword vocabulary decodes merge symbols and treats the document
    separator as no text, and the character vocabulary decodes known characters and treats its reserved
    unknown identifier as no text. Naming the shape rather than a class is what lets one metric function
    score two different tokenizations, which is the whole point of reporting bits per character.
    """

    @property
    def document_end_id(self) -> int: ...

    def decode(self, identifiers: Sequence[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """The measured behaviour of a model on one held-out set, in every unit a reader needs.

    Attributes:
        loss_nats: Mean cross-entropy per scored position.
        perplexity: ``exp(loss_nats)``, the effective branching factor. Meaningful only in the model's own
            identifier space, which is why it is never used to compare two tokenizations.
        bits_per_character: The comparable cross-tokenization number.
        positions_scored: How many predictions were made.
        characters_scored: How many characters those predictions describe.
        accuracy: Fraction of scored positions whose most-likely identifier was the target.
        separator_targets: How many scored positions were document separators.
    """

    loss_nats: float
    perplexity: float
    bits_per_character: float
    positions_scored: int
    characters_scored: int
    accuracy: float
    separator_targets: int

    def as_dict(self) -> dict[str, float | int]:
        """A machine-readable view for an experiment record."""
        return {
            "loss_nats_per_token": self.loss_nats,
            "perplexity": self.perplexity,
            "bits_per_character": self.bits_per_character,
            "positions_scored": self.positions_scored,
            "characters_scored": self.characters_scored,
            "token_accuracy": self.accuracy,
            "separator_targets": self.separator_targets,
        }


def bits_per_character(nats_total: float, characters_scored: int) -> float:
    """Convert a total description length in nats into bits per character.

    Raises:
        TrainingError: there are no characters to divide by, which would otherwise report a loss over an
            empty text as a number.
    """
    if characters_scored < 1:
        message = "cannot compute bits per character over zero characters"
        raise TrainingError(message)
    return (nats_total / _LN2) / characters_scored


def metrics_from_scores(
    nats: Sequence[float],
    targets: Sequence[int],
    predictions: Sequence[int],
    decoder: IdentifierDecoder,
) -> ModelMetrics:
    """Combine per-position cross-entropies, targets and argmaxes into the reported metrics.

    This is the whole measurement arithmetic in one place, so a model trained in a different framework or
    segmenting text differently is scored by the same function.

    Raises:
        TrainingError: the three sequences disagree in length, or a sequence is empty. A loss computed over
            nothing is not zero; it is undefined, and refusing is the honest answer.
    """
    if not nats or not targets or not predictions:
        message = (
            "metrics require at least one scored position; a measurement over nothing is undefined"
        )
        raise TrainingError(message)
    if not (len(nats) == len(targets) == len(predictions)):
        message = (
            f"nats, targets and predictions must have equal length, got {len(nats)}, "
            f"{len(targets)}, {len(predictions)}"
        )
        raise TrainingError(message)
    if any(value < 0 for value in nats):
        message = "a per-position cross-entropy cannot be negative"
        raise TrainingError(message)

    nats_total = math.fsum(nats)
    positions = len(targets)
    decoded = decoder.decode(targets)
    characters = len(decoded)
    correct = sum(
        1 for predicted, target in zip(predictions, targets, strict=True) if predicted == target
    )
    separators = sum(1 for target in targets if target == decoder.document_end_id)

    return ModelMetrics(
        loss_nats=nats_total / positions,
        perplexity=math.exp(nats_total / positions),
        bits_per_character=bits_per_character(nats_total, characters),
        positions_scored=positions,
        characters_scored=characters,
        accuracy=correct / positions,
        separator_targets=separators,
    )
