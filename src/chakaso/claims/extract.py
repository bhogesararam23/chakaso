"""Claim extraction: a replaceable boundary, not a semantic truth engine.

There is no trained model to decompose an answer, so extraction is a *boundary* with two
honest initial implementations and no pretence about either:

* `StructuredClaimExtractor` takes claims that were supplied explicitly (by a test fixture
  or, later, by a model that emits structured output) and turns them into `Claim` records.
* `SentenceClaimExtractor` is a deterministic *heuristic*: it splits on sentence-ending
  punctuation and emits one unevaluated claim per sentence. It is labelled a heuristic
  because that is what it is — an abbreviation splits a sentence, and it attaches no
  evidence. It is a stand-in for a real decomposer, never a claim to understand the text.

Both implement the `ClaimExtractor` protocol, so a trained Chakaso model can implement the
same boundary later and every caller — citation validation, grounding, correction — keeps
working unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final, Protocol, runtime_checkable

from chakaso.claims.model import Claim, ClaimStatus
from chakaso.core.identifiers import ChunkId

__all__ = ["ClaimExtractor", "SentenceClaimExtractor", "StructuredClaimExtractor"]

#: A deliberately crude sentence boundary. It exists so a development double's fixed
#: prose can be decomposed at all; it is not a claim about sentence structure.
_SENTENCE_BREAK: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?])\s+")


@runtime_checkable
class ClaimExtractor(Protocol):
    """The boundary every claim extractor satisfies: an answer in, ordered claims out."""

    def extract(self, answer_id: str, text: str) -> tuple[Claim, ...]:
        """Decompose ``text`` into claims belonging to ``answer_id``, in document order."""
        ...


class SentenceClaimExtractor:
    """A documented heuristic: one unevaluated claim per sentence, no evidence attached."""

    def extract(self, answer_id: str, text: str) -> tuple[Claim, ...]:
        claims: list[Claim] = []
        for raw in _SENTENCE_BREAK.split(text.strip()):
            sentence = raw.strip()
            if not sentence:
                continue
            claims.append(
                Claim(
                    answer_id=answer_id,
                    position=len(claims),
                    text=sentence,
                    status=ClaimStatus.NOT_EVALUATED,
                )
            )
        return tuple(claims)


class StructuredClaimExtractor:
    """Builds claims from explicitly supplied text and citations, position-ordered.

    The caller (a fixture, or a model with structured output) decides the claim boundary;
    this class only constructs valid, stable `Claim` records from that decision.
    """

    def __init__(self, specs: Iterable[tuple[str, Iterable[ChunkId]]]) -> None:
        # (claim text, cited chunk ids) in answer order.
        self._specs = tuple((text, tuple(chunks)) for text, chunks in specs)

    def extract(self, answer_id: str, text: str) -> tuple[Claim, ...]:  # noqa: ARG002
        # `text` is part of the shared extractor signature; a structured extractor does
        # not need it, because the caller already supplied the decomposition.
        return tuple(
            Claim(
                answer_id=answer_id,
                position=position,
                text=claim_text,
                cited_evidence_ids=claim_chunks,
            )
            for position, (claim_text, claim_chunks) in enumerate(self._specs)
        )
