"""The claim: the unit an answer is decomposed into, and the unit correction acts on.

A claim is not the whole answer. "Paris is the capital of France and has a population of
X" is two claims, and a system that can only accept or reject an entire message will
accept a partially-wrong one. Correction, citation validation and grounding are all
defined over claims, so the claim is a first-class, immutable record with a stable
identifier within its answer.

Status is an *evaluation* result, not a truth verdict. `SUPPORTED` means "the evidence
that was supplied supports this" — it does not mean the claim is true in the world, and
the distinction is the whole reason the project separates evidence from knowledge
(`docs/transparency.md`). The default is `NOT_EVALUATED`, so an unexamined claim is never
mistaken for a supported one.

A claim identifier is derived from the answer, the position and the text, mirroring how
chunk identifiers are content-derived (ADR-0007): the same claim in the same answer always
has the same id, so a citation or a correction can point at it across a run without the
answer owning a mutable registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from chakaso.claims.errors import InvalidClaimError
from chakaso.core.hashing import join_parts, short_sha256_hex
from chakaso.core.identifiers import ChunkId

__all__ = ["Claim", "ClaimId", "ClaimStatus", "derive_claim_id"]

CLAIM_ID_PREFIX: Final[str] = "clm_"
_CLAIM_ID_PATTERN: Final[re.Pattern[str]] = re.compile(rf"{CLAIM_ID_PREFIX}[0-9a-f]{{16}}")


class ClaimStatus(StrEnum):
    """How the supplied evidence relates to a claim — never whether the claim is true.

    `NOT_EVALUATED` is the default so that an unexamined claim cannot be read as a
    checked one. `UNCERTAIN` is for evidence that bears on a claim without settling it;
    `CONTRADICTED` is for evidence that actively opposes it, which is the seed of a
    correction.
    """

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"
    NOT_EVALUATED = "not_evaluated"


@dataclass(frozen=True, slots=True, order=True)
class ClaimId:
    """A stable identifier for one claim within one answer."""

    value: str

    def __post_init__(self) -> None:
        if _CLAIM_ID_PATTERN.fullmatch(self.value) is None:
            message = (
                f"{self.value!r} is not a valid claim identifier; expected "
                f"{CLAIM_ID_PREFIX!r} followed by 16 lowercase hexadecimal characters"
            )
            raise InvalidClaimError(message)

    def __str__(self) -> str:
        return self.value


def derive_claim_id(answer_id: str, position: int, text: str) -> ClaimId:
    """Derive a claim's identifier from its answer, position and text."""
    digest = short_sha256_hex(join_parts(answer_id, str(position), text), length=16)
    return ClaimId(f"{CLAIM_ID_PREFIX}{digest}")


@dataclass(frozen=True, slots=True)
class Claim:
    """One atomic assertion extracted from an answer, with the evidence it cites."""

    answer_id: str
    position: int
    text: str
    cited_evidence_ids: tuple[ChunkId, ...] = ()
    status: ClaimStatus = ClaimStatus.NOT_EVALUATED

    def __post_init__(self) -> None:
        if not self.answer_id.strip():
            message = "a claim must name the answer it came from; answer_id was empty"
            raise InvalidClaimError(message)
        if self.position < 0:
            message = f"claim position must not be negative, got {self.position}"
            raise InvalidClaimError(message)
        text = self.text.strip()
        if not text:
            message = f"a claim must have text (answer {self.answer_id}, position {self.position})"
            raise InvalidClaimError(message)
        object.__setattr__(self, "text", text)
        object.__setattr__(
            self, "cited_evidence_ids", tuple(dict.fromkeys(self.cited_evidence_ids))
        )

    @property
    def claim_id(self) -> ClaimId:
        """The claim's stable, content-derived identifier within its answer."""
        return derive_claim_id(self.answer_id, self.position, self.text)

    @property
    def is_cited(self) -> bool:
        """Whether the claim references at least one piece of evidence."""
        return bool(self.cited_evidence_ids)

    def with_status(self, status: ClaimStatus) -> Claim:
        """Return a copy carrying ``status``; the claim's identity is unchanged.

        Evaluation sets a status without touching the text or citations, so the derived
        claim id — what a citation or correction points at — stays stable.
        """
        return Claim(
            answer_id=self.answer_id,
            position=self.position,
            text=self.text,
            cited_evidence_ids=self.cited_evidence_ids,
            status=status,
        )
