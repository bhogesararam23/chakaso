"""Grounding model: which claims the supplied evidence supports, contradicts or leaves open.

A grounding result maps each claim to a status, and the status means what the *supplied
evidence* establishes — never that a claim is true in the world. The one implemented
automatic evaluator is structural: it can distinguish supported (a valid citation exists)
from unsupported (a claim needed evidence and had none) and leaves the rest unevaluated.
Contradiction and uncertainty can only be *asserted by a judgement* a caller supplies (the
manual evaluator, standing in for a human or a future model), never inferred from string
matching here.

`SourceAuthority` and `most_recent_wins` are hooks, not policy: they let a benchmark case
that needs a precedence rule state one explicitly, and nothing in the engine applies one
silently. That keeps the retrieval/grounding engine from smuggling in an "newer is truer"
or "authoritative domain wins" assumption it has not earned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from chakaso.claims.model import Claim, ClaimId, ClaimStatus
from chakaso.core.identifiers import ChunkId, SourceId

__all__ = [
    "Contradiction",
    "GroundingResult",
    "SourceAuthority",
    "most_recent_wins",
]


@dataclass(frozen=True, slots=True)
class Contradiction:
    """Two or more supplied chunks that bear on a claim and disagree.

    It records the disagreement without naming a winner; deciding which side prevails
    needs an explicit precedence (see `most_recent_wins`) that a case supplies, or a
    human/model judgement — not an implicit ranking buried in the engine.
    """

    claim_id: ClaimId
    chunk_ids: tuple[ChunkId, ...]
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_ids", tuple(dict.fromkeys(self.chunk_ids)))
        if len(self.chunk_ids) < 2:
            message = (
                f"a contradiction needs at least two disagreeing chunks, claim {self.claim_id}"
            )
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SourceAuthority:
    """Optional precedence metadata about a source, supplied by a case — never ranked by default.

    `authority` is a caller-chosen label (a domain tier, a publisher name); it carries no
    engine-defined ranking. It exists so a case that must resolve a conflict can express a
    rule, and so that rule is visible in the data rather than hidden in a scorer.
    """

    source_id: SourceId
    authority: str | None = None
    published_at: datetime | None = None
    retrieved_at: datetime | None = None


def most_recent_wins(candidates: Sequence[SourceAuthority]) -> SourceAuthority | None:
    """A *sample* precedence rule a case may opt into: the newest published source wins.

    This is not applied anywhere automatically. Ties on publication time, or candidates
    with no publication date, yield ``None`` (no winner declared) rather than an arbitrary
    pick — a precedence rule that guesses is indistinguishable from one that is wrong.
    """
    dated = [
        candidate.published_at for candidate in candidates if candidate.published_at is not None
    ]
    if not dated:
        return None
    newest = max(dated)
    winners = [candidate for candidate in candidates if candidate.published_at == newest]
    return winners[0] if len(winners) == 1 else None


@dataclass(frozen=True, slots=True)
class GroundingResult:
    """The outcome of grounding one answer: a status per claim, plus any conflicts."""

    answer_id: str
    statuses: Mapping[ClaimId, ClaimStatus] = field(default_factory=lambda: MappingProxyType({}))
    contradictions: tuple[Contradiction, ...] = ()
    evaluator_name: str = "structural"

    def status_of(self, claim: Claim | ClaimId) -> ClaimStatus:
        """The status assigned to a claim, or ``NOT_EVALUATED`` if it was not assessed."""
        claim_id = claim.claim_id if isinstance(claim, Claim) else claim
        return self.statuses.get(claim_id, ClaimStatus.NOT_EVALUATED)

    def claims_with(self, status: ClaimStatus) -> tuple[ClaimId, ...]:
        """Claim identifiers assigned ``status``, ordered by their string form."""
        return tuple(
            sorted((cid for cid, value in self.statuses.items() if value is status), key=str)
        )

    @property
    def supported(self) -> tuple[ClaimId, ...]:
        """Claims the supplied evidence supports."""
        return self.claims_with(ClaimStatus.SUPPORTED)

    @property
    def unsupported(self) -> tuple[ClaimId, ...]:
        """Claims that needed evidence and had none that was valid."""
        return self.claims_with(ClaimStatus.UNSUPPORTED)

    @property
    def contradicted(self) -> tuple[ClaimId, ...]:
        """Claims the supplied evidence contradicts (asserted by a judgement, not inferred)."""
        return self.claims_with(ClaimStatus.CONTRADICTED)

    @property
    def not_evaluated(self) -> tuple[ClaimId, ...]:
        """Claims no status was decided for."""
        return self.claims_with(ClaimStatus.NOT_EVALUATED)

    @property
    def is_fully_supported(self) -> bool:
        """Whether every evaluated claim is supported and none is otherwise."""
        return bool(self.statuses) and all(
            value is ClaimStatus.SUPPORTED for value in self.statuses.values()
        )
