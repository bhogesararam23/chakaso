"""The claim-to-evidence link: an inspectable relationship, not a re-derived identity.

A claim already carries the chunk identifiers it cites, so a "link" is a projection of
that into an explicit (claim, chunk) pair — the shape citation validation and grounding
iterate over. It deliberately holds only identifiers: the source of a chunk is resolved
through the evidence pack (`EvidencePack.chunk_for`/`source_for`), never copied here, so a
link cannot drift from the record it points at (ADR-0003).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from chakaso.claims.model import Claim, ClaimId
from chakaso.core.identifiers import ChunkId

__all__ = ["ClaimEvidenceLink", "links_from_claims"]


@dataclass(frozen=True, slots=True, order=True)
class ClaimEvidenceLink:
    """One claim's citation of one evidence chunk, by identifier only."""

    claim_id: ClaimId
    chunk_id: ChunkId


def links_from_claims(claims: Iterable[Claim]) -> tuple[ClaimEvidenceLink, ...]:
    """Project claims into their (claim, chunk) citation links, in claim order."""
    return tuple(
        ClaimEvidenceLink(claim_id=claim.claim_id, chunk_id=chunk_id)
        for claim in claims
        for chunk_id in claim.cited_evidence_ids
    )
