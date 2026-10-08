"""Structural citation validation: does each cited chunk exist and match expectations?

Citation validation answers a set of questions that can be decided *without* understanding
meaning, and it is explicit that it decides nothing more:

* **valid** — the cited chunk is present in the evidence pack supplied for this answer
  (ADR-0003: a reference outside the pack is never treated as a real citation);
* **unknown** — the citation names a chunk that was not supplied, which is a fabrication
  to be counted, not resolved;
* **irrelevant** — when the caller supplies the evidence a claim was *expected* to use, a
  citation that is present in the pack but not in that expected set. Relevance here is
  structural (set membership), decided against a benchmark judgement, not a semantic
  claim that the source actually supports the assertion;
* **uncited** — a claim that was required to carry evidence carries no valid citation.

A malformed citation (an identifier-shaped string that is not a valid id) is detected by
`chakaso.evidence.resolve_citations` when scanning answer *text*; a claim's own citations
are typed `ChunkId`s and so cannot be malformed by construction — the two surfaces are
complementary, not duplicated.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from chakaso.claims.model import Claim, ClaimId
from chakaso.core.identifiers import ChunkId
from chakaso.evidence import EvidencePack

__all__ = [
    "CitationCheck",
    "CitationStatus",
    "ClaimCitationResult",
    "validate_citations",
]


class CitationStatus(StrEnum):
    """How a single citation stands against the supplied evidence pack."""

    VALID = "valid"
    UNKNOWN = "unknown"
    IRRELEVANT = "irrelevant"


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """The verdict on one (claim, chunk) citation."""

    claim_id: ClaimId
    chunk_id: ChunkId
    status: CitationStatus


@dataclass(frozen=True, slots=True)
class ClaimCitationResult:
    """All citation verdicts for one claim, plus whether it was required to cite."""

    claim_id: ClaimId
    requires_citation: bool
    checks: tuple[CitationCheck, ...]

    @property
    def valid_count(self) -> int:
        """Citations present in the pack and, where expected evidence was given, relevant."""
        return sum(1 for check in self.checks if check.status is CitationStatus.VALID)

    @property
    def unknown_count(self) -> int:
        """Citations to chunks that were not supplied."""
        return sum(1 for check in self.checks if check.status is CitationStatus.UNKNOWN)

    @property
    def irrelevant_count(self) -> int:
        """Citations present in the pack but outside the claim's expected evidence."""
        return sum(1 for check in self.checks if check.status is CitationStatus.IRRELEVANT)

    @property
    def has_valid_citation(self) -> bool:
        """Whether at least one citation for this claim is valid."""
        return self.valid_count > 0

    @property
    def uncited(self) -> bool:
        """A required-to-cite claim with no valid citation."""
        return self.requires_citation and not self.has_valid_citation


def validate_citations(
    claims: Iterable[Claim],
    pack: EvidencePack,
    *,
    expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
    require_citation_for: Collection[ClaimId] | None = None,
) -> tuple[ClaimCitationResult, ...]:
    """Classify every claim's citations against ``pack``.

    Args:
        claims: the claims whose citations are checked.
        pack: the evidence supplied for the answer; a citation outside it is unknown.
        expected_evidence: optional per-claim expected chunk set. When given for a claim,
            a present-but-unexpected citation is *irrelevant*; when absent, presence in the
            pack is the only structural test and every present citation is valid.
        require_citation_for: the claims that must carry a citation. ``None`` means none
            are required, so ``uncited`` stays false; it is the caller (a benchmark case)
            that decides which claims bear a citation requirement.
    """
    expected = expected_evidence or {}
    required = require_citation_for or frozenset()

    results: list[ClaimCitationResult] = []
    for claim in claims:
        claim_id = claim.claim_id
        claim_expected = expected.get(claim_id)
        checks = tuple(
            CitationCheck(
                claim_id=claim_id,
                chunk_id=chunk_id,
                status=_status(chunk_id, pack, claim_expected),
            )
            for chunk_id in claim.cited_evidence_ids
        )
        results.append(
            ClaimCitationResult(
                claim_id=claim_id,
                requires_citation=claim_id in required,
                checks=checks,
            )
        )
    return tuple(results)


def _status(
    chunk_id: ChunkId,
    pack: EvidencePack,
    claim_expected: Collection[ChunkId] | None,
) -> CitationStatus:
    if pack.chunk_for(chunk_id) is None:
        return CitationStatus.UNKNOWN
    if claim_expected is not None and chunk_id not in claim_expected:
        return CitationStatus.IRRELEVANT
    return CitationStatus.VALID
