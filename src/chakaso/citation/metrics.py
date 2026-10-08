"""Citation metrics — structural counts, never a semantic support score.

These aggregate the output of `validate_citations`. They are precise about what they
measure: a citation is "valid" because it points at supplied evidence (and, where the
benchmark expected specific evidence, matches it) — *not* because a human judged that the
source supports the claim. That stronger judgement needs a semantic evaluator that does not
exist; conflating "cited the right place" with "the citation proves the claim" is the
exact error the project's transparency rules forbid.

The definitions are stated in the field names and are stable:

* citation precision = valid citations / all citations made — of the references issued, how
  many point at acceptable supplied evidence;
* citation recall = required claims with at least one valid citation / required claims — of
  the claims that had to be evidenced, how many were;
* unsupported / unknown / irrelevant counts are reported alongside, so a good precision
  cannot hide a pile of uncited claims.
"""

from __future__ import annotations

from collections.abc import Iterable

from chakaso.citation.validate import CitationStatus, ClaimCitationResult

__all__ = ["CitationMetrics", "citation_metrics"]


class CitationMetrics:
    """Aggregate citation counts and structural precision/recall over a set of results."""

    def __init__(
        self,
        *,
        claims: int,
        citations: int,
        valid: int,
        unknown: int,
        irrelevant: int,
        required_claims: int,
        uncited_claims: int,
    ) -> None:
        self.claims = claims
        self.citations = citations
        self.valid = valid
        self.unknown = unknown
        self.irrelevant = irrelevant
        self.required_claims = required_claims
        self.uncited_claims = uncited_claims

    @property
    def citation_precision(self) -> float:
        """Valid citations over all citations made; 1.0 when nothing was cited."""
        return self.valid / self.citations if self.citations else 1.0

    @property
    def citation_recall(self) -> float:
        """Required claims that carry a valid citation; 1.0 when none were required."""
        if not self.required_claims:
            return 1.0
        return (self.required_claims - self.uncited_claims) / self.required_claims


def citation_metrics(results: Iterable[ClaimCitationResult]) -> CitationMetrics:
    """Aggregate per-claim citation results into structural metrics."""
    collected = tuple(results)
    checks = [check for result in collected for check in result.checks]
    return CitationMetrics(
        claims=len(collected),
        citations=len(checks),
        valid=sum(1 for c in checks if c.status is CitationStatus.VALID),
        unknown=sum(1 for c in checks if c.status is CitationStatus.UNKNOWN),
        irrelevant=sum(1 for c in checks if c.status is CitationStatus.IRRELEVANT),
        required_claims=sum(1 for r in collected if r.requires_citation),
        uncited_claims=sum(1 for r in collected if r.uncited),
    )
