"""Tests for claim extraction and claim/evidence links.

The extractor is a boundary with two honest implementations, so the tests pin what each
claims to be: the sentence extractor is a deterministic heuristic that attaches no
evidence and defaults to not-evaluated; the structured extractor preserves the caller's
decomposition and citations; both satisfy the same protocol. Links carry identifiers only.
"""

from __future__ import annotations

from chakaso.claims import (
    ClaimExtractor,
    ClaimStatus,
    SentenceClaimExtractor,
    StructuredClaimExtractor,
    links_from_claims,
)
from chakaso.core.identifiers import ChunkId


def chk(n: int) -> ChunkId:
    return ChunkId(f"chk_{n:016x}")


def test_sentence_extractor_splits_into_ordered_unevaluated_claims() -> None:
    claims = SentenceClaimExtractor().extract("ans", "First sentence. Second one. Third?")

    assert [c.position for c in claims] == [0, 1, 2]
    assert [c.text for c in claims] == ["First sentence.", "Second one.", "Third?"]
    assert all(c.status is ClaimStatus.NOT_EVALUATED for c in claims)
    assert all(not c.is_cited for c in claims)


def test_sentence_extractor_is_deterministic_and_empty_for_blank_input() -> None:
    extractor = SentenceClaimExtractor()
    assert extractor.extract("a", "Same text.") == extractor.extract("a", "Same text.")
    assert extractor.extract("a", "   ") == ()


def test_structured_extractor_preserves_citations_and_order() -> None:
    extractor = StructuredClaimExtractor(
        [("Paris is the capital.", [chk(1)]), ("Its population is X.", [chk(2)])]
    )

    claims = extractor.extract("ans", "ignored")

    assert [c.text for c in claims] == ["Paris is the capital.", "Its population is X."]
    assert claims[0].cited_evidence_ids == (chk(1),)
    assert claims[1].cited_evidence_ids == (chk(2),)


def test_both_extractors_satisfy_the_protocol() -> None:
    assert isinstance(SentenceClaimExtractor(), ClaimExtractor)
    assert isinstance(StructuredClaimExtractor([("c", [])]), ClaimExtractor)


def test_links_project_claims_to_chunks_in_order() -> None:
    claims = StructuredClaimExtractor([("a", [chk(1), chk(2)]), ("b", [chk(3)])]).extract("ans", "")
    links = links_from_claims(claims)

    assert [(str(link.claim_id), link.chunk_id) for link in links] == [
        (str(claims[0].claim_id), chk(1)),
        (str(claims[0].claim_id), chk(2)),
        (str(claims[1].claim_id), chk(3)),
    ]


def test_uncited_claims_produce_no_links() -> None:
    claims = SentenceClaimExtractor().extract("ans", "Just a sentence.")

    assert links_from_claims(claims) == ()
