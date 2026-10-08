"""Tests for citation validation and metrics.

Each structural verdict is produced deliberately: a valid citation is in the pack, an
unknown one is not, an irrelevant one is in the pack but outside the expected evidence,
and an uncited claim is one that had to cite but produced no valid citation. The metrics
must count them without conflating "cited the right place" with "proves the claim."
"""

from __future__ import annotations

from chakaso.citation import CitationStatus, citation_metrics, validate_citations
from chakaso.claims.model import Claim
from chakaso.core.identifiers import ChunkId
from chakaso.evidence import EvidencePack
from chakaso.retrieval import ingest_text

_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
PACK = EvidencePack.of((*_ALPHA.chunks, *_BETA.chunks), [_ALPHA.source, _BETA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id
CHUNK_B = _BETA.chunks[0].chunk_id
UNKNOWN = ChunkId("chk_" + "f" * 16)


def claim(index: int, cited: tuple[ChunkId, ...]) -> Claim:
    return Claim(answer_id="ans", position=index, text=f"claim {index}", cited_evidence_ids=cited)


def test_valid_unknown_and_irrelevant_are_distinguished() -> None:
    valid = claim(0, (CHUNK_A,))
    unknown = claim(1, (UNKNOWN,))
    irrelevant = claim(2, (CHUNK_B,))  # in the pack, but expected to cite CHUNK_A

    results = validate_citations(
        [valid, unknown, irrelevant],
        PACK,
        expected_evidence={irrelevant.claim_id: {CHUNK_A}},
    )

    by_id = {r.claim_id: r.checks[0].status for r in results}
    assert by_id[valid.claim_id] is CitationStatus.VALID
    assert by_id[unknown.claim_id] is CitationStatus.UNKNOWN
    assert by_id[irrelevant.claim_id] is CitationStatus.IRRELEVANT


def test_without_expected_evidence_presence_in_the_pack_is_enough() -> None:
    result = validate_citations([claim(0, (CHUNK_A,))], PACK)[0]

    assert result.checks[0].status is CitationStatus.VALID


def test_a_required_claim_with_no_citation_is_uncited() -> None:
    uncited = claim(0, ())
    (result,) = validate_citations([uncited], PACK, require_citation_for={uncited.claim_id})

    assert result.requires_citation is True
    assert result.uncited is True


def test_an_optional_uncited_claim_is_not_flagged() -> None:
    uncited = claim(0, ())
    (result,) = validate_citations([uncited], PACK)

    assert result.uncited is False


def test_metrics_count_the_structural_outcomes() -> None:
    valid = claim(0, (CHUNK_A,))
    unknown = claim(1, (UNKNOWN,))
    irrelevant = claim(2, (CHUNK_B,))
    uncited = claim(3, ())

    results = validate_citations(
        [valid, unknown, irrelevant, uncited],
        PACK,
        expected_evidence={irrelevant.claim_id: {CHUNK_A}},
        require_citation_for={uncited.claim_id, valid.claim_id},
    )
    metrics = citation_metrics(results)

    assert metrics.claims == 4
    assert metrics.citations == 3
    assert metrics.valid == 1
    assert metrics.unknown == 1
    assert metrics.irrelevant == 1
    assert metrics.required_claims == 2
    assert metrics.uncited_claims == 1  # only `uncited`; `valid` cited successfully
    assert metrics.citation_precision == 1 / 3
    assert metrics.citation_recall == 1 / 2


def test_empty_inputs_do_not_fabricate_ratios() -> None:
    metrics = citation_metrics(validate_citations([], PACK))

    assert metrics.citations == 0
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0
