"""Tests for the answer-evaluation combiner.

The point of this layer is that an answer's trustworthiness stays *several* numbers, not
one. These tests pin the composite properties: a well-cited answer grounds, a required
uncited claim blocks grounding while citation precision stays high (the dimensions must not
collapse into each other), an unknown reference disqualifies grounding even when the claim
is otherwise supported, and a supplied judgement — not the structural pass — is what can
introduce a contradiction. No merged "quality score" is asserted anywhere, because none
exists.
"""

from __future__ import annotations

from chakaso.claims.model import Claim, ClaimStatus
from chakaso.core.identifiers import ChunkId
from chakaso.evaluation import AnswerEvaluation, evaluate_answer
from chakaso.evidence import EvidencePack
from chakaso.grounding import Contradiction, ManualGroundingEvaluator
from chakaso.retrieval import ingest_text

_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
PACK = EvidencePack.of((*_ALPHA.chunks, *_BETA.chunks), [_ALPHA.source, _BETA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id
CHUNK_B = _BETA.chunks[0].chunk_id
UNKNOWN = ChunkId("chk_" + "f" * 16)


def claim(index: int, cited: tuple[ChunkId, ...] = ()) -> Claim:
    return Claim(answer_id="ans", position=index, text=f"claim {index}", cited_evidence_ids=cited)


def test_a_well_cited_answer_grounds_on_the_structural_default() -> None:
    first = claim(0, (CHUNK_A,))
    second = claim(1, (CHUNK_B,))

    evaluation = evaluate_answer(
        "ans",
        [first, second],
        PACK,
        requires_citation={first.claim_id, second.claim_id},
    )

    assert isinstance(evaluation, AnswerEvaluation)
    assert evaluation.claims == 2
    assert evaluation.citations.valid == 2
    assert evaluation.citations.unknown == 0
    assert set(evaluation.grounding.supported) == {first.claim_id, second.claim_id}
    assert evaluation.evidence_coverage == 1.0
    assert evaluation.is_grounded is True
    assert evaluation.grounding.evaluator_name == "structural-0.1"


def test_a_required_uncited_claim_is_unsupported_and_blocks_grounding() -> None:
    cited = claim(0, (CHUNK_A,))
    uncited = claim(1)

    evaluation = evaluate_answer(
        "ans",
        [cited, uncited],
        PACK,
        requires_citation={cited.claim_id, uncited.claim_id},
    )

    assert evaluation.unsupported_claims == (uncited.claim_id,)
    assert evaluation.citations.uncited_claims == 1
    assert evaluation.evidence_coverage == 0.5
    assert evaluation.is_grounded is False


def test_citation_precision_and_evidence_coverage_stay_separate() -> None:
    # Every citation made was valid (precision 1.0) yet only half the claims are supported
    # (coverage 0.5). A single merged score would hide one fact behind the other; keeping
    # them as distinct dimensions is the reason this layer exists.
    good = claim(0, (CHUNK_A,))
    missing = claim(1)

    evaluation = evaluate_answer(
        "ans",
        [good, missing],
        PACK,
        requires_citation={good.claim_id, missing.claim_id},
    )

    assert evaluation.citations.citation_precision == 1.0
    assert evaluation.evidence_coverage == 0.5


def test_an_unknown_citation_blocks_grounding_even_when_the_claim_is_supported() -> None:
    mixed = claim(0, (CHUNK_A, UNKNOWN))

    evaluation = evaluate_answer("ans", [mixed], PACK)

    assert evaluation.grounding.is_fully_supported is True
    assert evaluation.citations.unknown == 1
    assert evaluation.evidence_coverage == 1.0
    assert evaluation.is_grounded is False


def test_expected_evidence_demotes_a_present_but_wrong_source_citation() -> None:
    wrong = claim(0, (CHUNK_B,))

    evaluation = evaluate_answer(
        "ans",
        [wrong],
        PACK,
        requires_citation={wrong.claim_id},
        expected_evidence={wrong.claim_id: {CHUNK_A}},
    )

    assert evaluation.citations.irrelevant == 1
    assert evaluation.citations.valid == 0
    assert evaluation.unsupported_claims == (wrong.claim_id,)
    assert evaluation.is_grounded is False


def test_a_supplied_judgement_can_introduce_a_contradiction() -> None:
    conflicted = claim(0, (CHUNK_A,))
    contradiction = Contradiction(
        conflicted.claim_id,
        (CHUNK_A, CHUNK_B),
        detail="dates differ",
    )
    evaluator = ManualGroundingEvaluator(
        {conflicted.claim_id: ClaimStatus.CONTRADICTED},
        [contradiction],
    )

    evaluation = evaluate_answer("ans", [conflicted], PACK, grounding_evaluator=evaluator)

    assert evaluation.contradicted_claims == (conflicted.claim_id,)
    assert evaluation.evidence_coverage == 0.0
    assert evaluation.is_grounded is False
    # the citation itself was still valid — grounding and citation stay distinct dimensions
    assert evaluation.citations.valid == 1


def test_an_answer_with_no_claims_does_not_fabricate_coverage() -> None:
    evaluation = evaluate_answer("ans", [], PACK)

    assert evaluation.claims == 0
    assert evaluation.evidence_coverage == 0.0
    assert evaluation.is_grounded is False
    assert evaluation.citations.citations == 0
