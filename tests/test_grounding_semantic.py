"""Tests for the semantic grounding boundary and its composition.

The boundary must be real and the capability must not be faked, so the tests assert the fixture
is honest (it returns what a caller supplied, ignoring the evidence text), that the adapter
conforms to the existing grounding boundary without touching callers, that a contradiction is
recorded only when the judge names disagreeing chunks and never picks a winner, and that
composition keeps grounding a *separate* dimension from citation — a claim can cite valid evidence
and still be judged contradicted.
"""

from __future__ import annotations

from chakaso.claims.model import Claim, ClaimStatus
from chakaso.core.identifiers import ChunkId
from chakaso.correction import CorrectionDecision, reassess
from chakaso.evaluation import evaluate_answer
from chakaso.evidence import EvidencePack
from chakaso.grounding import (
    FixtureSemanticJudge,
    GroundingEvaluator,
    SemanticEvaluation,
    SemanticGroundingEvaluator,
    SemanticJudge,
    SemanticJudgeError,
    SemanticJudgement,
    describe_semantic_capability,
)
from chakaso.retrieval import ingest_text

_ALPHA = ingest_text("Alpha is a fact.", label="alpha")
_BETA = ingest_text("Beta is another fact.", label="beta")
PACK = EvidencePack.of((*_ALPHA.chunks, *_BETA.chunks), [_ALPHA.source, _BETA.source])
CHUNK_A = _ALPHA.chunks[0].chunk_id
CHUNK_B = _BETA.chunks[0].chunk_id


def claim(index: int, cited: tuple[ChunkId, ...] = ()) -> Claim:
    return Claim(answer_id="ans", position=index, text=f"claim {index}", cited_evidence_ids=cited)


class _NamingJudge:
    """A judge that also names the chunks it found in conflict, to exercise that path."""

    name = "naming-fixture-0.1"

    def __init__(self, conflicts: tuple[ChunkId, ...]) -> None:
        self._conflicts = conflicts

    def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:
        return SemanticEvaluation(
            claim_id=claim.claim_id,
            judgement=SemanticJudgement.CONTRADICTED,
            evaluator_name=self.name,
            rationale="two sources disagree",
            conflicting_chunk_ids=self._conflicts,
        )


def test_fixture_returns_the_supplied_relation_and_ignores_the_evidence() -> None:
    target = claim(0)
    other = claim(1)
    judge = FixtureSemanticJudge({target.claim_id: SemanticJudgement.SUPPORTED})

    supported = judge.judge(target, PACK)
    unlisted = judge.judge(other, PACK)

    assert supported.judgement is SemanticJudgement.SUPPORTED
    assert supported.status is ClaimStatus.SUPPORTED
    assert unlisted.judgement is SemanticJudgement.NOT_EVALUATED  # nothing supplied -> undecided


def test_the_semantic_adapter_conforms_to_the_grounding_boundary() -> None:
    evaluator = SemanticGroundingEvaluator(FixtureSemanticJudge({}))
    assert isinstance(evaluator, GroundingEvaluator)
    assert isinstance(FixtureSemanticJudge({}), SemanticJudge)


def test_adapter_maps_judgements_to_statuses_and_records_the_evaluator_name() -> None:
    supported = claim(0)
    contradicted = claim(1)
    evaluator = SemanticGroundingEvaluator(
        FixtureSemanticJudge(
            {
                supported.claim_id: SemanticJudgement.SUPPORTED,
                contradicted.claim_id: SemanticJudgement.CONTRADICTED,
            }
        )
    )

    result = evaluator.evaluate("ans", [supported, contradicted], PACK)

    assert result.status_of(supported) is ClaimStatus.SUPPORTED
    assert result.status_of(contradicted) is ClaimStatus.CONTRADICTED
    assert result.evaluator_name == "fixture-semantic-0.1"


def test_a_named_conflict_is_recorded_without_a_winner() -> None:
    conflicted = claim(0)
    evaluator = SemanticGroundingEvaluator(_NamingJudge((CHUNK_A, CHUNK_B)))

    result = evaluator.evaluate("ans", [conflicted], PACK)

    assert result.contradicted == (conflicted.claim_id,)
    assert [c.chunk_ids for c in result.contradictions] == [(CHUNK_A, CHUNK_B)]
    # the adapter names no winner; resolving the conflict is correction's job (ADR-0015/0016)


def test_an_unnamed_contradiction_sets_status_but_records_no_conflict() -> None:
    conflicted = claim(0)
    evaluator = SemanticGroundingEvaluator(
        FixtureSemanticJudge({conflicted.claim_id: SemanticJudgement.CONTRADICTED})
    )

    result = evaluator.evaluate("ans", [conflicted], PACK)

    assert result.status_of(conflicted) is ClaimStatus.CONTRADICTED
    assert result.contradictions == ()


def test_composition_keeps_grounding_separate_from_citation() -> None:
    # The claim cites valid supplied evidence (citation dimension: valid), yet the injected
    # semantic judge says the evidence contradicts it (grounding dimension: contradicted). The two
    # facts coexist because evaluate_answer keeps dimensions apart rather than merging them.
    good_citation = claim(0, (CHUNK_A,))
    evaluator = SemanticGroundingEvaluator(
        FixtureSemanticJudge({good_citation.claim_id: SemanticJudgement.CONTRADICTED})
    )

    result = evaluate_answer("ans", [good_citation], PACK, grounding_evaluator=evaluator)

    assert result.citations.valid == 1
    assert result.grounding.contradicted == (good_citation.claim_id,)
    assert result.evidence_coverage == 0.0
    assert result.is_grounded is False


def test_reassessment_can_take_the_semantic_evaluator_unchanged() -> None:
    prior = Claim(
        answer_id="ans",
        position=0,
        text="alpha claim",
        cited_evidence_ids=(CHUNK_A,),
        status=ClaimStatus.SUPPORTED,
    )
    evaluator = SemanticGroundingEvaluator(
        FixtureSemanticJudge({prior.claim_id: SemanticJudgement.CONTRADICTED})
    )

    result = reassess("ans", [prior], PACK, grounding_evaluator=evaluator)

    assert result.decision is CorrectionDecision.CORRECT
    assert result.evaluator_name == "fixture-semantic-0.1"


# -- contract hardening: evaluator version, errors, capability detection ------


class _VersionedJudge:
    name = "custom-semantic"
    version = "2.3"

    def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:
        return SemanticEvaluation(
            claim_id=claim.claim_id,
            judgement=SemanticJudgement.SUPPORTED,
            evaluator_name=self.name,
            evaluator_version=self.version,
        )


class _ErroringJudge:
    """Raises on one claim (an evaluator failure or abstention) and supports another."""

    name = "flaky-semantic"
    version = "0.1"

    def __init__(self, raises_for: object) -> None:
        self._raises_for = raises_for

    def judge(self, claim: Claim, evidence: EvidencePack) -> SemanticEvaluation:
        if claim.claim_id == self._raises_for:
            message = "evaluator could not decide"
            raise SemanticJudgeError(message)
        return SemanticEvaluation(
            claim_id=claim.claim_id,
            judgement=SemanticJudgement.SUPPORTED,
            evaluator_name=self.name,
            evaluator_version=self.version,
        )


def test_the_fixture_evaluation_carries_the_evaluator_version() -> None:
    target = claim(0)
    evaluation = FixtureSemanticJudge({target.claim_id: SemanticJudgement.SUPPORTED}).judge(
        target, PACK
    )

    assert evaluation.evaluator_name == "fixture-semantic-0.1"
    assert evaluation.evaluator_version == "0.1"


def test_a_changed_evaluator_identity_and_version_reaches_the_result() -> None:
    # The result's provenance reflects exactly which evaluator judged, so a different evaluator or
    # version is visible rather than hidden behind a single generic name.
    result = SemanticGroundingEvaluator(_VersionedJudge()).evaluate("ans", [claim(0)], PACK)

    assert result.evaluator_name == "custom-semantic"
    assert result.evaluator_version == "2.3"


def test_an_evaluator_error_is_recorded_as_not_evaluated_never_as_support() -> None:
    failed = claim(0)
    supported = claim(1)
    evaluator = SemanticGroundingEvaluator(_ErroringJudge(raises_for=failed.claim_id))

    result = evaluator.evaluate("ans", [failed, supported], PACK)

    # The judge raised for `failed`; it must not read as support, and the error must not abort
    # the answer — the other claim is still evaluated.
    assert result.status_of(failed) is ClaimStatus.NOT_EVALUATED
    assert result.status_of(supported) is ClaimStatus.SUPPORTED


def test_structural_grounding_still_works_without_a_semantic_evaluator() -> None:
    # The boundary is optional composition: omitting a semantic evaluator leaves the structural
    # dimension usable, judged by the default structural evaluator.
    result = evaluate_answer("ans", [claim(0, (CHUNK_A,))], PACK)

    assert result.citations.valid == 1
    assert result.grounding.evaluator_name == "structural-0.1"


def test_capability_detection_is_honest_that_no_real_evaluator_exists() -> None:
    capability = describe_semantic_capability()

    assert capability.real_evaluator_available is False
    assert capability.available_judges == ("fixture-semantic-0.1",)
    assert "not semantic" in capability.note or "FixtureSemanticJudge" in capability.note
