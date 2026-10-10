"""Tests for the query-planning layer.

Planning is a decision boundary (ADR-0022), so the tests are about decisions and their
honesty: normalization preserves meaning and never rewrites a query; a plan refuses to
misreport itself (a mode that contradicts its ``retrieval_required`` flag, an evidence-reuse
claim with nothing reused, a plan with no explanation); the deterministic rules route the
conversational turns the specification names — greeting → no retrieval, question → fresh
retrieval, follow-up → reuse prior sources, topic change → no stale reuse; and the planner
refuses the dense/hybrid modes no retriever backs. The planner is checked to satisfy the
protocol, because a future planner replacing it must fit the same shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from chakaso.conversation import Conversation, new_conversation_id
from chakaso.core.identifiers import SourceId
from chakaso.planning import (
    PLANNER_VERSION,
    DeterministicQueryPlanner,
    InvalidQueryPlanError,
    PlanningError,
    QueryMode,
    QueryPlan,
    QueryPlanner,
    normalize_query,
)
from chakaso.retrieval import ingest_text

NOW = datetime(2026, 10, 10, tzinfo=UTC)
_SPEC = ingest_text("Chakaso grounds answers in retrieved evidence.", label="spec")
_SPEC_SOURCE = _SPEC.source.source_id


def _conversation(*, topic: str = "chakaso") -> Conversation:
    conversation = Conversation(conversation_id=new_conversation_id())
    conversation = conversation.with_assistant_turn(
        "Noted.", at=NOW, evidence_source_ids=(_SPEC_SOURCE,)
    )
    return conversation.with_active_topic(topic)


def base_plan(
    *,
    mode: QueryMode = QueryMode.LEXICAL,
    original_query: str = "some query",
    normalized_query: str = "some query",
    retrieval_required: bool = True,
    planner_version: str = "planner-0.1",
    source_constraints: tuple[SourceId, ...] = (),
    top_k: int | None = None,
    reused_evidence: bool = False,
    context_used: tuple[str, ...] = (),
    explanation: tuple[str, ...] = ("a real reason",),
) -> QueryPlan:
    return QueryPlan(
        mode=mode,
        original_query=original_query,
        normalized_query=normalized_query,
        retrieval_required=retrieval_required,
        planner_version=planner_version,
        source_constraints=source_constraints,
        top_k=top_k,
        reused_evidence=reused_evidence,
        context_used=context_used,
        explanation=explanation,
    )


# -- QueryMode -------------------------------------------------------------------


def test_only_none_skips_retrieval() -> None:
    assert QueryMode.NONE.requires_retrieval is False
    assert QueryMode.LEXICAL.requires_retrieval is True
    assert QueryMode.DENSE.requires_retrieval is True
    assert QueryMode.HYBRID.requires_retrieval is True


# -- query normalization ---------------------------------------------------------


def test_normalization_collapses_whitespace_and_trims_the_ends() -> None:
    assert normalize_query("  What   is\nChakaso?\t ") == "What is Chakaso"


def test_normalization_composes_unicode() -> None:
    # "e" + combining acute accent normalizes to the single precomposed code point.
    assert normalize_query("cafe\u0301") == "caf\u00e9"


def test_normalization_preserves_words_case_and_interior_punctuation() -> None:
    assert normalize_query("uses C++ in Rankings.") == "uses C++ in Rankings"


def test_normalization_is_idempotent_and_total() -> None:
    for text in ("", "   ", "?!!", "What is Chakaso??", "a  b"):
        once = normalize_query(text)
        assert normalize_query(once) == once


# -- plan validation -------------------------------------------------------------


def test_mode_and_retrieval_flag_cannot_disagree() -> None:
    with pytest.raises(InvalidQueryPlanError, match="disagree"):
        base_plan(mode=QueryMode.NONE, retrieval_required=True)
    with pytest.raises(InvalidQueryPlanError, match="disagree"):
        base_plan(mode=QueryMode.LEXICAL, retrieval_required=False)


def test_a_retrieving_plan_must_have_a_query() -> None:
    with pytest.raises(InvalidQueryPlanError, match="non-empty query"):
        base_plan(normalized_query="")


def test_reuse_requires_something_to_reuse() -> None:
    with pytest.raises(InvalidQueryPlanError, match="naming no source"):
        base_plan(reused_evidence=True, source_constraints=())

    reused = base_plan(reused_evidence=True, source_constraints=(_SPEC_SOURCE,))
    assert reused.is_follow_up is True


def test_a_plan_must_explain_itself() -> None:
    with pytest.raises(InvalidQueryPlanError, match="explain"):
        base_plan(explanation=())


def test_blank_planner_version_is_rejected() -> None:
    with pytest.raises(InvalidQueryPlanError, match="version"):
        base_plan(planner_version="   ")


def test_top_k_must_be_positive_when_given() -> None:
    with pytest.raises(InvalidQueryPlanError, match="top_k"):
        base_plan(top_k=0)
    assert base_plan(top_k=5).top_k == 5


def test_source_constraints_are_deduplicated_in_order() -> None:
    other = ingest_text("Another fact.", label="other").source.source_id
    plan = base_plan(source_constraints=(_SPEC_SOURCE, other, _SPEC_SOURCE))
    assert plan.source_constraints == (_SPEC_SOURCE, other)


# -- deterministic planner: construction guards ----------------------------------


@pytest.mark.parametrize("mode", [QueryMode.NONE, QueryMode.DENSE, QueryMode.HYBRID])
def test_the_planner_refuses_modes_it_cannot_back(mode: QueryMode) -> None:
    # Only LEXICAL has a retriever today; naming dense/hybrid must not mean promising them.
    with pytest.raises(PlanningError, match=r"not backed|NONE"):
        DeterministicQueryPlanner(retrieval_mode=mode)


def test_planner_rejects_bad_top_k() -> None:
    with pytest.raises(PlanningError, match="default_top_k"):
        DeterministicQueryPlanner(default_top_k=0)


def test_planner_satisfies_the_protocol_and_versions_its_plans() -> None:
    planner: QueryPlanner = DeterministicQueryPlanner()
    assert isinstance(planner, QueryPlanner)
    assert planner.planner_version == PLANNER_VERSION
    assert planner.plan("hello").planner_version == PLANNER_VERSION


# -- deterministic planner: the decisions ----------------------------------------


def test_a_greeting_plans_no_retrieval() -> None:
    plan = DeterministicQueryPlanner().plan("hello there")

    assert plan.mode is QueryMode.NONE
    assert plan.retrieval_required is False
    assert plan.reused_evidence is False


def test_an_empty_turn_plans_no_retrieval() -> None:
    assert DeterministicQueryPlanner().plan("   ").mode is QueryMode.NONE


def test_a_knowledge_question_plans_fresh_retrieval() -> None:
    plan = DeterministicQueryPlanner().plan("  What is Chakaso? ")

    assert plan.mode is QueryMode.LEXICAL
    assert plan.retrieval_required is True
    assert plan.reused_evidence is False
    assert plan.source_constraints == ()
    assert plan.original_query == "  What is Chakaso? "
    assert plan.normalized_query == "What is Chakaso"
    assert any("question mark" in reason for reason in plan.explanation)


def test_a_follow_up_reuses_the_conversations_prior_sources() -> None:
    conversation = _conversation()

    plan = DeterministicQueryPlanner().plan(
        "What does it say about chakaso evidence?", conversation=conversation
    )

    assert plan.mode is QueryMode.LEXICAL
    assert plan.reused_evidence is True
    assert plan.is_follow_up is True
    assert plan.source_constraints == (_SPEC_SOURCE,)
    assert "conversation_state" in plan.context_used


def test_an_anaphoric_follow_up_reuses_even_with_a_cue_word() -> None:
    conversation = _conversation()

    plan = DeterministicQueryPlanner().plan("tell me more", conversation=conversation)

    # "more" is anaphoric and prior sources exist, so reuse wins over the "tell" cue.
    assert plan.reused_evidence is True
    assert "continuation_cue" in plan.context_used


def test_a_topic_change_does_not_reuse_stale_sources() -> None:
    conversation = _conversation()

    plan = DeterministicQueryPlanner().plan(
        "What is the population of Paris?", conversation=conversation
    )

    assert plan.mode is QueryMode.LEXICAL
    assert plan.reused_evidence is False
    assert plan.source_constraints == ()


def test_without_prior_sources_a_continuation_word_still_retrieves_fresh() -> None:
    # An empty conversation has nothing to reuse, so "tell me more" is fresh retrieval.
    empty = Conversation(conversation_id=new_conversation_id())
    plan = DeterministicQueryPlanner().plan("tell me more", conversation=empty)

    assert plan.reused_evidence is False
    assert plan.source_constraints == ()


def test_planning_is_deterministic_for_identical_input() -> None:
    planner = DeterministicQueryPlanner()
    conversation = _conversation()

    first = planner.plan("What does it say about chakaso?", conversation=conversation)
    second = planner.plan("What does it say about chakaso?", conversation=conversation)

    assert first == second


def test_default_top_k_is_carried_onto_a_retrieval_plan() -> None:
    plan = DeterministicQueryPlanner(default_top_k=3).plan("what is evidence")

    assert plan.top_k == 3
