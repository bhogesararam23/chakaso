"""Tests for the retrieval-aware conversation runtime.

The runtime is where the phases meet, so the tests walk real conversational sequences and check
the properties that only emerge from the composition: retrieval is genuinely optional (a greeting
reaches no retriever), a knowledge turn retrieves and records its evidence, a follow-up reuses the
conversation's prior sources, the planner's decision and the retrieval strategy are inspectable on
the answer, and a failure before the recording turn leaves the conversation untouched. The model
behind it is the deterministic double and the embedding is the non-semantic fixture, so nothing
here asserts answer quality or semantic understanding — only that grounding, citation, provenance
and persistence are wired correctly and atomically.
"""

from __future__ import annotations

import pytest

from chakaso.answers import InMemoryAnswerStore
from chakaso.claims import SentenceClaimExtractor
from chakaso.config import load_config
from chakaso.conversation import Conversation, ConversationManager, new_conversation_id
from chakaso.evidence import EvidencePack
from chakaso.models import create_model
from chakaso.planning import DeterministicQueryPlanner, QueryMode
from chakaso.retrieval import (
    Corpus,
    RetrievalError,
    RetrievalService,
    ingest_text,
)
from chakaso.runtime import (
    EVIDENCE_CONTEXT_HEADER,
    ConversationRuntimeError,
    RetrievalAwareConversation,
    build_retrieval_services,
    render_evidence_context,
)

_SPEC = ingest_text("Chakaso grounds answers in retrieved evidence.", label="spec")
_GEO = ingest_text("Paris is the capital of France.", label="geo")


def _corpus() -> Corpus:
    corpus = Corpus()
    corpus.add_document(_SPEC)
    corpus.add_document(_GEO)
    return corpus


def _manager(store: InMemoryAnswerStore) -> ConversationManager:
    return ConversationManager(
        create_model(load_config().config),
        Conversation(conversation_id=new_conversation_id()),
        answer_store=store,
        claim_extractor=SentenceClaimExtractor(),
    )


def _runtime(
    store: InMemoryAnswerStore | None = None,
) -> tuple[RetrievalAwareConversation, InMemoryAnswerStore]:
    answer_store = store if store is not None else InMemoryAnswerStore()
    runtime = RetrievalAwareConversation(
        manager=_manager(answer_store),
        planner=DeterministicQueryPlanner(),
        services=build_retrieval_services(_corpus()),
    )
    return runtime, answer_store


# -- evidence-context renderer ---------------------------------------------------


def test_render_empty_pack_yields_no_context() -> None:
    assert render_evidence_context(EvidencePack.of((), ())) == ()


def test_rendered_context_carries_ids_reference_and_text() -> None:
    pack = EvidencePack.of(_SPEC.chunks, [_SPEC.source])

    blocks = render_evidence_context(pack)

    assert blocks[0] == EVIDENCE_CONTEXT_HEADER
    assert str(_SPEC.chunks[0].chunk_id) in blocks[1]
    assert "text:spec" in blocks[1]
    assert _SPEC.chunks[0].text in blocks[1]


def test_rendered_context_is_deterministic() -> None:
    pack = EvidencePack.of(_SPEC.chunks, [_SPEC.source])

    assert render_evidence_context(pack) == render_evidence_context(pack)


# -- service wiring --------------------------------------------------------------


def test_services_are_labelled_with_their_actual_strategy() -> None:
    services = build_retrieval_services(_corpus())

    assert set(services) == {QueryMode.LEXICAL, QueryMode.DENSE, QueryMode.HYBRID}
    assert services[QueryMode.HYBRID].corpus is not None


# -- construction guards ---------------------------------------------------------


def test_runtime_requires_every_backed_mode_to_be_wired() -> None:
    store = InMemoryAnswerStore()
    services = build_retrieval_services(_corpus())
    del services[QueryMode.HYBRID]

    with pytest.raises(ConversationRuntimeError, match="hybrid"):
        RetrievalAwareConversation(
            manager=_manager(store), planner=DeterministicQueryPlanner(), services=services
        )


def test_runtime_rejects_bad_default_top_k() -> None:
    store = InMemoryAnswerStore()

    with pytest.raises(ConversationRuntimeError, match="default_top_k"):
        RetrievalAwareConversation(
            manager=_manager(store),
            planner=DeterministicQueryPlanner(),
            services=build_retrieval_services(_corpus()),
            default_top_k=0,
        )


# -- the conversational sequence (§42) ------------------------------------------


def test_a_greeting_retrieves_nothing_but_still_records() -> None:
    runtime, store = _runtime()

    turn = runtime.turn("hello there")

    assert turn.retrieval is None
    assert turn.retrieved is False
    assert turn.plan.mode is QueryMode.NONE
    assert turn.evidence_context == ()
    record = store.list()[0]
    assert record.evidence_source_ids == ()
    assert record.provenance["retrieval_strategy"] == "none"


def test_a_knowledge_question_retrieves_records_and_reports_strategy() -> None:
    runtime, store = _runtime()

    turn = runtime.turn("What is Chakaso?")

    assert turn.retrieved is True
    assert turn.plan.mode is QueryMode.LEXICAL
    assert turn.retrieval is not None
    assert _SPEC.chunks[0].chunk_id in {c.chunk_id for c in turn.retrieval.pack.chunks}

    record = store.list()[0]
    assert _SPEC.source.source_id in record.evidence_source_ids
    assert record.provenance["planner_mode"] == "lexical"
    assert record.provenance["retrieval_strategy"] == "lexical"
    assert record.retrieval_metadata["retriever"] == "lexical"


def test_a_follow_up_reuses_the_conversations_prior_sources() -> None:
    runtime, _ = _runtime()
    runtime.turn("What is Chakaso?")

    follow_up = runtime.turn("tell me more about it")

    assert follow_up.plan.reused_evidence is True
    assert _SPEC.source.source_id in follow_up.plan.source_constraints


def test_a_topic_change_does_not_reuse_stale_sources() -> None:
    runtime, _ = _runtime()
    runtime.turn("What is Chakaso?")

    turn = runtime.turn("What is the capital of France?")

    assert turn.plan.reused_evidence is False
    assert turn.plan.source_constraints == ()
    assert turn.retrieval is not None
    assert _GEO.source.source_id in turn.retrieval.pack.sources


def test_dense_mode_records_dense_strategy() -> None:
    store = InMemoryAnswerStore()
    services = build_retrieval_services(_corpus())
    runtime = RetrievalAwareConversation(
        manager=_manager(store),
        planner=DeterministicQueryPlanner(retrieval_mode=QueryMode.DENSE),
        services=services,
    )

    turn = runtime.turn("What is Chakaso?")

    assert turn.plan.mode is QueryMode.DENSE
    assert store.list()[0].provenance["retrieval_strategy"] == "dense"


# -- atomicity -------------------------------------------------------------------


def test_a_retrieval_failure_leaves_the_conversation_unchanged() -> None:
    store = InMemoryAnswerStore()
    manager = _manager(store)
    services = build_retrieval_services(_corpus())

    class _Boom:
        def retrieve(self, query, *, top_k, source_ids=None):
            message = "retriever exploded"
            raise RetrievalError(message)

    services[QueryMode.LEXICAL] = RetrievalService(
        _corpus(), retriever=_Boom(), retriever_name="lexical"
    )
    runtime = RetrievalAwareConversation(
        manager=manager, planner=DeterministicQueryPlanner(), services=services
    )

    before = manager.conversation
    with pytest.raises(RetrievalError, match="exploded"):
        runtime.turn("What is Chakaso?")

    # Retrieval happens before the manager mutates anything, so a failed turn changes nothing.
    assert manager.conversation is before
    assert store.list() == ()


def test_retrieval_outcome_provenance_is_captured_without_invented_citations() -> None:
    # The double cites nothing, so the record holds evidence but no citations — the honest
    # state of a grounded turn with a non-generative model.
    runtime, store = _runtime()

    runtime.turn("What is Chakaso?")

    record = store.list()[0]
    assert record.evidence_source_ids  # evidence was supplied
    assert record.cited_source_ids == ()  # the double invented no citation
