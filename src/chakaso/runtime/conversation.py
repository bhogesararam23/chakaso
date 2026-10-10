"""The retrieval-aware conversation runtime: plan, retrieve, answer, record — per turn.

This is the composition the earlier phases were building toward. It is deliberately a thin
coordinator, not a giant orchestrator (ADR-0022): it runs the planner, hands the chosen
retriever's evidence pack to the `ConversationManager`, and lets the manager do what it already
does — generate through the model boundary, resolve citations, extract claims, evaluate, build the
`AnswerRecord` and persist it atomically. Nothing here duplicates a validator or the store.

The pieces it wires are the ones that already exist and are tested on their own:

* a `QueryPlanner` decides whether and how to retrieve (retrieval is genuinely optional — a
  greeting reaches no retriever);
* a `Mapping[QueryMode, RetrievalService]` maps each mode to a service built over a corpus, so
  lexical, dense and hybrid are all selectable;
* the manager, given an `AnswerStore`, records each grounded answer atomically.

Because planning and retrieval happen *before* `send`, and `send` is transactional (ADR-0008),
the whole turn is all-or-nothing: a plan for a mode nothing serves, or a retrieval that fails,
raises before the conversation changes, so no phantom answer and no half-recorded correction are
possible. The planner's decision and the retrieval strategy are recorded on the answer as
structured provenance and metadata, so a grounded turn is inspectable after the fact (ADR-0025).

The only model behind the boundary is still the deterministic development double and the only
embedding is the non-semantic fixture (ADR-0023); the runtime therefore grounds, cites and persists
honestly, but produces no real answer and makes no semantic claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from chakaso.conversation import ConversationManager, Reply
from chakaso.planning import QueryMode, QueryPlan, QueryPlanner
from chakaso.retrieval import (
    DEFAULT_TOP_K,
    Corpus,
    DenseRetriever,
    FixtureEmbeddingModel,
    HybridConfig,
    HybridRetriever,
    LexicalRetriever,
    RetrievalOutcome,
    RetrievalService,
    build_dense_index,
    build_lexical_index,
)
from chakaso.retrieval.embeddings import EmbeddingModel
from chakaso.runtime.context import render_evidence_context
from chakaso.runtime.errors import ConversationRuntimeError

__all__ = ["GroundedTurn", "RetrievalAwareConversation", "build_retrieval_services"]


@dataclass(frozen=True, slots=True)
class GroundedTurn:
    """One completed runtime turn: the plan, what retrieval did, and the manager's reply."""

    plan: QueryPlan
    retrieval: RetrievalOutcome | None
    evidence_context: tuple[str, ...]
    reply: Reply

    @property
    def retrieved(self) -> bool:
        """Whether this turn performed a retrieval (the planner requested evidence)."""
        return self.retrieval is not None


def build_retrieval_services(
    corpus: Corpus,
    *,
    embedder: EmbeddingModel | None = None,
    hybrid_config: HybridConfig | None = None,
) -> dict[QueryMode, RetrievalService]:
    """Wire one `RetrievalService` per retrieval mode over ``corpus``.

    This is the composition-root helper the runtime is handed: it builds the lexical, dense and
    hybrid retrievers over the same corpus and labels each service with the strategy it actually
    uses, so a dense pack never claims to be lexical. The dense index uses the embedding boundary,
    defaulting to the non-semantic fixture double (ADR-0023).
    """
    chunks = corpus.chunks
    lexical = LexicalRetriever(build_lexical_index(chunks))
    model = embedder if embedder is not None else FixtureEmbeddingModel()
    dense = DenseRetriever(build_dense_index(chunks, model), model)
    hybrid = HybridRetriever(lexical, dense, config=hybrid_config)
    return {
        QueryMode.LEXICAL: RetrievalService(corpus, retriever=lexical, retriever_name="lexical"),
        QueryMode.DENSE: RetrievalService(corpus, retriever=dense, retriever_name="dense"),
        QueryMode.HYBRID: RetrievalService(corpus, retriever=hybrid, retriever_name="hybrid"),
    }


class RetrievalAwareConversation:
    """Conducts turns where a planner decides retrieval and the manager records the answer."""

    def __init__(
        self,
        *,
        manager: ConversationManager,
        planner: QueryPlanner,
        services: Mapping[QueryMode, RetrievalService],
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Compose a manager, a planner and a per-mode retrieval service.

        Raises:
            ConversationRuntimeError: ``default_top_k`` is below 1, or no service is wired for
                a mode the runtime may be asked to retrieve with (every backed mode needs one).
        """
        if default_top_k < 1:
            message = f"default_top_k must be at least 1, got {default_top_k}"
            raise ConversationRuntimeError(message)
        missing = sorted(
            mode.value
            for mode in (QueryMode.LEXICAL, QueryMode.DENSE, QueryMode.HYBRID)
            if mode not in services
        )
        if missing:
            message = f"no retrieval service wired for mode(s): {', '.join(missing)}"
            raise ConversationRuntimeError(message)

        self._manager = manager
        self._planner = planner
        self._services = dict(services)
        self._default_top_k = default_top_k

    @property
    def conversation(self) -> ConversationManager:
        """The managed conversation this runtime advances."""
        return self._manager

    def turn(self, message: str) -> GroundedTurn:
        """Run one retrieval-aware turn and return its grounded result.

        Planning and retrieval are read-only and happen before the manager mutates anything, so a
        failure in either leaves the conversation untouched; the recording turn itself is
        atomic in the manager (ADR-0008/ADR-0018).
        """
        plan = self._planner.plan(message, conversation=self._manager.conversation)

        if not plan.retrieval_required:
            reply = self._manager.send(message, provenance=_plan_provenance(plan, strategy=None))
            return GroundedTurn(plan=plan, retrieval=None, evidence_context=(), reply=reply)

        service = self._services.get(plan.mode)
        if service is None:  # defensive: the constructor requires every backed mode be wired
            message = f"the planner chose {plan.mode.value!r}, but no service is wired for it"
            raise ConversationRuntimeError(message)

        top_k = plan.top_k if plan.top_k is not None else self._default_top_k
        outcome = service.search(
            plan.normalized_query,
            top_k=top_k,
            source_ids=plan.source_constraints or None,
        )
        context = render_evidence_context(outcome.pack)
        reply = self._manager.send(
            message,
            evidence=outcome.pack,
            provenance=_plan_provenance(plan, strategy=outcome.retrieval_config["retriever"]),
        )
        return GroundedTurn(plan=plan, retrieval=outcome, evidence_context=context, reply=reply)


def _plan_provenance(plan: QueryPlan, *, strategy: str | None) -> dict[str, str]:
    """Turn the planner's decision into structured, inspectable answer provenance (§56).

    Only what the planner actually decided is recorded — the mode, its version, whether it
    reused conversation evidence, and the reasons it reported — never anything inferred.
    """
    data = {
        "planner_mode": plan.mode.value,
        "planner_version": plan.planner_version,
        "retrieval_strategy": strategy if strategy is not None else "none",
        "reused_evidence": str(plan.reused_evidence).lower(),
    }
    if plan.explanation:
        data["planner_reasons"] = "; ".join(plan.explanation)
    return data
