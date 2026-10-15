"""The application composition root: configuration becomes a running system here.

The runtime (`chakaso.runtime`) conducts one turn over a manager, a planner and a set of
retrieval services, but it is handed those pieces rather than building them — ADR-0025 keeps
it a thin coordinator. Something has to turn a resolved `Config` into those concrete
dependencies, and that wiring belongs in exactly one place so a run is reproducible from its
configuration. This module is that place.

It composes, in the order the boundaries were designed for:

1. the model, selected through the registry behind the `LanguageModel` boundary (ADR-0002);
2. the answer store, selected from `[storage]` (ADR-0021) — the configured backend is used,
   never silently substituted with the in-memory one;
3. the corpus, an in-memory set of evidence built from whatever documents a caller supplies;
4. the retrieval services for every backed strategy, and the query planner configured with the
   selected strategy and its limits (ADR-0022/ADR-0024);
5. the conversation manager, which owns the atomic turn, the claim extractor and the record
   (ADR-0008/ADR-0018);
6. the `RetrievalAwareConversation` that ties them together — or, when retrieval is disabled by
   configuration, the manager on its own, which is a deliberate mode rather than a degraded one.

Nothing here is behaviour the components do not already own: no ranking, no citation logic, no
persistence mechanics. It is construction and dependency injection only, which is why a test can
hand it a fake model, an injected store or a prepared corpus and exercise the real composition.
The only model that exists is still the deterministic development double and the only embedding
is still the non-semantic fixture, so the assembled application grounds, cites and records
honestly while producing no real answer and making no semantic claim.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from chakaso.answers import (
    AnswerId,
    AnswerStore,
    InMemoryAnswerStore,
    SQLiteAnswerStore,
    UnknownAnswerError,
)
from chakaso.claims.extract import ClaimExtractor, SentenceClaimExtractor
from chakaso.config import Config, LoadedConfig
from chakaso.conversation import Conversation, ConversationManager, Reply, new_conversation_id
from chakaso.core.errors import ChakasoError
from chakaso.core.identifiers import SourceId
from chakaso.correction import (
    CorrectionRecord,
    Reassessment,
    merge_evidence,
    reassess_and_record,
)
from chakaso.correction.errors import AmbiguousReassessmentTargetError
from chakaso.correction.trigger import (
    DeterministicReassessmentTrigger,
    ReassessmentTriggerKind,
    TriggerDecision,
)
from chakaso.evidence import EvidencePack, SourceRecord
from chakaso.grounding import describe_semantic_capability
from chakaso.models import GenerationParams, LanguageModel, ModelMetadata, create_model
from chakaso.models.registry import ModelRegistry
from chakaso.planning import DeterministicQueryPlanner, QueryMode, QueryPlan
from chakaso.retrieval import Corpus, HybridConfig, ingest_file
from chakaso.runtime import RetrievalAwareConversation, build_retrieval_services

__all__ = [
    "Application",
    "ApplicationError",
    "ApplicationTurn",
    "Capabilities",
    "ReassessmentOutcome",
    "build_application",
    "build_corpus",
    "describe_capabilities",
]


class ApplicationError(ChakasoError):
    """The configured dependencies cannot be assembled into a running application.

    Raised for composition-time contradictions a caller must fix — most often a durable
    backend chosen with no path to persist to — rather than a failure inside a turn, which
    keeps its own, more specific type.
    """


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class ApplicationTurn:
    """One completed turn through the application, uniform over retrieval on and off.

    ``plan`` and ``retrieved`` describe what grounding happened when the runtime is wired;
    with retrieval disabled they are ``None`` and ``False`` because no planner ran. The reply
    is the same object the manager produces either way, so a caller reads one type.
    """

    reply: Reply
    plan: QueryPlan | None
    retrieved: bool
    evidence_context: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The answer text, exactly as the model produced it."""
        return self.reply.text


@dataclass(frozen=True, slots=True)
class ReassessmentOutcome:
    """The result of a composed, explicit reassessment of a stored answer (P21).

    It carries the trigger decision that identified the target, the reassessment the existing
    boundary produced, and the append-only ``CorrectionRecord`` that stands as the distinct event —
    the original ``AnswerRecord`` is never touched. The chunk counts make the evidence basis visible
    so a reader can see what the decision was made against; the record's own ``summary`` is a
    structural tally, never revised prose (there is no language model).
    """

    target: AnswerId
    decision: TriggerDecision
    reassessment: Reassessment
    record: CorrectionRecord
    standing_chunk_count: int
    new_chunk_count: int

    @property
    def changed(self) -> bool:
        """Whether the reassessment decided to do anything to the answer at all."""
        return self.reassessment.changed

    @property
    def evidence_available(self) -> bool:
        """Whether any evidence (standing or newly retrieved) backed the reassessment."""
        return (self.standing_chunk_count + self.new_chunk_count) > 0


class Application:
    """A fully wired conversation application and the store that persists its answers."""

    def __init__(
        self,
        *,
        manager: ConversationManager,
        runtime: RetrievalAwareConversation | None,
        store: AnswerStore,
        corpus: Corpus,
        config: Config,
        store_path: str,
        close_store: Callable[[], None] | None,
        reassessment_trigger: DeterministicReassessmentTrigger | None = None,
    ) -> None:
        self._manager = manager
        self._runtime = runtime
        self._store = store
        self._corpus = corpus
        self._config = config
        self._store_path = store_path
        self._close_store = close_store
        self._reassessment_trigger = (
            reassessment_trigger
            if reassessment_trigger is not None
            else DeterministicReassessmentTrigger()
        )

    # -- capabilities a caller reports -----------------------------------------

    @property
    def conversation(self) -> ConversationManager:
        """The manager that owns the live conversation and records each turn."""
        return self._manager

    @property
    def model_metadata(self) -> ModelMetadata:
        """Identity of the model behind the boundary, including its double status."""
        return self._manager.model_metadata

    @property
    def store(self) -> AnswerStore:
        """The configured answer store the application records into."""
        return self._store

    @property
    def corpus(self) -> Corpus:
        """The evidence corpus retrieval searches (empty when nothing was supplied)."""
        return self._corpus

    @property
    def retrieval_enabled(self) -> bool:
        """Whether turns are grounded by the retrieval-aware runtime."""
        return self._runtime is not None

    @property
    def strategy(self) -> QueryMode:
        """The retrieval strategy the planner was configured to select."""
        return QueryMode(self._config.retrieval.strategy)

    @property
    def storage_backend(self) -> str:
        """The configured answer-store backend name."""
        return self._config.storage.backend

    @property
    def storage_path(self) -> str:
        """The durable store path, or an empty string for the in-memory backend."""
        return self._store_path

    # -- conducting turns ------------------------------------------------------

    def turn(self, message: str) -> ApplicationTurn:
        """Run one turn, through the runtime when retrieval is enabled, else the manager.

        Raises whatever the runtime or manager raises — a planning or retrieval failure leaves
        the conversation unchanged (the turn is atomic), and a persistence failure aborts it
        before the new state is adopted (ADR-0008/ADR-0018).
        """
        if self._runtime is not None:
            grounded = self._runtime.turn(message)
            return ApplicationTurn(
                reply=grounded.reply,
                plan=grounded.plan,
                retrieved=grounded.retrieved,
                evidence_context=grounded.evidence_context,
            )
        return ApplicationTurn(reply=self._manager.send(message), plan=None, retrieved=False)

    # -- explicit, composed reassessment (P21) --------------------------------

    def reassess_followup(self, message: str) -> ReassessmentOutcome | None:
        """Reassess a stored answer when a follow-up *explicitly names* it, else do nothing.

        This is the composition P21 adds, and it is deliberately not an autonomous loop. The front
        edge is the deterministic :class:`~chakaso.correction.trigger.DeterministicReassessmentTrigger`;
        the operation is the existing ``reassess_and_record`` boundary. The two are separate so intent
        and execution are independently testable.

        Returns ``None`` when the follow-up names no stored answer, which is an ordinary conversational
        turn — reassessment is never inferred from a vague reference, and an ordinary follow-up is never
        treated as a correction request. When exactly one answer is named, the prior record is read from
        the store and re-grounded against the evidence this follow-up makes available: the chunks still
        standing in the corpus for the answer's own sources (withdrawn support then reads as withdrawal)
        merged with whatever the retrievers return for the follow-up. It raises the boundary's own typed
        errors rather than guessing — never mutating the original answer or its provenance (ADR-0028).

        Raises:
            AmbiguousReassessmentTargetError: the follow-up named two or more distinct answers; the
                runtime refuses to choose between them.
            UnknownAnswerError: the named answer is not recorded in the store.
        """
        decision = self._reassessment_trigger.detect(message)
        if decision.kind is ReassessmentTriggerKind.NO_TARGET:
            return None
        if decision.kind is ReassessmentTriggerKind.AMBIGUOUS:
            named = ", ".join(str(answer_id) for answer_id in decision.references)
            message_text = (
                f"cannot reassess: the follow-up names {len(decision.references)} stored answers "
                f"({named}); name exactly one to reconsider"
            )
            raise AmbiguousReassessmentTargetError(message_text)

        target = decision.answer_id
        if target is None:  # defensive: EXPLICIT always carries one target
            return None
        prior = self._store.get(target)
        if prior is None:
            message_text = f"cannot reassess answer {target}: it is not recorded in the store"
            raise UnknownAnswerError(message_text)

        standing = _standing_evidence(self._corpus, prior.evidence_source_ids)
        new_evidence = self._runtime.gather_evidence(message) if self._runtime is not None else None
        composed = standing if new_evidence is None else merge_evidence(standing, new_evidence)
        reassessment, record = reassess_and_record(self._store, target, composed)
        return ReassessmentOutcome(
            target=target,
            decision=decision,
            reassessment=reassessment,
            record=record,
            standing_chunk_count=len(standing.chunks),
            new_chunk_count=0 if new_evidence is None else len(new_evidence.chunks),
        )

    def close(self) -> None:
        """Release the store's resources, if it holds any. Idempotent."""
        if self._close_store is not None:
            self._close_store()
            self._close_store = None

    def __enter__(self) -> Application:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class Capabilities:
    """A read-only report of what the configuration selects, without running a turn.

    It names the real capabilities a command would wire: the model adapter and whether it is a
    development double, the retrieval strategy and limits, and the answer-store backend. It reads
    the model's own metadata rather than a hard-coded string, so it reports what the component says
    about itself. It opens no store and touches no corpus, so inspecting capabilities never creates
    a database or reads a file.
    """

    model_id: str
    model_display_name: str
    model_is_development_double: bool
    retrieval_enabled: bool
    retrieval_strategy: str
    retrieval_top_k: int
    storage_backend: str
    storage_path: str
    semantic_real_evaluator_available: bool
    semantic_available_judges: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "model": {
                "model_id": self.model_id,
                "display_name": self.model_display_name,
                "development_double": self.model_is_development_double,
            },
            "retrieval": {
                "enabled": self.retrieval_enabled,
                "strategy": self.retrieval_strategy,
                "top_k": self.retrieval_top_k,
            },
            "storage": {"backend": self.storage_backend, "path": self.storage_path},
            "semantic_evaluation": {
                "real_evaluator_available": self.semantic_real_evaluator_available,
                "available_judges": list(self.semantic_available_judges),
            },
        }


def describe_capabilities(
    loaded: LoadedConfig,
    *,
    model: LanguageModel | None = None,
    registry: ModelRegistry | None = None,
) -> Capabilities:
    """Report the capabilities the resolved configuration selects, offline and without a turn.

    Builds only the model (through the registry, ADR-0002) to read its self-declared metadata; a
    misconfigured adapter raises :class:`~chakaso.models.UnknownModelError` here, exactly as the
    composition would. Nothing is opened or fetched.
    """
    config = loaded.config
    resolved_model = model if model is not None else create_model(config, registry=registry)
    metadata = resolved_model.metadata
    semantic = describe_semantic_capability()
    return Capabilities(
        model_id=metadata.model_id,
        model_display_name=metadata.display_name,
        model_is_development_double=metadata.development_double,
        retrieval_enabled=config.retrieval.enabled,
        retrieval_strategy=config.retrieval.strategy,
        retrieval_top_k=config.retrieval.top_k,
        storage_backend=config.storage.backend,
        storage_path=config.storage.path,
        semantic_real_evaluator_available=semantic.real_evaluator_available,
        semantic_available_judges=semantic.available_judges,
    )


def build_corpus(
    paths: Sequence[str | Path],
    *,
    max_bytes: int,
    clock: Callable[[], datetime] | None = None,
) -> Corpus:
    """Ingest the named local documents into one corpus.

    Ingestion reads only the explicit paths given and never crawls a directory (the rules the
    retrieval layer already enforces). A path that cannot be read raises the retrieval layer's
    own :class:`~chakaso.retrieval.IngestionError`; nothing is swallowed here.
    """
    corpus = Corpus()
    for path in paths:
        corpus.add_document(ingest_file(path, max_bytes=max_bytes, now=clock))
    return corpus


def _standing_evidence(corpus: Corpus, source_ids: Sequence[SourceId]) -> EvidencePack:
    """The chunks still present in ``corpus`` for the named sources, assembled as a valid pack.

    A persisted answer references its evidence by source identifier only (P20): the bodies live in the
    in-memory corpus and are not re-readable from the answer store. So the evidence "still standing" for
    an answer is exactly what the current corpus holds for the sources it named — no more, and nothing
    invented (ADR-0003). A source the corpus no longer holds contributes nothing, which is precisely how
    support that was withdrawn reads as withdrawn rather than silently retained.
    """
    wanted = set(source_ids)
    if not wanted:
        return EvidencePack.of((), ())
    chunks = tuple(chunk for chunk in corpus.chunks if chunk.source_id in wanted)
    sources: dict[SourceId, SourceRecord] = {}
    for chunk in chunks:
        source = corpus.source_for(chunk.source_id)
        if source is not None:
            sources[source.source_id] = source
    return EvidencePack.of(chunks, sources.values())


def _select_store(config: Config) -> tuple[AnswerStore, str, Callable[[], None] | None]:
    """Build the answer store the configuration selects, with its path and a releaser."""
    storage = config.storage
    if storage.backend == "memory":
        return InMemoryAnswerStore(), "", None
    if storage.backend == "sqlite":
        path = storage.path.strip()
        if not path:
            message = (
                "storage.backend is 'sqlite' but storage.path is empty; the durable store "
                "needs an explicit file path. Set [storage].path or choose backend 'memory'."
            )
            raise ApplicationError(message)
        store = SQLiteAnswerStore(path, journal_mode=storage.journal_mode, initialize=True)
        return store, path, store.close
    # The schema pattern admits only these two; reaching here means a backend bypassed it.
    message = f"unsupported storage backend {storage.backend!r}"
    raise ApplicationError(message)


def build_application(
    loaded: LoadedConfig,
    *,
    corpus: Corpus | None = None,
    model: LanguageModel | None = None,
    store: AnswerStore | None = None,
    claim_extractor: ClaimExtractor | None = None,
    registry: ModelRegistry | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Application:
    """Assemble the configured dependencies into a runnable application.

    Every keyword argument exists for dependency injection in tests: pass a ``model``, a
    ``store`` or a prepared ``corpus`` to exercise the real composition without the built-in
    double, a durable file or a filesystem. Omit them to build exactly what configuration names
    — which is the path the CLI takes.

    Raises:
        ApplicationError: a durable backend is selected with no path to write to.
        UnknownModelError: the configured model adapter is not registered.
    """
    config = loaded.config
    resolved_model = model if model is not None else create_model(config, registry=registry)

    close_store: Callable[[], None] | None = None
    store_path = ""
    if store is None:
        store, store_path, close_store = _select_store(config)

    resolved_extractor = (
        claim_extractor if claim_extractor is not None else SentenceClaimExtractor()
    )
    manager = ConversationManager(
        resolved_model,
        Conversation(conversation_id=new_conversation_id()),
        params=GenerationParams(
            max_new_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
        ),
        max_context_turns=config.model.max_context_turns,
        now=clock if clock is not None else _utcnow,
        answer_store=store,
        claim_extractor=resolved_extractor,
    )

    resolved_corpus = corpus if corpus is not None else Corpus()
    runtime: RetrievalAwareConversation | None = None
    if config.retrieval.enabled:
        hybrid = HybridConfig(
            rrf_k=config.retrieval.rrf_k,
            candidate_k=config.retrieval.candidate_k,
        )
        services = build_retrieval_services(resolved_corpus, hybrid_config=hybrid)
        planner = DeterministicQueryPlanner(
            default_top_k=config.retrieval.top_k,
            retrieval_mode=QueryMode(config.retrieval.strategy),
        )
        runtime = RetrievalAwareConversation(
            manager=manager,
            planner=planner,
            services=services,
            default_top_k=config.retrieval.top_k,
        )

    return Application(
        manager=manager,
        runtime=runtime,
        store=store,
        corpus=resolved_corpus,
        config=config,
        store_path=store_path,
        close_store=close_store,
    )
