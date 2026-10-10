"""The conversation manager: orchestration for one conversation.

One turn, in order:

1. reject a user turn that has no text;
2. append the user turn to a *pending* conversation, not to the live one;
3. project the recent conversation into messages a model can be given;
4. call the model through the :class:`~chakaso.models.LanguageModel` boundary;
5. reject a response that is empty or attributed to a different model;
6. resolve any evidence references against the evidence that was supplied;
7. append the assistant turn, and only then adopt the new state as the live one.

Step 7 is the reason steps 2 to 6 work on a copy. A turn either completes or the
conversation is unchanged (ADR-0008), so a model failure cannot leave a user turn
stranded without a reply or an assistant turn recorded for a response that was
never produced.

The manager depends on the model *boundary*, never on a concrete adapter (ADR-0002).
Tests use the deterministic development double; the manager cannot tell the
difference, which is the property that makes a real model a configuration change
rather than a rewrite.

**The seam.** :meth:`ConversationManager.send` accepts an
:class:`~chakaso.evidence.EvidencePack`. Nothing in this module obtains one: there
is no retrieval, no fetching, no ranking and no index. When query planning and
retrieval exist, they will run before `send` and pass the pack in. Until then the
parameter is exercised only by tests and by callers that already have content, and
a request with no evidence resolves no citations — which is the honest behaviour,
not a degraded one.

**Answer persistence.** When the manager is constructed with an
:class:`~chakaso.answers.AnswerStore`, each completed turn is also recorded as an
:class:`~chakaso.answers.AnswerRecord` — claims extracted through an injected
``ClaimExtractor``, an evaluation against the supplied evidence, and the model identity — and
saved *before* the live conversation is adopted. A persistence failure therefore aborts the
turn and leaves the conversation unchanged, extending ADR-0008's all-or-nothing guarantee to
the answer history (ADR-0018). Without a store the manager behaves exactly as it did before.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from chakaso.answers import AnswerRecord, AnswerStore, new_answer_id
from chakaso.claims.extract import ClaimExtractor
from chakaso.conversation.errors import (
    ConversationManagerError,
    InvalidGenerationError,
    InvalidUserInputError,
)
from chakaso.conversation.state import Conversation, Turn
from chakaso.evaluation import evaluate_answer
from chakaso.evidence import CitationResolution, EvidencePack, resolve_citations
from chakaso.models import (
    FinishReason,
    GenerationParams,
    GenerationResult,
    LanguageModel,
    ModelMetadata,
)

__all__ = ["ConversationManager", "Reply"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Reply:
    """The outcome of one completed turn.

    Carries the generation result and the citation resolution alongside the new
    state, because both are evidence about the answer rather than decoration: a
    caller that drops them cannot tell a truncated answer from a complete one, or
    an answer that referenced evidence it was never given.
    """

    conversation: Conversation
    turn: Turn
    generation: GenerationResult
    citation_resolution: CitationResolution
    answer_record: AnswerRecord | None = None

    @property
    def text(self) -> str:
        """The answer text, exactly as the model produced it."""
        return self.turn.text

    @property
    def is_truncated(self) -> bool:
        """Whether generation stopped at the length ceiling rather than finishing.

        A caller that cannot tell these apart will present a cut-off answer as a
        complete one.
        """
        return self.generation.finish_reason is FinishReason.LENGTH

    @property
    def has_unresolved_references(self) -> bool:
        """Whether the answer referenced evidence that was not supplied."""
        return not self.citation_resolution.is_clean


class ConversationManager:
    """Conducts one conversation against a language model.

    Holds the current :class:`~chakaso.conversation.Conversation` and advances it
    only when a turn completes. It does not implement a model, does not retrieve
    anything, and does not decide whether retrieval would be useful — that is query
    planning's job, and it does not exist yet.
    """

    def __init__(
        self,
        model: LanguageModel,
        conversation: Conversation,
        *,
        params: GenerationParams | None = None,
        max_context_turns: int | None = None,
        now: Callable[[], datetime] | None = None,
        answer_store: AnswerStore | None = None,
        claim_extractor: ClaimExtractor | None = None,
    ) -> None:
        """Create a manager for ``conversation``, using ``model``.

        Args:
            model: Any implementation of the model boundary. The manager never
                imports a concrete one.
            conversation: The conversation to continue. Pass a fresh
                :class:`~chakaso.conversation.Conversation` to start one, or an
                existing one to resume.
            params: Generation parameters for every turn. Defaults to
                :class:`~chakaso.models.GenerationParams` defaults, which are the
                same values the shipped configuration carries.
            max_context_turns: How many of the most recent turns to project into a
                model request. ``None`` projects the whole conversation. This is a
                stopgap: it bounds growth without a tokenizer, and it is not a token
                budget, so it cannot be described as fitting a model's context
                window.
            now: Clock, injectable so that tests do not depend on wall-clock time.
            answer_store: If given, each completed turn is recorded as an
                :class:`~chakaso.answers.AnswerRecord` and saved here before the
                conversation is adopted, so a persistence failure aborts the turn and
                leaves the conversation unchanged (ADR-0008, ADR-0018). Omitting it
                leaves behaviour exactly as it was: an in-memory conversation, no
                answer record.
            claim_extractor: How an answer is decomposed into claims for its record.
                Used only when ``answer_store`` is given. ``None`` records an answer
                with no claims rather than inventing a decomposition.

        Raises:
            ConversationManagerError: ``max_context_turns`` is present but below 1.
        """
        if max_context_turns is not None and max_context_turns < 1:
            message = f"max_context_turns must be at least 1 when given, got {max_context_turns}"
            raise ConversationManagerError(message)

        self._model = model
        self._conversation = conversation
        self._params = params if params is not None else GenerationParams()
        self._max_context_turns = max_context_turns
        self._now = now if now is not None else _utcnow
        self._answer_store = answer_store
        self._claim_extractor = claim_extractor

    @property
    def conversation(self) -> Conversation:
        """The current conversation. Immutable; advancing it produces a new one."""
        return self._conversation

    @property
    def model_metadata(self) -> ModelMetadata:
        """Identity and capabilities of the model this manager is using.

        Exposed so that a caller can report what produced an answer, and so that a
        front end can say plainly when the model is a development double rather
        than a language model.
        """
        return self._model.metadata

    @property
    def max_context_turns(self) -> int | None:
        """The projection bound, or ``None`` when the whole conversation is used."""
        return self._max_context_turns

    def send(
        self,
        text: str,
        *,
        evidence: EvidencePack | None = None,
        provenance: Mapping[str, str] | None = None,
    ) -> Reply:
        """Conduct one turn: record the user's message, answer it, record the answer.

        Args:
            text: The user's message. Surrounding whitespace is stripped before it
                becomes a turn; the answer is never altered.
            evidence: The evidence supplied to the model for this turn. Omit it when
                nothing was retrieved. A request with no evidence resolves no
                citations, so a reference in the answer is reported as unresolved
                rather than resolved to a source.
            provenance: Extra key/value metadata to attach to the recorded answer, when
                a record is produced — how a caller (e.g. a retrieval-aware runtime) ran
                the turn. It is merged into the record's provenance alongside the model
                identity and never changes what is cited, evaluated or persisted; omitted,
                the record carries only its own provenance, exactly as before.

        Returns:
            The reply, including the new conversation state.

        Raises:
            InvalidUserInputError: ``text`` is empty or whitespace.
            InvalidGenerationError: the model returned an empty response, or one
                attributed to a different model.
            ModelError, UnsupportedCapabilityError: propagated unchanged from the
                model boundary.
            ConversationError: the resulting turn would make the conversation state
                inconsistent. The conversation is left unchanged.
        """
        prompt = text.strip()
        if not prompt:
            message = "a user turn must contain text; the message was empty or whitespace"
            raise InvalidUserInputError(message)

        # An empty pack rather than a special case: "nothing was retrieved" and
        # "retrieval returned nothing" are the same situation, and treating them
        # alike means a reference in the answer is unresolved either way.
        supplied = evidence if evidence is not None else EvidencePack.of((), ())

        pending = self._conversation.with_user_turn(
            prompt, at=self._timestamp_after(self._conversation)
        )
        messages = pending.to_model_messages(max_turns=self._max_context_turns)

        result = self._model.generate(messages, self._params)
        self._reject_unusable(result)

        resolution = resolve_citations(result.text, supplied)

        updated = pending.with_assistant_turn(
            result.text,
            at=self._timestamp_after(pending),
            evidence_source_ids=tuple(supplied.sources),
            cited_source_ids=resolution.cited_source_ids,
        )

        answer_record: AnswerRecord | None = None
        if self._answer_store is not None:
            answer_record = self._build_answer_record(
                updated, result, resolution, supplied, extra_provenance=provenance
            )
            # Persistence is part of the turn's atomicity (ADR-0018): a save that fails
            # raises here, before the live conversation is swapped, so the conversation is
            # left exactly as it was and no half-recorded answer is left behind.
            self._answer_store.save(answer_record)

        # The live conversation changes only here, once every step that can fail
        # has succeeded.
        self._conversation = updated
        return Reply(
            conversation=updated,
            turn=updated.turns[-1],
            generation=result,
            citation_resolution=resolution,
            answer_record=answer_record,
        )

    def _build_answer_record(
        self,
        conversation: Conversation,
        result: GenerationResult,
        resolution: CitationResolution,
        supplied: EvidencePack,
        *,
        extra_provenance: Mapping[str, str] | None = None,
    ) -> AnswerRecord:
        """Assemble the persistent record for the assistant turn just completed.

        Built after the new conversation exists, so its timestamp and turn index match the
        turn actually recorded. It cites only what resolution found in the supplied pack, so it
        cannot violate the record's citation-ownership rule. Claims and evaluation are honest
        structure, not understanding: the deterministic double has no trained decomposition, so a
        caller supplies an extractor or the answer is recorded with no claims.
        """
        answer_id = new_answer_id()
        claims = (
            self._claim_extractor.extract(str(answer_id), result.text)
            if self._claim_extractor is not None
            else ()
        )
        metadata = self._model.metadata
        provenance: dict[str, str] = {"model_display_name": metadata.display_name}
        if extra_provenance:
            provenance.update(extra_provenance)
        return AnswerRecord(
            answer_id=answer_id,
            conversation_id=conversation.conversation_id,
            text=result.text,
            created_at=conversation.turns[-1].created_at,
            model_id=result.model_id,
            model_is_development_double=metadata.development_double,
            claims=claims,
            evidence_source_ids=tuple(supplied.sources),
            cited_source_ids=resolution.cited_source_ids,
            retrieval_metadata=supplied.retrieval_config,
            evaluation=evaluate_answer(str(answer_id), claims, supplied),
            turn_index=len(conversation.turns) - 1,
            provenance=provenance,
        )

    def _reject_unusable(self, result: GenerationResult) -> None:
        """Raise unless ``result`` can be recorded as an answer."""
        if not result.text.strip():
            message = f"model {result.model_id!r} returned an empty response"
            raise InvalidGenerationError(message)

        expected = self._model.metadata.model_id
        if result.model_id != expected:
            message = (
                f"the response is attributed to {result.model_id!r}, but the manager is "
                f"using {expected!r}. Generation provenance cannot be trusted, and an "
                "answer whose model is unknown cannot be evaluated."
            )
            raise InvalidGenerationError(message)

    def _timestamp_after(self, conversation: Conversation) -> datetime:
        """Return the current time, never earlier than ``conversation``'s last turn.

        History is append-only and the state type rejects a turn that predates its
        predecessor. A clock that moved backwards — an NTP correction, a container
        with a bad clock — would otherwise turn into a confusing state error, so the
        previous turn's timestamp is used as a floor.
        """
        moment = self._now()
        last = conversation.last_turn
        if last is not None and moment < last.created_at:
            return last.created_at
        return moment
