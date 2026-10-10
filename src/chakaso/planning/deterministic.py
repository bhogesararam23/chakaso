"""The deterministic query planner.

This is the first — and only — implementation of the `QueryPlanner` boundary. It is a
typed rule system, not a language model and not a natural-language discourse resolver
(ADR-0022). Every decision it reports is a real outcome of the rules below; nothing here
understands meaning, and no mode a retriever cannot back is ever selected. The rules are
small and written out precisely so they can be tested and so a future planner can be
compared against them under a different `planner_version`.

The rules, in precedence order (each decision is echoed in the plan's ``explanation``):

1. **Empty after normalization.** If the turn has nothing left to search for once
   whitespace and terminal punctuation are normalized away, no retrieval is planned.
2. **Follow-up reusing prior evidence.** With a conversation in hand that already has
   sources, a turn is a follow-up when it either shares a word with the conversation's
   active topic or entities, or uses an anaphoric continuation cue (``it``, ``that``,
   ``more``, ``else``, ``again``, ``first``, ``second`` …). A follow-up retrieves
   constrained to those prior sources rather than freshly — the conversation's evidence,
   not a new sweep.
3. **Information-seeking turn.** A turn carrying an interrogative or a retrieval verb
   (``what``, ``who``, ``why``, ``how``, ``find``, ``list``, ``according``, ``explain`` …)
   or ending in a question mark requests fresh, unconstrained retrieval.
4. **Everything else** — a greeting, a bare statement with no cue — needs no evidence, so
   no retrieval.

The cue lists are the documented heuristic this planner is built on: they are keyword
rules, deliberately simple and inspectable, and they are *not* presented as understanding.
Today the planner can only choose ``LEXICAL`` retrieval (or none); ``DENSE`` and ``HYBRID``
are refused at construction because no retriever backs them yet.
"""

from __future__ import annotations

from collections.abc import Sequence

from chakaso.conversation.state import Conversation
from chakaso.core.identifiers import SourceId
from chakaso.planning.errors import PlanningError
from chakaso.planning.mode import QueryMode
from chakaso.planning.normalize import normalize_query
from chakaso.planning.plan import QueryPlan

__all__ = ["PLANNER_VERSION", "DeterministicQueryPlanner"]

#: The version of these rules, stamped on every plan so a later planner is comparable
#: against an earlier one and a plan records exactly what produced it (ADR-0022).
PLANNER_VERSION = "deterministic-planner-0.1"

#: Interrogatives and retrieval verbs that mark a turn as asking for evidence. A closed,
#: documented list — the whole of this planner's "is this a question" signal.
_INFORMATION_CUES = frozenset(
    {
        "according",
        "compare",
        "define",
        "describe",
        "explain",
        "find",
        "how",
        "list",
        "lookup",
        "retrieve",
        "search",
        "tell",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
    }
)

#: Anaphoric and continuation words that tie a turn back to what is already in the
#: conversation. Their presence, with prior sources, is what makes a turn a follow-up.
#: Deliberately excludes the definite article "the": it is too common to signal a
#: continuation on its own, and a new-topic question that merely says "the" must not be
#: misread as reusing prior evidence.
_CONTINUATION_CUES = frozenset(
    {
        "again",
        "also",
        "both",
        "either",
        "else",
        "first",
        "it",
        "its",
        "more",
        "neither",
        "second",
        "than",
        "that",
        "these",
        "they",
        "them",
        "this",
        "those",
    }
)


class DeterministicQueryPlanner:
    """A `QueryPlanner` that decides by written rules, never by a model."""

    def __init__(
        self,
        *,
        default_top_k: int | None = None,
        retrieval_mode: QueryMode = QueryMode.LEXICAL,
    ) -> None:
        """Configure the planner.

        Args:
            default_top_k: How many results a planned retrieval asks for, or ``None`` to
                leave the decision to the retriever's own default.
            retrieval_mode: The mode used when the planner decides to retrieve. Only
                ``QueryMode.LEXICAL`` is backed by a retriever today; ``DENSE`` and
                ``HYBRID`` are refused rather than promised, and ``NONE`` cannot retrieve.

        Raises:
            PlanningError: ``retrieval_mode`` is not a mode a retriever backs, or
                ``default_top_k`` is present but below 1.
        """
        if retrieval_mode is QueryMode.NONE:
            message = "a planner cannot retrieve with retrieval_mode=NONE"
            raise PlanningError(message)
        if retrieval_mode is not QueryMode.LEXICAL:
            message = (
                f"retrieval_mode {retrieval_mode.value!r} is not backed by a retriever yet; "
                "only 'lexical' is available (dense and hybrid arrive with their retrievers)"
            )
            raise PlanningError(message)
        if default_top_k is not None and default_top_k < 1:
            message = f"default_top_k must be at least 1 when given, got {default_top_k}"
            raise PlanningError(message)

        self._retrieval_mode = retrieval_mode
        self._default_top_k = default_top_k

    @property
    def planner_version(self) -> str:
        """The version stamped on every plan this planner produces."""
        return PLANNER_VERSION

    def plan(self, message: str, *, conversation: Conversation | None = None) -> QueryPlan:
        """Return the plan for ``message`` given ``conversation`` context."""
        normalized = normalize_query(message)
        original = message

        if not normalized:
            return QueryPlan(
                mode=QueryMode.NONE,
                original_query=original,
                normalized_query=normalized,
                retrieval_required=False,
                planner_version=PLANNER_VERSION,
                explanation=("nothing to retrieve: the turn is empty after normalization",),
            )

        tokens = _tokens(normalized)
        prior_sources = conversation.previous_source_ids if conversation is not None else ()
        shares_topic = conversation is not None and bool(tokens & _context_tokens(conversation))
        anaphoric = bool(tokens & _CONTINUATION_CUES)
        follow_up = bool(prior_sources) and (shares_topic or anaphoric)

        if follow_up:
            return self._retrieval_plan(
                original,
                normalized,
                source_constraints=prior_sources,
                reused=True,
                shares_topic=shares_topic,
                anaphoric=anaphoric,
            )

        cue_matches = frozenset(tokens & _INFORMATION_CUES)
        if cue_matches or _ends_with_question(original):
            return self._retrieval_plan(
                original,
                normalized,
                cue_matches=cue_matches,
                question=_ends_with_question(original),
            )

        return QueryPlan(
            mode=QueryMode.NONE,
            original_query=original,
            normalized_query=normalized,
            retrieval_required=False,
            planner_version=PLANNER_VERSION,
            explanation=("no information-seeking cue and not a follow-up: no evidence is needed",),
        )

    def _retrieval_plan(
        self,
        original: str,
        normalized: str,
        *,
        source_constraints: Sequence[SourceId] = (),
        reused: bool = False,
        shares_topic: bool = False,
        anaphoric: bool = False,
        cue_matches: frozenset[str] = frozenset(),
        question: bool = False,
    ) -> QueryPlan:
        """Build a retrieve-lexically plan, reporting the reasons that selected it."""
        reasons: list[str] = []
        context_used: list[str] = []
        if reused:
            reasons.append(
                f"follow-up: retrieval constrained to {len(source_constraints)} prior source(s)"
            )
            context_used.append("conversation_state")
            if shares_topic:
                reasons.append("shares vocabulary with the conversation's active topic or entities")
                context_used.append("active_topic_or_entities")
            if anaphoric:
                reasons.append("uses an anaphoric continuation cue (e.g. 'it', 'that', 'more')")
                context_used.append("continuation_cue")
        else:
            reasons.append("information-seeking turn: fresh retrieval requested")
            if cue_matches:
                reasons.append(f"matched cue word(s): {', '.join(sorted(cue_matches))}")
            if question:
                reasons.append("turn ends with a question mark")

        return QueryPlan(
            mode=self._retrieval_mode,
            original_query=original,
            normalized_query=normalized,
            retrieval_required=True,
            planner_version=PLANNER_VERSION,
            source_constraints=tuple(source_constraints),
            top_k=self._default_top_k,
            reused_evidence=reused,
            context_used=tuple(context_used),
            explanation=tuple(reasons),
        )


def _ends_with_question(text: str) -> bool:
    """Whether the turn, before normalization removed it, ended in a question mark."""
    return text.rstrip().endswith("?")


def _context_tokens(conversation: Conversation) -> set[str]:
    """The tokens the conversation is about: its active topic and its entities."""
    tokens: set[str] = set()
    if conversation.active_topic:
        tokens |= _tokens(conversation.active_topic)
    for entity in conversation.entities:
        tokens |= _tokens(entity)
    return tokens


def _tokens(text: str) -> set[str]:
    """Casefolded alphanumeric word tokens, for cue and topic matching only.

    This is the whole of the planner's lexical awareness: split on whitespace, keep only
    alphanumeric characters, casefold. It never runs on the stored query — only on the
    copies used to compare against cue lists — so it cannot rewrite what the user asked.
    """
    result: set[str] = set()
    for word in text.split():
        token = "".join(char for char in word if char.isalnum()).casefold()
        if token:
            result.add(token)
    return result
