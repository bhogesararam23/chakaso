"""The retrieval-aware conversation runtime.

This package composes the layers built in earlier phases into a single per-turn flow —
planner → retrieval (lexical / dense / hybrid) → model → claims → citation validation →
grounding → `AnswerRecord` → durable store — as a thin coordinator, not a monolith
(ADR-0025). It reuses the existing `ConversationManager`, `QueryPlanner`, `Retriever`s and
`AnswerStore` rather than reimplementing any of them, keeps retrieval optional (the planner
decides), preserves the turn's all-or-nothing atomicity, and records the planner's decision and
the retrieval strategy as inspectable provenance and metadata on the answer.

It changes nothing about the absence of a language model: generation still goes through the
model boundary to the deterministic development double, and dense/hybrid still run over the
non-semantic fixture embedding. The runtime makes Chakaso *executable end to end around a real
model that does not yet exist*; it does not produce one, and it makes no semantic claim.
"""

from __future__ import annotations

from chakaso.runtime.context import EVIDENCE_CONTEXT_HEADER, render_evidence_context
from chakaso.runtime.conversation import (
    GroundedTurn,
    RetrievalAwareConversation,
    build_retrieval_services,
)
from chakaso.runtime.errors import ConversationRuntimeError

__all__ = [
    "EVIDENCE_CONTEXT_HEADER",
    "ConversationRuntimeError",
    "GroundedTurn",
    "RetrievalAwareConversation",
    "build_retrieval_services",
    "render_evidence_context",
]
