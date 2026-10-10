# ADR-0025: The retrieval-aware conversation runtime is a thin coordinator, not a monolith

- Status: Accepted
- Date: 2026-10-10
- Relates to: [ADR-0002](ADR-0002-language-model-boundary.md), [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0008](ADR-0008-transactional-turns.md), [ADR-0021](ADR-0021-sqlite-answer-store.md), [ADR-0022](ADR-0022-query-planner-boundary.md), [ADR-0024](ADR-0024-hybrid-rank-fusion.md)

## Context

Every part of the target pipeline now exists on its own and is tested on its own: a
deterministic query planner (ADR-0022), three retrievers behind one `Retriever` protocol
(ADR-0010/ADR-0023/ADR-0024), the model boundary with only a development double behind it
(ADR-0002), claims, citation validation, structural grounding, the `AnswerRecord` and a durable
store (ADR-0017/ADR-0018/ADR-0021). What is missing is the thing that makes them a *system*: a
per-turn flow that runs the planner, retrieves the evidence it asks for, hands it to the manager,
and lets the manager record and persist the answer.

The obvious failure mode of "integrate everything" is a god object — one class that plans,
retrieves, generates, validates, evaluates and stores, owning every detail and untestable as a
unit. The whole architecture has been arranged to avoid exactly that, with each capability behind
its own boundary. The integration must compose those boundaries, not absorb them.

## Problem

How should the runtime conduct a retrieval-aware turn — planner → retrieval → model → claims →
citation validation → grounding → `AnswerRecord` → durable store — while keeping every component
in its own boundary, keeping retrieval optional, and preserving the all-or-nothing turn guarantee
across the longer sequence?

## Options considered

- **Grow `ConversationManager` to plan and retrieve.** Rejected: it would drag fetching, ranking
  and query-shaping into the orchestration layer, the coupling `RetrievalService` was separated
  from the manager to prevent, and it would make a turn depend on retrieval machinery it must be
  able to skip (a greeting).
- **One runtime class that owns all the stages.** Rejected: the monolith the modular architecture
  exists to avoid; nothing would be independently testable and every boundary would collapse.
- **A thin coordinator over the existing services, chosen by configuration.** Chosen.

## Decision

`chakaso.runtime` adds a `RetrievalAwareConversation` that composes, but does not absorb, the
layers. For one turn it:

1. asks the `QueryPlanner` for a `QueryPlan` given the current conversation;
2. if the plan says retrieval is required, selects the `RetrievalService` wired for the plan's mode
   and searches with the plan's normalized query, `top_k` and prior-source constraints; otherwise
   passes no evidence;
3. renders the retrieved pack into a structured, citation-safe model-facing context
   (`render_evidence_context`) — identifiers and text only, never instructions (ADR-0003);
4. calls the unchanged `ConversationManager.send`, which generates through the model boundary,
   resolves citations, extracts claims, evaluates, builds the `AnswerRecord` and persists it
   atomically — and passes the plan along as answer provenance;
5. returns a `GroundedTurn` carrying the plan, the retrieval outcome, the evidence context and the
   manager's reply.

`build_retrieval_services(corpus, …)` is the composition-root helper that wires one service per
mode over a corpus, each labelled with the strategy it actually uses; the runtime is handed the
mode→service mapping rather than constructing retrievers itself, so it stays a coordinator. The
manager gained one optional parameter — caller-supplied `provenance` merged onto the recorded
answer — the only change to an existing component, made so the planner's decision and the retrieval
strategy become inspectable on the answer without the runtime rebuilding the record.

Retrieval stays optional because the planner, not the runtime, decides it: a greeting yields a
`NONE` plan and the manager is called with no evidence, producing a normal ungrounded turn. Atomicity
follows from ordering: planning, retrieval and rendering are read-only and happen *before* `send`;
`send` alone mutates, and it is already transactional (ADR-0008, extended to the record by
ADR-0018). So a failed plan, an unwired mode, or a retriever that raises aborts the turn before any
conversation or store change — no phantom answer, no half-recorded correction.

## Reasoning

- **A coordinator that only delegates keeps each boundary testable in isolation** and makes the
  runtime an assembly of proven parts rather than a new, untested behemoth. The "no giant
  orchestrator" requirement is honoured structurally: the runtime holds references to services,
  it does not reimplement them.
- **Doing retrieval before the transactional `send` reuses the existing guarantee** instead of
  inventing a second one; the turn is atomic because the only mutating step is atomic.
- **One optional manager parameter, not a record-building copy.** The manager already owns
  record construction correctly; the minimal seam to surface the planner's decision is a
  provenance argument, not duplicated evaluation logic.

## Trade-offs

- The runtime is inert without a real model: with the deterministic double it retrieves and records
  an answer that cites nothing (the double emits fixed text), so the loop is genuinely exercised but
  no citation is produced. That is the honest state of a runtime around a model that does not exist.
- `build_retrieval_services` builds all three retrievers eagerly over the corpus; at research scale
  this is fine, and a caller that only ever plans one mode can pass a smaller mapping (the runtime
  requires the modes it might use, not all three).
- The evidence-context renderer produces the model-facing representation but there is still no prompt
  template to place it in (no real model), so it is the boundary the model will consume rather than a
  claim that a prompt exists.

## Consequences

- The architecture's target flow is now executable end to end offline and deterministically, which is
  what the end-to-end and restart integration tests (this run's final units) exercise.
- The planner's decision and the retrieval strategy are recorded on each answer, making a grounded
  turn auditable: which mode ran, why, and against which sources.
- A real model plugs into this exact runtime through the existing boundaries; nothing in the
  coordinator changes when one appears.
- The no-network guard extends to `chakaso.runtime`; the runtime imports only offline layers and
  performs no I/O beyond what its injected services do.
