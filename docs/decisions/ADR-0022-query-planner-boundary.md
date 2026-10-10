# ADR-0022: Query planning is a deterministic decision boundary, not a model

- Status: Accepted
- Date: 2026-10-10
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0002](ADR-0002-language-model-boundary.md), [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0013](ADR-0013-benchmark-identity-and-versioning.md), [ADR-0018](ADR-0018-answer-persistence.md)

## Context

The target runtime routes a turn through *planner → retrieval → model → …*, and retrieval must stay
optional: a greeting should not fetch evidence, a knowledge question should, and a follow-up can lean
on what the conversation already has. Today retrieval exists but is only ever invoked explicitly by a
caller; `ConversationManager.send` accepts an `EvidencePack` but nothing decides whether one is
needed or which prior sources a follow-up should be constrained to. That decision-maker does not
exist yet.

The obvious move is to ask a language model to plan. There is no language model (ADR-0002's boundary
has one development double behind it), and reaching for a commercial one would break ADR-0001. So the
first planner cannot be an LLM, must not pretend to understand meaning, and must be replaceable when a
real model lands.

## Problem

How should the system decide, for a turn, what retrieval to perform — deterministically, offline,
inspectably and in a versioned way — without an LLM, without a natural-language discourse resolver,
and without selecting retrieval strategies that no retriever backs yet?

## Options considered

- **An LLM planner.** Rejected: there is no model, an external one breaks local-first (ADR-0001), and
  its reasoning could not be reported honestly as a deterministic decision.
- **Undocumented keyword hacks in the conversation manager.** Rejected: it couples a policy the
  architecture wants isolated into the orchestration layer, and an undisclosed keyword rule is exactly
  the "arbitrary hack" the project forbids — untestable and unattributable.
- **A typed decision boundary with a deterministic rule implementation.** Chosen.

## Decision

`chakaso.planning` defines a `QueryPlanner` protocol — `plan(message, *, conversation=None) ->
QueryPlan` — with one implementation, `DeterministicQueryPlanner`. The application depends on the
boundary, not the rules, exactly as it depends on the model boundary rather than an adapter
(ADR-0002): a future intelligent planner replaces the implementation behind the same signature.

A `QueryPlan` is a frozen, inspectable record of the decision and carries, at minimum: the retrieval
`mode` and whether retrieval is `retrieval_required`; the user's `original_query` and a
`normalized_query`; `source_constraints` (identifiers owned by retrieval — the planner never mints one,
ADR-0003); `top_k`; a `reused_evidence` flag; the `context_used` labels and `explanation` reasons that
actually fired; and a `planner_version`. Construction enforces the honesty rules so they cannot be
violated by a careless caller: a mode and its `retrieval_required` flag cannot disagree, a plan that
retrieves must have a non-empty query, an evidence-reuse claim must name the sources it reuses, and
every plan must carry a non-empty explanation.

**The planner's vocabulary is fixed, its capability is not.** `QueryMode` names `none`, `lexical`,
`dense` and `hybrid`, but the deterministic planner selects only `lexical` (or none): `DENSE` and
`HYBRID` are refused at construction because no retriever backs them yet (dense and hybrid arrive in
later phases behind an embedding boundary). Declaring the modes keeps the interface stable; refusing
to emit unbacked ones keeps the output honest.

**Follow-up vs new-topic uses conversation state, not a discourse parser.** `DeterministicQueryPlanner`
reads `Conversation`'s `active_topic`, `entities` and `previous_source_ids`. A turn is a follow-up when
a conversation with prior sources exists and the turn either shares a word with the topic/entities or
uses an anaphoric continuation cue (`it`, `that`, `more` …); a follow-up retrieves constrained to the
prior sources, otherwise retrieval is fresh. The cue lists are a documented, closed heuristic — keyword
rules, echoed verbatim in each plan's `explanation`, and never presented as understanding. Resolving
references like "the second one" from natural language is explicitly out of scope.

**Queries are normalized without being rewritten.** `normalize_query` collapses whitespace, composes to
Unicode NFC and trims trailing terminal punctuation, changing no words, no case and no interior
punctuation. Both the original and normalized forms are stored so nothing is lost.

**Plans are versioned.** Every plan carries `planner_version`; the rules ship under a version string,
so a later planner can be compared against this one on the same input (the reproducibility discipline
of ADR-0013/ADR-0020 applied to planning).

## Reasoning

- **A decision layer earns its boundary by having a second implementation in mind.** Naming
  `QueryPlanner` now, while only a deterministic one exists, is not premature the way a database was
  (ADR-0018): the runtime on the other side of this seam is real and imminent, and the plan shape is
  exercised by the deterministic rules immediately.
- **Inspectability is the research requirement.** A plan that lists the rules that fired lets an
  experiment attribute a retrieval choice to a specific reason and a specific planner version; an
  LLM's opaque choice could not be. Refusing to fabricate reasoning is why `explanation` is mandatory.
- **Refusing unbacked modes is the honest half of naming them.** The enum is the vocabulary the system
  is moving toward; refusing to emit `dense` until a retriever exists is what keeps a plan from lying
  about what will actually run.

## Trade-offs

- The rules are keyword heuristics. They will misclassify some turns (an over-eager "tell me about
  your day" retrieves and finds nothing; a subtle follow-up that shares no word and uses no cue reads
  as a new topic). That is accepted: the alternative — pretending semantic judgment exists — is worse,
  and the boundary makes a better planner a replacement, not a rewrite.
- Only `lexical` is backed today, so the planner cannot yet realize the `dense`/`hybrid` modes it can
  name. The gap is stated, not hidden.
- Reusing conversation state means follow-up detection inherits whatever quality the topic/entity
  annotation has; nothing here does discourse resolution, and the extension point is where a real one
  would go.

## Consequences

- The conversation runtime (a later phase) can plan → retrieve → generate → persist by consuming a
  `QueryPlan`, with retrieval genuinely optional and the all-or-nothing turn guarantee preserved.
- A planner must never mint evidence identifiers or render a URL from a mode it inferred; it references
  sources retrieval owns (ADR-0003).
- Retrieval strategies that do not exist yet cannot be selected, so no plan can silently promise a
  capability the system does not have.
- `planner_version` makes planner behavior a controlled variable in experiments: changing the rules is
  a version change, comparable against the old one rather than an invisible drift.
