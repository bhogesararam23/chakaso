# ADR-0026: The application composition root is a thin, injectable layer separate from the runtime

- Status: Accepted
- Date: 2026-10-15
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0002](ADR-0002-language-model-boundary.md), [ADR-0006](ADR-0006-toml-configuration.md), [ADR-0018](ADR-0018-answer-persistence.md), [ADR-0021](ADR-0021-sqlite-answer-store.md), [ADR-0025](ADR-0025-retrieval-aware-runtime.md)

## Context

ADR-0025 made the retrieval-aware runtime a thin coordinator that is *handed* a manager, a
planner and a mode→service mapping rather than building them. Something still has to turn a
resolved `Config` into those concrete dependencies, and until now no such place existed: the CLI's
`chat` command composed only a bare `ConversationManager` with a model and an empty conversation,
so the runtime, the planner, retrieval and the answer store — all built and tested since ADR-0021 —
were exercised only by tests, never by an actual command.

Where that construction lives is a real choice with consequences. If it goes into the runtime, the
runtime stops being a coordinator (the monolith ADR-0025 avoided). If it is scattered across CLI
command functions, then configuration→dependency mapping is duplicated per command, an offline
default cannot be verified in one place, and there is no seam to inject a fake model or a prepared
store in tests. ADR-0002 already fixed one such point for the model (the registry); persistence
(ADR-0021) and retrieval (ADR-0024) need the same single, explicit composition.

## Problem

How should the assembled application be constructed from typed configuration so that: the CLI and
any future front end build the same graph; the configured answer store is used rather than silently
replaced with the in-memory default; a strategy or limit in a config file actually changes runtime
behaviour; and the whole thing is testable by injecting a model, store or corpus without a
filesystem, a network or a real model?

## Options considered

- **Compose inside `RetrievalAwareConversation`.** Rejected: it would absorb dependency selection
  and store construction, undoing ADR-0025's "thin coordinator".
- **Compose inline in each CLI command.** Rejected: duplicate configuration→object mapping, no
  single place to prove the offline default, and no injectable seam.
- **A dedicated composition module returning an assembled application object.** Chosen.

## Decision

`chakaso.app` is the application composition root. `build_application(loaded, *, corpus=…,
model=…, store=…, claim_extractor=…, registry=…, clock=…)` reads only the resolved `Config` and
wires, in one place: the model through `create_model` behind the `LanguageModel` boundary
(ADR-0002); the answer store selected from `[storage]` (ADR-0021) — `memory` gives the in-memory
store, `sqlite` gives the durable store at the configured path, and choosing `sqlite` with no path
is an explicit `ApplicationError` rather than a silent downgrade to memory; the retrieval services
for every backed strategy over a corpus (ADR-0023/ADR-0024, via `build_retrieval_services`); the
`DeterministicQueryPlanner` configured with `[retrieval].strategy` and `[retrieval].top_k`
(ADR-0022); and a `ConversationManager` that owns the atomic turn and records each answer. It
returns an `Application` that exposes a uniform `turn(message)` and reports its real capabilities
(model metadata, whether retrieval is wired, the strategy, the store backend and path, the corpus).

Retrieval configuration is part of the existing typed schema (ADR-0006), not a parallel system: a
new `[retrieval]` section (`enabled`, `strategy`, `top_k`, `candidate_k`, `rrf_k`,
`max_ingest_bytes`) is validated by the same specs/dataclass-parity mechanism as `model` and
`storage`. When `[retrieval].enabled` is false the `Application` conducts turns through the manager
alone — no planner, no retrieval — which is a deliberate mode, not a degraded one; persistence is
unaffected.

Every keyword argument is an injection point, so a test exercises the real composition with a fake
model, an in-memory store or a prepared corpus and none of the machinery is bypassed. `chakaso
chat` now builds its graph exclusively through this layer.

## Reasoning

- **One composition point makes offline-by-default verifiable.** The single function that maps
  configuration to objects is where "the default needs no credentials and no network" is guaranteed
  and tested, exactly as the registry localised model selection.
- **The configured store is authoritative, not advisory.** Selecting `sqlite` and quietly using
  memory would break the durability guarantee ADR-0021 exists to provide; raising when the durable
  backend has nowhere to write makes the contract honest.
- **A separate `Application` keeps ADR-0025's coordinator thin.** Construction and per-turn
  orchestration stay in different places; the runtime still does neither.
- **Typed config drives real behaviour and is tested that way.** Changing `strategy`, `top_k` or
  `enabled` is asserted to change what the assembled application does, not merely how a dataclass
  serialises.

## Trade-offs

- The `Application` wrapper is a small extra type; it exists because "retrieval enabled" and
  "retrieval disabled" produce two different conductors, and callers should read one `turn`
  interface rather than branch on configuration.
- CLI overrides (`--strategy`, `--top-k`, `--no-retrieval`) fold into the config with
  `dataclasses.replace` and are reported on stderr; they are deliberately named so a run never
  looks like it used the file's values when it did not.
- The composition still runs the deterministic double and the non-semantic fixture embedding, so the
  assembled application grounds, cites and records honestly but produces no real answer and makes no
  semantic claim.

## Consequences

- `chakaso chat` uses the actual retrieval-aware runtime, closing the gap ADR-0025 knowingly left.
- A real model or a real embedding is a configuration change at this one composition point; the CLI
  and the runtime do not change when one appears.
- The no-network guard extends to `chakaso.app`: it composes only offline layers and touches the
  filesystem solely through the explicit `--file` paths a caller supplies.
- Follow-up capabilities (a real semantic evaluator, an automatic correction trigger) plug into the
  same `build_application` seam rather than introducing a second framework.
