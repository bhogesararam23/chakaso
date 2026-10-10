# Active task

## Current unit: P10 — deterministic query planner

**Goal.** Build the first decision layer that answers, for a turn, *what retrieval operation should
be performed* — a typed `QueryPlan` with a mode (`none` / `lexical` / `dense` / `hybrid`), a
normalized query, source constraints, top-k and a planner version, produced by **deterministic
rules**, not a model. It must tell a follow-up from a new topic by reusing existing conversation
state, expose an inspectable explanation of its actual decisions, and leave a clean extension point
for a future intelligent planner. It is not an LLM and must not claim semantic reasoning.

**Why this is next.** The durable-storage phase is complete: answer history persists to SQLite and is
configurable and inspectable (ADR-0021, P8/P9). Retrieval exists but is only ever invoked explicitly
by a caller. The runtime the project is moving toward — planner → retrieval → model → AnswerRecord →
durable store — needs a planner before retrieval can be chosen per turn, and retrieval must stay
optional: a greeting should not trigger it, a knowledge question should.

## What the previous unit completed (P9 — typed storage configuration and minimal storage CLI)

Implemented, tested, documented and green on Python 3.11–3.13. It extends the existing typed config
(ADR-0006); it is not a second configuration mechanism.

- [x] A `storage` section (`config.storage`) in the typed schema: backend (`memory` default |
      `sqlite`), database path and SQLite journal mode, each validated by the same spec mechanism,
      with a section-deriving parity test so a field cannot be added to a dataclass without a spec.
- [x] Sensible local defaults that write nothing: the default backend is in-memory and the default
      path is empty, so the built-in configuration never touches the filesystem.
- [x] `chakaso storage check` (read-only schema/integrity report) and `chakaso storage migrate`
      (create or confirm a schema), wired through the store's own `check()`/`migrate()` — the CLI adds
      no storage logic of its own.
- [x] Storage tests use isolated temporary databases and assert a `check` creates nothing; the durable
      backend refuses a database it cannot safely interpret rather than repairing it.
- [x] `CURRENT_STATE` (CLI + configuration rows, what-works, tested-coverage), `getting-started`
      command list and usage section, and `CHANGELOG` synced.

## What this unit is not

- Not an LLM planner and not semantic reasoning. The planner is a typed rule system with a version;
  every decision it reports is a real rule outcome, never fabricated reasoning.
- Not dense or hybrid retrieval. The plan may *name* those modes, but they are built in P11 behind an
  embedding boundary; there is no real embedding model, and a fixture double must not be called semantic.
- Not the retrieval-aware conversation runtime. Choosing retrieval per turn and wiring it into
  `ConversationManager` is P12; this unit produces and tests the planner as a standalone layer.
- Not a natural-language discourse resolver. Follow-up vs new-topic uses conversation state and an
  explicit extension point, without pretending pronoun/ellipsis resolution exists.

## Ordering after this unit

1. Dense and hybrid retrieval behind an embedding boundary, with a deterministic development double —
   not a real semantic embedding; each result labelled with its strategy (P11).
2. A retrieval-aware conversation runtime routing a turn through planner → retrieval → model →
   AnswerRecord → durable store, retrieval optional and turn atomicity preserved (P12).
3. A real semantic judge behind `SemanticJudge`, validated against the fixture relations the correction
   benchmark already labels; then a triggered follow-up that decides *when* to reassess.
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload; every benchmark here is synthetic.
- K-007 (closed by ADR-0021): answers persist across a restart; the runtime does not yet select the
  durable store by itself, and a `CorrectionRecord` / `ExperimentResult` is still not auto-persisted.
- K-008: the fetcher's destination screen is partial; fetching stays opt-in and offline by default.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the metric
  (`unresolved_reference_count`) exists but no run feeds it (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); follow-ups carry the
  previous turns and nothing more specific — which is exactly the extension point P10 formalizes.
