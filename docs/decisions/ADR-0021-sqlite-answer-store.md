# ADR-0021: Durable answer storage is a SQLite backend behind the existing store boundary

- Status: Accepted
- Date: 2026-10-10
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0008](ADR-0008-transactional-turns.md), [ADR-0017](ADR-0017-answer-identity.md), [ADR-0018](ADR-0018-answer-persistence.md)

## Context

ADR-0018 deliberately left answer persistence in memory, on the reasoning that freezing a storage
schema before a workload had exercised it would be a guess. It made `AnswerStore` a protocol so that
a backend could arrive later as an *implementation*, not a redesign. That day is the durable-storage
phase: the correction history, reassessment and experiment records now produce real `AnswerRecord`
objects, and a research transcript that vanishes at process exit cannot be audited or replayed
across runs — the gap K-007 named. The abstraction is exercised for the first time, which is the
event ADR-0018 said would justify a backend.

Choosing the backend is constrained by the project's own rules. ADR-0001 forbids the runtime
requiring a hosted or commercial service; a client-server database (Postgres), an embedded NoSQL
store (MongoDB, Redis) or a cloud store would each add a dependency the local-first posture exists
to avoid. The store must stay local, deterministic enough to test, transactional, and — because
this is a research project whose history is its data — impossible to silently corrupt.

## Problem

How should the answer history be persisted so that it survives a process, keeps ADR-0018's
append-only invariants exactly, fails loudly rather than repairing corrupted research history, and
does so without introducing a third-party or hosted database dependency or a schema that becomes an
undocumented accident?

## Options considered

- **Stay in-memory.** Rejected: does not close K-007; a transcript still cannot be replayed across
  a restart, which the correction and experiment work now depends on.
- **JSON-lines file store.** Considered and rejected: it gets durability but not the atomic,
  partially-failing batch, the enforced foreign key from a correction to the answer it supersedes, or
  an indexed, queryable history without a bespoke index layer — i.e. it would reimplement, worse, the
  three things a database already guarantees, while still writing files.
- **A client-server or embedded NoSQL database.** Rejected: a runtime dependency and a service
  ADR-0001 exists to keep out.
- **SQLite via the standard library `sqlite3`.** Chosen: it ships with Python (no new dependency),
  is a single local file, enforces transactions and foreign keys, and supports the exact append-only,
  integrity-first discipline the history needs.

## Decision

`chakaso.answers.sqlite.SQLiteAnswerStore` implements the existing `AnswerStore` protocol against a
local SQLite file. The conversation manager, retrieval, model and correction layers are unchanged and
may not call SQLite directly (ADR-0018's boundary holds): the durable store is selected at the
composition point, and the default backend remains the in-memory store, so nothing writes to disk
unless a caller opts in.

**Schema and versioning.** One table, `answers`, keyed by `answer_id`, with a `correction_of` column
carrying a real foreign key to `answers(answer_id)` and indexes on `conversation_id` and
`correction_of`. The schema version is recorded in `PRAGMA user_version`, not guessed from the file's
existence. A fresh database is created and stamped version 1. A database reporting a version *newer*
than this build, or a version *older* than the registered migration path, is refused with
`StorageSchemaError` and **not opened for writing** — silently reading or mutating a schema this code
cannot fully interpret is the way research history gets destroyed. A version that matches but whose
table is missing is reported as corrupt, not quietly recreated over the top.

**Migration boundary, not a migration framework.** `_MIGRATIONS` maps an old version to its next
step. It is empty today, and that is the honest state: there is one schema version, so there is
nothing to migrate *from*. The boundary is what makes it a decision rather than an accident — a future
schema change registers a step and bumps `SCHEMA_VERSION`, and an older database is then brought
forward explicitly instead of being reset.

**Durability is a claim about a restart.** Atomicity is real transactions: every append runs inside
one `BEGIN`/`COMMIT`, and any failure in a batch triggers a `ROLLBACK` that leaves the history exactly
as it was — a correction chain is never half-written. The store's durability is proven by a test that
opens the database in a *separate process*, writes an answer, exits, and reads it back in a fresh
connection; a same-process close/reopen alone would not justify the word "durable".

**Strict serialization.** A row stores the answer's canonical JSON document (produced by the codec in
`chakaso.answers.serialize`, `sort_keys=True` for byte determinism) alongside indexed scalar columns.
The document is authoritative for reconstruction; the columns exist to index, to enforce the
correction foreign key, and to give integrity a second witness. On every read the columns are
cross-checked against the decoded document, and a divergence raises as corruption rather than being
reconciled. Decoding is strict: a malformed document is `AnswerSerializationError`, never a repaired
record.

**Evaluation is not persisted.** The codec deliberately omits the in-memory `evaluation` object. An
answer's evaluation is a view computed from its claims and evidence by a particular evaluator;
freezing it would capture a possibly-stale judgement of an ephemeral computation into permanent
history, and would imply the project has committed to an evaluation semantics that may still change.
The durable record is the answer, its claims and its references; the evaluation is recomputed against
the current evaluator when needed.

**Invariants have one source.** The overwrite, dangling-correction and cross-conversation checks are
not restated in the SQLite store; it imports the in-memory store's `_ensure_appendable` and feeds it
the two rows it needs from indexed primary-key lookups. A conformance suite runs the whole invariant
set against both backends, so the durable store cannot quietly weaken a guarantee the protocol
already made.

## Reasoning

- **SQLite is the smallest thing that is actually durable.** It is in the standard library, so the
  dependency-free foundation (pyproject declares none) stays intact; it is a single file, so it is
  local-first (ADR-0001); and it gives transactions and foreign keys for free, which is precisely the
  integrity an append-only research history needs and that a file store would have to rebuild.
- **Version the schema because the schema will change.** A research project edits its records as it
  learns; a version marker plus a refusal-when-unknown turns "the file format drifted under us" into
  "this build can't read version 3 yet," which is a comprehensible, non-destructive failure.
- **Refuse to repair.** This project's product is trustworthy history. A store that silently fixes a
  corrupt record has, in that moment, fabricated one. Every corruption path here reports.

## Trade-offs

- **A document column plus scalar columns duplicate the identity fields.** They can drift, and drift
  is treated as corruption to be reported rather than prevented by construction. The alternative —
  normalizing claims and references into their own tables — is a larger schema and more migration
  surface than one version justifies; it is deferred until the query shapes actually need it.
- **Concurrency is scoped, not solved.** The store holds one connection and is used from one thread;
  cross-process restart is tested, simultaneous multi-process *writing* is not, and no such safety is
  claimed. SQLite's file locking provides a floor, but the honesty rule is that untested guarantees
  are not advertised.
- **`initialize` defaults to true**, so opening a fresh path creates the schema; the safe-restart
  default most callers want. A read-only caller passes `initialize=False`, and an absent schema is
  then an error rather than a surprise write.

## Consequences

- K-007 is closed: an `AnswerRecord` and its `correction_of` lineage now survive the process that
  wrote them and can be replayed across a restart.
- ADR-0018 is not superseded — its boundary, its append-only invariants and its in-memory *default*
  all stand. This adds the second implementation its protocol was designed to admit.
- Durability becomes a runtime option the retrieval-aware conversation can select, so the conversation
  history a later experiment analyzes can be persisted rather than reconstructed.
- Typed configuration for the storage backend, path and journal mode is the follow-up that makes the
  choice a configuration change rather than code; it uses the existing config system, not a second one
  (ADR-0006).
