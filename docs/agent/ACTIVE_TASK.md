# Active task

## Current unit: P9 — typed storage configuration and minimal storage CLI

**Goal.** The durable `SQLiteAnswerStore` exists (ADR-0021). Bring it into the existing typed
configuration system — backend (`memory` default | `sqlite`), database path, journal mode — with
sensible local defaults, and add the minimal developer commands `chakaso storage check` and
`chakaso storage migrate`, wired through the store's own `check()`/`migrate()` rather than new
logic in the CLI. Tests must use isolated temporary databases and must never write into the
repository tree.

**Why this is next.** The store is built and proven durable, but selecting it currently requires
code. Configuration is the project's single place where a runtime choice is made (ADR-0006), so
the durable backend is not operator-usable until it is configurable and inspectable. Typed storage
settings and the `storage check` diagnostic are small, and they close out the durable-storage phase
before query planning begins.

## What the previous unit completed (P8 — durable answer persistence)

Implemented, tested, documented and green on Python 3.11–3.13. It is durable local storage, not a
persistence *service*, and it is opt-in: the in-memory store remains the default.

- [x] `chakaso.answers.sqlite.SQLiteAnswerStore`: a durable `AnswerStore` backend on the standard
      library's `sqlite3` — no new dependency, no third-party or hosted database (ADR-0001, ADR-0021).
- [x] A schema version in `PRAGMA user_version` and an explicit migration boundary (`_MIGRATIONS`, empty
      today by design): a newer or unsupported schema is refused, never mutated; a version marker with no
      table is reported as corrupt rather than recreated.
- [x] Transactional appends: every save runs in one `BEGIN`/`COMMIT`; a failure in a `save_many` batch
      rolls the whole batch back, so no partial correction chain survives. `save_many` added to the
      protocol and both backends.
- [x] Restart durability, including a test that writes an answer from a *separate process* and reads it
      back after restart — the honest proof behind the word "durable".
- [x] Strict-read integrity: a row's indexed columns are cross-checked against its serialized document;
      a malformed or divergent record raises rather than being repaired. The append-only invariants are
      shared with the in-memory store (one validator) and run against both backends in a conformance suite.
- [x] The serialization codec (`chakaso.answers.serialize`) defining exactly what is persisted; an
      answer's `evaluation` is deliberately not persisted — it is recomputed, not frozen into history.
- [x] `StorageSchemaError`; ADR-0021 and the decisions index; `CURRENT_STATE`, `KNOWN_ISSUES` (K-007
      closed), `evaluation`/`transparency`/`getting-started` status labels and `CHANGELOG` all synced.

## What this unit is not

- Not a language model, tokenizer or training. The only model is the deterministic double; nothing
  generates an answer or revised prose.
- Not a real semantic judge. `contradicted` / `uncertain` still come only from a supplied fixture or
  manual judgement (ADR-0019); the boundary is ready for a future one.
- Not runtime-wired persistence or an automatic correction loop. The durable store exists and is
  selectable by a caller; the conversation runtime does not yet reach for it, and no reassessment
  triggers itself.
- Not multi-process write safety. The store is single-thread; cross-process restart is tested, and no
  broader concurrency guarantee is claimed.
- Not a second configuration mechanism. Storage settings extend the existing typed schema (ADR-0006).

## Ordering after this unit

1. Query planning: a deterministic, typed decision of whether and how to retrieve for a turn (P10).
2. Dense and hybrid retrieval behind an embedding boundary, with a deterministic development double —
   not a real semantic embedding (P11).
3. A retrieval-aware conversation runtime routing a turn through planner → retrieval → model →
   AnswerRecord → durable store, retrieval optional and turn atomicity preserved (P12).
4. A real semantic judge behind `SemanticJudge`, validated against the fixture relations the correction
   benchmark already labels; then a triggered follow-up that decides *when* to reassess.
5. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload; every benchmark here is synthetic.
- K-007 (closed by ADR-0021): answer records can now persist durably across a process restart; the
  remaining gap is that the runtime does not select the durable store by itself and a
  `CorrectionRecord` / `ExperimentResult` is still not auto-persisted.
- K-008: the fetcher's destination screen is partial; fetching stays opt-in and offline by default.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the metric
  (`unresolved_reference_count`) exists but no run feeds it (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); follow-ups carry the
  previous turns and nothing more specific.
