# Active task

## Current unit: P8 — durable answer persistence

**Goal.** Give the answer history a backend that survives a process. `chakaso.answers` already
defines the `AnswerStore` boundary and an in-memory, append-only implementation; the next unit
implements persistence behind that boundary (a deterministic file/JSONL store, or a simple
embedded one) and a durable `AnswerRecord` lineage, so a correction chain can be written, closed and
replayed after a restart — with the same integrity rules enforced on load that the store enforces on
save. No model, and no revised prose.

**Why this is next.** Everything the correction loop decides is now recorded and measurable in
memory, but nothing survives a run, so a transcript cannot be audited later and an experiment cannot
be re-derived from stored answers. Persistence is the dependency K-007 has been pointing at, and the
`AnswerStore` boundary already makes it an addition rather than a redesign (ADR-0018). No storage
schema is chosen before a real need appears; the smallest deterministic file format that round-trips
records is enough to start.

## What the previous unit completed (P7 — measurement, wiring and the reproducibility layer)

Implemented, tested, documented and green on Python 3.11–3.13. None of it is a claim about answer
quality or a measured model-quality result; every number is a development measurement over synthetic
fixtures.

- [x] Persistent answer records and store (`chakaso.answers`): a per-event `AnswerId` (ADR-0017), an
      immutable `AnswerRecord` referencing turn/model/evidence/claims/evaluation/`correction_of`, and
      an in-memory append-only `AnswerStore` that refuses overwrites and broken correction links
      (ADR-0018). **In-memory only — durable persistence is the next unit.**
- [x] Conversation integration: `ConversationManager` optionally records and saves an `AnswerRecord`
      before adopting a turn, so a persistence failure aborts the turn (ADR-0008 extended).
- [x] Application-level reassessment (`chakaso.correction`): `reassess_stored_answer` /
      `reassess_and_record` over a stored answer by identifier — a deterministic, id-addressed
      follow-up boundary; no natural-language reference resolver is pretended.
- [x] Semantic grounding boundary (`chakaso.grounding`, ADR-0019): `SemanticJudge` /
      `SemanticGroundingEvaluator` with only a `FixtureSemanticJudge`; composes into answer evaluation
      and reassess with no signature change. No real semantic capability is claimed.
- [x] Correction development benchmark (`chakaso.benchmark.correction`): versioned cases covering
      every reachable decision, a runner over `reassess`, per-case and aggregate metrics,
      deterministic text/JSON reports, and a pinned-fingerprint regression.
- [x] Reproducible experiments (`chakaso.experiments`, ADR-0020): `Experiment` and a
      content-fingerprinted `ExperimentResult` (environment kept separate), `run_experiment`, and a
      deterministic `compare_results` regression baseline that refuses incompatible versions.
- [x] CLI: `chakaso benchmark retrieval|correction` and `chakaso experiment run <subject>`;
      deterministic and offline.
- [x] Security/integrity regressions extended to the new layers (no-network import scan, append-only
      history, correction-chain integrity, experiment reproducibility), with deliberate-corruption
      tests proving the guards can fail.

## What this unit is not

- Not a language model, tokenizer or training. The only model is the deterministic double; nothing
  generates an answer or revised prose.
- Not a real semantic judge. `contradicted` / `uncertain` still come only from a supplied fixture or
  manual judgement (ADR-0019); the boundary is ready for a future one.
- Not an automatic correction trigger. Reassessment is invoked by a caller, not decided by the
  system, and a follow-up is addressed by answer id rather than resolved from natural language.
- Not a durable-storage service. A file or embedded store, chosen minimally, not a database server.
- Not a dense retriever, query planner or live crawl.

## Ordering after this unit

1. A real semantic judge behind `SemanticJudge`, validated against the fixture relations the
   correction benchmark already labels.
2. A triggered follow-up: decide *when* a turn warrants reassessment, on the id-addressed boundary.
3. Query planning: when to retrieve, and which sources (P4).
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer (P7–P8).

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload; every benchmark here is synthetic.
- K-007: answer records are in-memory; nothing persists across a process until this unit lands.
- K-008: the fetcher's destination screen is partial; fetching stays opt-in and offline by default.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the metric
  (`unresolved_reference_count`) exists but no run feeds it (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); follow-ups carry the
  previous turns and nothing more specific.
