# Active task

## Current unit: P12 — hybrid retrieval (rank fusion)

**Goal.** Build a hybrid retriever that composes the two retrievers that now exist — lexical
BM25 and dense similarity — into one ranking through a transparent, deterministic fusion,
Reciprocal Rank Fusion as the starting method. It implements the existing `Retriever` protocol,
returns a single `RetrievalResult` sequence tagged `hybrid`, and preserves per-component
provenance: for each fused result, the lexical rank and score (when present), the dense rank and
score (when present), the final fused score and the final rank. Fusion configuration (method,
candidate pool size, final top-k, any component weighting) uses the existing typed config system,
not a second mechanism. Once it exists, the query planner's `hybrid` mode becomes backed and
selectable.

**Why this is next.** The dense retriever now backs `dense` (ADR-0023); `hybrid` is the only
named-but-refused mode left. Fusing lexical and dense is the standard way to combine their
complementary strengths, and building it now — over an exact lexical and exact dense retriever —
keeps the whole retrieval stack deterministic and testable before anything wires it into a turn.

## What the previous unit completed (P11 — dense retrieval boundary and exact index)

Implemented, tested, documented and green on Python 3.11–3.13. It makes the dense *shape* real, not
dense *quality*.

- [x] `chakaso.retrieval.embeddings`: an `EmbeddingModel` protocol (`metadata` + batch `embed`) with
      `EmbeddingVector` (fixed-length, finite) and `EmbeddingMetadata` carrying an explicit
      `is_semantic` flag. The only implementation, `FixtureEmbeddingModel`, is a deterministic hashed
      bag-of-tokens double declared `is_semantic=False` / `development_double=True` — never called
      semantic (ADR-0023).
- [x] `chakaso.retrieval.dense`: an exact in-memory `DenseIndex` (uniform dimension, no duplicate
      chunk) and a `DenseRetriever`, a second `Retriever` implementation ranking by cosine/dot
      similarity. No approximate index and no ML dependency; both are deferred until measured need.
- [x] `RetrievalResult` now carries a required `strategy` (`lexical`/`dense`/`hybrid`); a dense result
      reports empty `matched_terms` rather than inventing term overlaps.
- [x] Because a dense retriever exists, `DeterministicQueryPlanner` now accepts `dense` and still
      refuses `hybrid` until this unit builds the fusion layer.
- [x] `CURRENT_STATE`, `architecture`, `retrieval` (status table and a corrected stale sentence),
      `CHANGELOG`, the decisions index and ADR-0023 synced.

## What this unit is not

- Not learned or semantic fusion. The fusion is a written, deterministic formula (RRF or a documented
  weighted variant), independently testable and configurable — no model ranks the merge.
- Not a real embedding model. Dense candidates come from the same non-semantic fixture double; a
  hybrid benchmark run over it is labelled a synthetic development instrument and may not be reported
  as dense/hybrid superiority on meaning.
- Not the retrieval-aware runtime. Wiring planner → retrievers into a conversation turn is the next
  unit; this one only makes a `hybrid` result exist behind the `Retriever` protocol.

## Ordering after this unit

1. A retrieval-aware conversation runtime routing a turn through planner → retrieval → model → claims →
   citation validation → grounding → `AnswerRecord` → durable store, retrieval optional and turn
   atomicity preserved; the planner's decision and the retrieval strategy become structured,
   inspectable metadata on the answer (P13).
2. Benchmark and experiment expansion comparing lexical / dense / hybrid over the synthetic fixtures,
   labelled honestly as fixture/synthetic, using the existing experiment framework for regression
   comparison — only measured results, never invented ones (P14).
3. A real semantic judge behind `SemanticJudge`, then a triggered follow-up that decides *when* to
   reassess.
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload; every benchmark here is synthetic, and
  dense/hybrid over a fixture embedding proves the mechanism, not retrieval quality.
- K-007 (closed by ADR-0021): answers persist across a restart; the runtime does not yet select the
  durable store by itself.
- K-008: the fetcher's destination screen is partial; fetching stays opt-in and offline by default.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the metric
  (`unresolved_reference_count`) exists but no run feeds it (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); the planner uses an
  anaphoric-cue heuristic and conversation state, not a discourse resolver.
