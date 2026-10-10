# Active task

## Current unit: P11 — dense retrieval boundary and an exact dense index

**Goal.** Introduce a dense retrieval *architecture* behind the existing `Retriever` boundary.
Define an `EmbeddingModel` protocol (`embed(texts) -> vectors`, plus dimensionality, model identity
and normalization information), and a **deterministic fixture provider** that stands in for a real
embedding model — declared, in its own name and metadata, as a development double, never as a
"semantic" embedding. Implement an exact, in-memory dense index (cosine or dot similarity,
deterministic tie-breaking, empty-index and duplicate handling) and a `DenseRetriever` that returns
the same `RetrievalResult` shape as the lexical retriever, with the retrieval strategy that produced
each result identified.

**Why this is next.** The deterministic query planner can now name `dense` and `hybrid` but refuses to
emit them, because no retriever backs them (ADR-0022). Building the dense boundary and an exact index
is what makes those modes honest to select. The first milestone is the *abstraction plus a CPU-only,
dependency-free implementation* — not a downloaded embedding model.

## What the previous unit completed (P10 — deterministic query planner)

Implemented, tested, documented and green on Python 3.11–3.13. It is a typed decision layer, not a
model, and it is not yet wired into the conversation runtime.

- [x] `chakaso.planning`: a `QueryPlanner` protocol with one implementation,
      `DeterministicQueryPlanner` (ADR-0022). A future planner replaces it behind the same signature.
- [x] A frozen, inspectable `QueryPlan`: mode, `retrieval_required`, original + normalized query,
      prior-source constraints, `top_k`, a reuse flag, `context_used`, `explanation`, and a
      `planner_version`. Construction refuses a plan that would misreport itself.
- [x] Meaning-preserving `normalize_query` (whitespace, Unicode NFC, trailing terminal punctuation;
      no word/case/interior-punctuation change), storing both original and normalized forms.
- [x] Documented, testable rules: empty → no retrieval; follow-up (conversation prior sources + topic
      overlap or anaphoric cue) → lexical retrieval constrained to prior sources; information cue or
      `?` → fresh lexical retrieval; otherwise → none. Retrieval is genuinely optional.
- [x] `QueryMode` names `none`/`lexical`/`dense`/`hybrid`; the planner refuses `dense`/`hybrid` because
      they are not backed yet. The no-network import guard covers the package.
- [x] `CURRENT_STATE`, `architecture` (data-flow steps 2 and 7 and the Query Planner row),
      `CHANGELOG` and the decisions index synced.

## What this unit is not

- Not a real semantic embedding. The dense provider is a deterministic fixture / development double;
  a hash or trivial vector must never be described as a semantic embedding, and dense results from it
  support no claim about understanding.
- Not hybrid fusion. Combining lexical and dense is the next unit (P-hybrid); this unit only makes a
  dense retriever exist behind the `Retriever` protocol.
- Not heavyweight dependencies. No FAISS, no model download, no GPU. The index is an exact CPU
  implementation in the standard library; an embedding model stays optional and behind the boundary.
- Not the retrieval-aware runtime. Wiring the planner and retrievers into a turn is P12.

## Ordering after this unit

1. Hybrid retrieval: a transparent, deterministic fusion (reciprocal rank fusion as the starting
   point) over lexical and dense, preserving each component's rank and score as provenance, with a
   typed weight/candidate-k/top-k configuration (P13).
2. A retrieval-aware conversation runtime routing a turn through planner → retrieval → model →
   AnswerRecord → durable store, retrieval optional and turn atomicity preserved (P14).
3. A real semantic judge behind `SemanticJudge`, then a triggered follow-up that decides *when* to
   reassess.
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload; every benchmark here is synthetic. A
  fixture dense provider makes the *shape* real, not the retrieval quality.
- K-007 (closed by ADR-0021): answers persist across a restart; the runtime does not yet select the
  durable store by itself.
- K-008: the fetcher's destination screen is partial; fetching stays opt-in and offline by default.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the metric
  (`unresolved_reference_count`) exists but no run feeds it (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); the planner uses an
  anaphoric-cue heuristic and conversation state, not a discourse resolver.
