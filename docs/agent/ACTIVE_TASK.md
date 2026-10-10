# Active task

## Current unit: P13 — retrieval-aware conversation runtime

**Goal.** Compose the layers that now exist into a real per-turn runtime, without a giant
orchestrator: for a user turn, the **planner** decides whether and how to retrieve; the chosen
**retriever** (lexical, dense or hybrid) produces an `EvidencePack`; the pack is handed to the
**model** through a structured, citation-safe context; the answer is decomposed into **claims**, its
**citations validated**, its **grounding evaluated**, recorded as an `AnswerRecord`, and saved to the
**durable store** when one is selected. Retrieval must stay genuinely optional (a greeting triggers
none), the existing validators and boundaries must be reused rather than duplicated, and the turn's
all-or-nothing atomicity (ADR-0008) must hold across every new stage. The planner's decision and the
retrieval strategy become structured, inspectable metadata on the answer.

**Why this is next.** Every component the runtime needs is now built and independently tested — the
deterministic planner (ADR-0022), three `Retriever`s (ADR-0023/ADR-0024), the model boundary,
claims, citation validation, grounding, `AnswerRecord`, and a durable `AnswerStore` (ADR-0021). What
is missing is the seam that runs the planner and feeds its chosen evidence into `send`, so retrieval
stops being caller-invoked and becomes a turn behaviour. This is the run's major runtime milestone.

## What the previous unit completed (P12 — hybrid retrieval, rank fusion)

Implemented, tested, documented and green on Python 3.11–3.13. It backs the planner's last mode.

- [x] `chakaso.retrieval.hybrid`: a `HybridRetriever`, a third `Retriever`, fusing the lexical and
      dense retrievers with transparent, deterministic reciprocal rank fusion; the formula is written
      out, not hidden (ADR-0024).
- [x] A `HybridConfig` (method, `rrf_k`, `candidate_k`, lexical/dense weights) validated at
      construction — typed tuning, not a second configuration system.
- [x] `RetrievalResult` gained an optional `provenance`; a hybrid result carries `HybridProvenance`
      (each component's rank/score, or `None`, plus the fused score), keeping a fused hit auditable.
- [x] The query planner now accepts `hybrid`; lexical, dense and hybrid are all backed modes.
- [x] `CURRENT_STATE`, `architecture`, `retrieval` status table, `CHANGELOG`, the decisions index and
      ADR-0024 synced.

## What this unit is not

- Not a language model. Generation still goes through the model boundary; its only adapter is the
  deterministic development double, so the runtime produces the double's fixed text, not a real answer.
- Not semantic retrieval or semantic grounding claims. Dense/hybrid run over the non-semantic fixture
  embedding, and grounding stays structural (a fixture semantic judge at most); nothing here may be
  described as understanding meaning.
- Not an automatic correction loop. This unit wires retrieval into a turn; deciding *when* to reassess
  and generating revised prose remain later work.
- Not persisted conversation state. Answer records persist (ADR-0021); the conversation object itself
  still lives in memory and is not the thing durable storage covers.

## Ordering after this unit

1. Benchmark and experiment expansion comparing lexical / dense / hybrid and top-k settings through
   the existing experiment framework, reporting only measured, honestly-labelled development numbers.
2. An end-to-end + restart integration demonstration, deliberate "test-the-tests" corruption checks,
   and a security review of the new runtime (untrusted retrieved content, prompt injection, citations).
3. A real semantic judge behind `SemanticJudge`, then a triggered follow-up that decides *when* to
   reassess.
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload; the runtime is exercised only by offline,
  deterministic tests over synthetic fixtures and the development double.
- K-007 (closed by ADR-0021): answers persist across a restart; this unit is where the runtime can
  first select the durable store, but that selection stays opt-in and the default backend is in-memory.
- K-008: the fetcher's destination screen is partial; the runtime must not fetch on any default path.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the runtime makes the
  condition detectable per turn, but no store yet aggregates the rate across a session (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); the planner uses an
  anaphoric-cue heuristic and conversation state, not a discourse resolver.
