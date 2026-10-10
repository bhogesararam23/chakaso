# Active task

## Current unit: P14 — benchmark and experiment expansion over the three retrieval strategies

**Goal.** Extend the existing retrieval benchmark and experiment framework so it can score and
compare the retrievers that now exist — lexical, dense and hybrid — and over a small top-k matrix,
turning the runtime's capabilities into *measured* development numbers. Reuse the versioned case
schema, the `run_experiment` / `compare_results` regression baseline, and the pinned-fingerprint
discipline; add the missing runner wiring (score retrieval with an injected retriever/strategy) and
the experiment definitions. Every number must come from an actual run over the synthetic fixtures and
be labelled a development instrument; nothing may be presented as a real-world or semantic-quality
result.

**Why this is next.** The pipeline now runs end to end in the library (ADR-0025), but no measured
evidence yet compares the strategies or shows whether fusion or top-k changes anything on the
fixtures. The experiment framework exists specifically to answer "did this change help, measurably?"
— and to do it honestly, including the obligation to report when a fixture dense/hybrid result shows
*no* advantage. This is the research payload the earlier infrastructure was built to support.

## What the previous unit completed (P13 — retrieval-aware conversation runtime)

Implemented, tested, documented and green on Python 3.11–3.13. It composes the pipeline; it does not
generate a real answer.

- [x] `chakaso.runtime.RetrievalAwareConversation`: planner → chosen retrieval → model → record →
      store as a thin coordinator, retrieval optional, turn atomicity preserved (ADR-0025).
- [x] `build_retrieval_services(corpus)` wiring one service per mode, each labelled with its actual
      strategy; `render_evidence_context(pack)` as the structured, citation-safe model-facing boundary.
- [x] The planner decision and retrieval strategy recorded on each answer; `ConversationManager.send`
      gained one optional `provenance` parameter for exactly this.
- [x] Runtime tests: the conversational sequence, dense/hybrid flow-through, construction guards, a
      retriever-failure leaving state untouched, and the evidence-context renderer.
- [x] `CURRENT_STATE`, `architecture`, `CHANGELOG`, the decisions index and ADR-0025 synced; the
      no-network guard extended to `chakaso.runtime`.

## What this unit is not

- Not a real model, embedding or benchmark. The dense/hybrid components are the non-semantic fixture
  double; any comparison is over small synthetic fixtures and checks structure and recall of the
  fixture corpus, never meaning or real-world quality.
- Not inventing numbers. If hybrid does not beat lexical on the fixtures, that is the reported result;
  no favourable figure is manufactured, and no conclusion is written before a run produces it.
- Not the end-to-end / restart integration demonstration or the security review — those are the next
  unit (P15).

## Ordering after this unit

1. An end-to-end and restart integration demonstration over the runtime, deliberate "test-the-tests"
   corruption checks on the critical invariants, and a security review of the new runtime (untrusted
   retrieved content, prompt injection, citations, persisted records).
2. Wiring the runtime into `chakaso chat` / the CLI behind typed retrieval configuration.
3. A real semantic judge behind `SemanticJudge`, then a triggered follow-up that decides *when* to
   reassess.
4. A local CPU model adapter, then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: nothing has been measured against a real workload; this unit produces the project's first
  *retrieval-strategy* comparisons, and they are development measurements over synthetic fixtures.
- K-007 (closed by ADR-0021): answers persist; the runtime can select the durable store but that is
  opt-in and off the default path.
- K-008: the fetcher's destination screen is partial; fetching stays opt-in and offline by default.

## What previous units knowingly left undone

- Nothing records across turns how often a model invents an evidence reference; the runtime makes the
  condition detectable per turn, but no store yet aggregates the rate across a session (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); the planner uses an
  anaphoric-cue heuristic and conversation state, not a discourse resolver.
