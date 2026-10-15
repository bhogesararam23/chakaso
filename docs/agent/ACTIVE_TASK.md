# Active task

## Current unit: P17 — a credible semantic-evaluation boundary

**Goal.** Turn the fixture-only `SemanticJudge` into an honest path toward semantic evaluation
without claiming it already exists: sharpen the semantic-evaluation contract so it preserves the
claim, the relevant evidence references, the judgement, the evaluator identity/version, a structured
reason, the supported/contradicted/unresolved distinction, and an explicit indeterminate outcome; keep
structural grounding and semantic evaluation as separate dimensions that never overwrite each other;
deliver a versioned, clearly-labelled synthetic semantic-evaluation dataset; and connect a semantic
judgement to correction decisions only through explicit composition. If a genuinely local, CPU-feasible
semantic evaluator can be built and tested honestly, add it behind the boundary with optional
dependencies, explicit identity, no implicit downloads and clear failure when weights/hardware are
missing; if it cannot, deliver the tested interface, capability detection, configuration and
integration tests and document precisely what is missing. The fixture judge must remain available for
deterministic tests and must never be relabelled as a real judge.

**Why this is next.** P16 connected the tested runtime to an actual command (`chakaso chat`) through a
typed composition root, so the system is now usable end to end offline — but every turn is still
grounded only structurally. The single largest honesty gap in the project (K-001 aside) is that nothing
judges whether a cited source actually *bears on* a claim: a citation resolves to structure, not
meaning. The next real unit is therefore the semantic-evaluation boundary, which is what a real
retrieval-grounded answer needs before its grounding claims mean anything.

## What the previous unit completed (P16 — connect `chakaso chat` to the retrieval-aware runtime)

Implemented, tested, documented and green on Python 3.11–3.13. It wires what already existed; it
invents no capability and produces no real answer.

- [x] `chakaso.app` — a thin, injectable application composition root: resolved `Config` → model
      (registry), the *configured* answer store (never a silent downgrade), retrieval services over a
      corpus, the planner at the configured strategy/limits, and the manager (ADR-0026).
- [x] A typed `[retrieval]` configuration section (`enabled`, `strategy`, `top_k`, `candidate_k`,
      `rrf_k`, `max_ingest_bytes`) validated by the existing specs/dataclass-parity mechanism.
- [x] `chakaso chat` now drives the runtime over any `--file` documents and records every answer to
      the configured store, with an honest notice naming the engine, the active strategy, the corpus
      state and the persistence mode; `--no-retrieval`, `--strategy`, `--top-k` are real overrides.
- [x] Tests proving a configuration change changes runtime behaviour (store used, strategy, top-k,
      enabled/disabled), durable recording, sqlite-without-path refusal, and the offline no-network
      import guard extended to `chakaso.app`.

## What this unit is not

- Not a trained model, not a semantic claim: any evaluator that ships must declare its identity and
  limits and the fixture judge stays a fixture.
- Not an automatic rewrite of an answer from a semantic judgement, and not a merged answer-quality
  score; structural and semantic results stay separate dimensions.

## Ordering after this unit

1. A triggered follow-up that decides *when* to reassess, wired into the runtime behind explicit
   composition (not a fake autonomous loop).
2. A local CPU model adapter behind the `LanguageModel` boundary, only if a genuinely available,
   maintainable dependency exists; then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: nothing here has been run against a real workload; a semantic-evaluation dataset is synthetic
  and its numbers describe structure/mechanism, not general language understanding.
- K-002: `docs/architecture.md` still names Planned components; the semantic-boundary unit must keep
  its status labels honest (an interface is not an implemented capability).

## What previous units knowingly left undone

- The runtime is wired to `chakaso chat` (P16), but no reassessment is triggered automatically; the
  correction path is still an explicit application operation over a stored answer.
- Nothing records across turns how often a model invents an evidence reference (K-006), and the
  planner uses an anaphoric-cue heuristic, not a discourse resolver.
