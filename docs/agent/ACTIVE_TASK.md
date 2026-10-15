# Active task

## Current unit: P18 — evidence and source usability, with adversarial security tests

**Goal.** Make the retrieval workflow practical end to end from the CLI — explicit local ingestion,
inspection of indexed sources, retrieval over a selected subset, evidence-pack inspection, and clear
handling of empty corpora and no-match results — without pretending durability the in-memory corpus
does not have, and without a second source store competing with the answer store. Harden the surface
against hostile content: retrieved text stays data, never an instruction; ingestion escapes no
explicit directory and respects size/format limits; add adversarial prompt-injection tests over the
new command surface. Keep web fetching opt-in and off every default path.

**Why this is next.** The runtime is now wired to `chakaso chat` and the semantic-evaluation boundary
is hardened and measurable (P16/P17). The next usability gap is the *input* side: making it simple to
curate, inspect and trust the evidence a grounded answer rests on, and proving the new surface resists
untrusted content.

## What the previous unit completed (P17 — a credible semantic-evaluation boundary)

Implemented, tested, documented and green on Python 3.11–3.13. It hardened the boundary and built the
measurement; it shipped no real evaluator, because none is dependency-feasible offline (§2.1, ADR-0001).

- [x] `SemanticEvaluation` and the grounding result carry evaluator **name + version**, so a change of
      judge is visible in provenance; `describe_semantic_capability()` reports honestly that no real
      semantic evaluator exists and the only judge is `FixtureSemanticJudge`.
- [x] A dedicated `SemanticJudgeError` is recorded per claim as *not evaluated* — never support — and one
      failing claim does not abort the answer; structural grounding stays usable with no evaluator wired.
- [x] A versioned, hand-labelled synthetic semantic-evaluation dataset and a deterministic scorer
      (accuracy, per-class precision/recall, confusion, unresolved + error handling) as a development
      instrument — its fixture score is labelled structure-replay, not capability — via `chakaso
      benchmark semantic` (ADR-0027).
- [x] Correction-integrity tests: an evaluator exception is never a support judgment; an unresolved
      judgment is never a confirmed contradiction; structural grounding works alone.

**What P17 did not do (and will not pretend to):** it did not add a real semantic evaluator. A genuine
local, CPU-feasible, dependency-free one remains an explicit, documented gap; it plugs into the same
`SemanticJudge` boundary later as an optional adapter.

## What this unit is not

- Not a durable source index or a second database — the corpus stays in-memory unless a real need is
  shown; do not invent persistence to look complete.
- Not a web-frontend, crawler, or a URL chooser; the fetcher stays opt-in and off every default path.
- Not a real semantic evaluator or model (P17's boundary stands).

## Ordering after this unit

1. A triggered follow-up that decides *when* to reassess, wired into the runtime behind explicit
   composition (not a fake autonomous loop).
2. A local CPU model adapter behind the `LanguageModel` boundary, only if a genuinely available,
   maintainable dependency exists; then the tokenizer, dataset pipeline and a tiny Transformer.

## Known gaps that bound this unit

- K-001: offline and synthetic; a real workload has still not been run.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- No automatic, triggered correction; the correction path is still an explicit application operation.
- Nothing records across turns how often a model invents an evidence reference (K-006); the planner
  uses an anaphoric-cue heuristic, not a discourse resolver.
