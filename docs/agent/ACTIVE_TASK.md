# Active task

## Current unit: P21 — a triggered, explicitly-composed reassessment (no fake autonomous loop)

**Goal.** Let a follow-up that names a stored answer trigger reassessment through the existing
identifier-addressed boundary (`reassess_stored_answer` / `reassess_and_record`), wired into the
runtime behind explicit composition rather than an autonomous loop that guesses when to revise. The
trigger must be deterministic and inspectable (an explicit `AnswerId` reference, or the planner's
cue), never a model-like inference that does not exist. Reassessment remains a separate append-only
event; the original answer and its historical provenance are never rewritten (ADR-0028). No revised
prose is generated — there is no model.

**Why this is next.** P19/P20 made a completed turn's judgement durable, auditable and honestly
represented. The remaining genuine engineering gap on the correction side is composition: nothing
yet *initiates* a reassessment from a conversation turn. It is delivered behind explicit wiring, not
a fabricated autonomous decision, and stays fully offline and deterministic.

## What the previous unit completed (P20 — provenance-reference integrity in answer inspection)

Implemented, tested, documented and green on Python 3.11–3.13.

- [x] `chakaso answers show` now discloses — in the human view and as `evidence_retention` in
      `--json` — that an answer's evidence/citation references are identifiers only: the durable
      store keeps what the turn decided and which evidence it named, not the evidence text, which
      lived in the ephemeral in-memory corpus and is not re-readable from the store.
- [x] No evidence bodies were copied into the answer store (an immutable snapshot was weighed against
      size and privacy and judged unjustified without a consumer); the unit is a truthful disclosure
      of an existing boundary, not new surface area.

**What P19–P20 did not do (and will not pretend to):** they compute no aggregate invented-reference
rate (K-006 advanced, not closed — no generating model, no real workload, K-001); they add no
evaluator, model or dependency; the development double gains no fabricated capability.

## What this unit is not

- Not an autonomous correction loop that guesses when to revise; the trigger is deterministic and
  inspectable, wired behind explicit composition.
- Not generated revised prose — there is no language model.
- Not a real semantic evaluator (P17's boundary stands) or a durable evidence snapshot (P20 said so).

## Ordering after this unit

1. A local CPU model adapter behind the `LanguageModel` boundary, only if a genuinely available,
   maintainable, offline dependency exists — a consequential dependency decision that needs an ADR
   first (ADR-0001), then the tokenizer, dataset pipeline and a tiny Transformer.
2. A real `SemanticJudge` behind the existing boundary, measured on the synthetic semantic dataset;
   likewise gated on a dependency-feasibility decision recorded before any code.

## Known gaps that bound this unit

- K-001: offline and synthetic; a real workload and any real model have still not been run — the
  frontier of the whole project, and the reason P21 stays deterministic and composed rather than
  autonomous.
- K-006: the invented-reference condition is persisted per answer, but nothing aggregates it into a
  rate, and it is only durable when the SQLite backend is selected.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- No automatic, triggered correction; the correction path is still an explicit application operation
  (P21 addresses composition, not autonomy).
- The planner uses an anaphoric-cue heuristic, not a discourse resolver.
