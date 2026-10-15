# Active task

## Current unit: P22 — a real-model feasibility assessment (no fake local model)

**Goal.** Decide, on evidence rather than aspiration, what it would actually take to put a *real*
language behaviour behind Chakaso's existing boundaries. Assess two viable directions against the
available environment — (A) a genuinely runnable, small local pretrained model integrated behind the
`LanguageModel` boundary, and (B) a minimal from-scratch training experiment that demonstrates the
learning loop (data, tokenizer strategy, model definition, training, checkpointing, held-out
evaluation). Compare them on hardware/RAM, download and storage size, CPU feasibility, dependency and
licence constraints, offline reproducibility, scientific usefulness, fit with the model boundary, and
maintenance. Record a recommendation in a research note that separates verified facts from assumptions
and estimates.

**Why this is next.** P21 closed the last deterministic gap on the correction side: a follow-up that
explicitly names a stored answer now composes into a reassessment through the runtime. Everything left
on the roadmap that would make answers *mean* something — a real embedding, a real judge, a real model
— is gated on the same unresolved dependency question (K-001, and the ordering note under the previous
units). A feasibility assessment is the honest, dependency-free unit that turns "someday a local model"
into a documented, evidence-based decision before any code or weight is adopted.

**What this unit is not.** Not an implementation of a model, an adapter, a tokenizer or a training loop.
Not a downloaded weight set or a new heavy dependency — those need their own ADR first (ADR-0001).
Not a claim of any measured capability: if nothing is run and measured, nothing is written as a result.

## What the previous unit completed (P21 — a triggered, explicitly-composed reassessment)

Implemented, tested, documented and green on Python 3.11–3.13.

- [x] A deterministic, inspectable **reassessment trigger** (`chakaso.correction.trigger`) reads a
      follow-up for an explicit, validated `ans_…` reference. It detects intent only — it reads no store,
      runs no evaluator, retrieves nothing and mutates nothing, so intent detection is independently
      testable from execution. A vague reference is never inferred; two distinct identifiers are
      *ambiguous* and refused, not guessed.
- [x] The trigger is **composed at the application root** (`Application.reassess_followup`) with the
      existing `reassess_and_record` boundary: a named, recorded answer is re-grounded against its
      standing evidence (the corpus chunks still present for the sources it named) merged with whatever
      the follow-up retrieves through a new read-only `gather_evidence` seam on the runtime. Unknown and
      ambiguous targets raise the boundary's typed errors (`UnknownAnswerError`,
      `AmbiguousReassessmentTargetError`); a follow-up that names nothing is an ordinary turn.
- [x] **Reachable from the supported runtime interface**: `chakaso chat` routes an answer-naming
      follow-up into reassessment and reports the decision honestly, and surfaces each recorded answer's
      identifier so a later turn can name it. The original `AnswerRecord` and its historical provenance
      are never rewritten; a reassessment is a distinct append-only `CorrectionRecord`; no revised prose
      is generated (there is no model). Recorded in [ADR-0029](../decisions/ADR-0029-explicit-reassessment-trigger.md).

**What P21 did not do (and will not pretend to):** it added no autonomous decision — nothing revises an
answer *unbidden*; it generated no revised wording; it persisted no correction record to disk (a
`CorrectionRecord` is returned and displayed, not auto-stored); and the development double gained no
capability.

## Ordering after this unit

1. A local CPU model adapter behind the `LanguageModel` boundary, only if the P22 assessment finds a
   genuinely available, maintainable, offline dependency — a consequential dependency decision that
   needs its own ADR first (ADR-0001), then the tokenizer, dataset pipeline and a tiny Transformer.
2. A real `SemanticJudge` behind the existing boundary, measured on the synthetic semantic dataset;
   likewise gated on a dependency-feasibility decision recorded before any code.

## Known gaps that bound this unit

- K-001: offline and synthetic; a real workload and any real model have still not been run — the
  frontier of the whole project, and exactly what P22 assesses rather than assumes.
- K-006: the invented-reference condition is persisted per answer, but nothing aggregates it into a
  rate, and it is only durable when the SQLite backend is selected.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- Automatic persistence of `CorrectionRecord`s and `ExperimentResult`s: produced and surfaced, not
  written to disk unless a caller wires a store in.
- The planner uses an anaphoric-cue heuristic, not a discourse resolver; and the reassessment trigger
  reads an explicit identifier, not natural-language intent (P21 was explicit composition, not autonomy).
