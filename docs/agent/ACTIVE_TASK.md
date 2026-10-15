# Active task

## Current unit: P23 — the dependency decision that gates any real model (ADR before code)

**Goal.** Before a single line of model, tokenizer or training code exists, make the one decision the
P22 feasibility assessment surfaced as the true gate: name the minimal **offline numerical stack** the
project is willing to admit as an *optional, non-default* dependency (a `[train]` / `[models]` extra that
never enters the base install or the CI matrix), and record it as an ADR. That ADR is the pre-condition
for every subsequent model unit — a runnable adapter or a from-scratch training experiment — per the
roadmap's rule that a consequential dependency decision precedes its code (ADR-0001).

**Why this is next.** P22's assessment (`docs/research/model-feasibility.md`) found that both directions —
a small pretrained local model and a tiny from-scratch training run — fit the `LanguageModel` boundary
cleanly and are bounded not by architecture but by the willingness to carry a numerical runtime and a
memory budget. No model work can be honest until that dependency choice is written down. This unit
remains research plus a decision record: it writes no model code and downloads nothing.

**What this unit is not.** Not the model adapter, the tokenizer or the training loop themselves; not a
dependency install; not any measured result. It ends with an ADR and a stated smallest-experiment, gated
behind that ADR.

## What the previous unit completed (P22 — a real-model feasibility assessment)

Delivered as a research note; no experiment was run and no dependency was added.

- [x] `docs/research/model-feasibility.md` inspects the available machine directly (Python 3.13.5,
      18 logical CPUs, ~15.4 GB RAM with ~1.4 GB free at inspection, Intel Arc — no CUDA, no ML library
      installed, ~160 GiB free disk) and assesses a runnable pretrained model (Option A) against a
      from-scratch training experiment (Option B) across hardware, licence, reproducibility, fit with the
      boundary, and maintenance.
- [x] A recommendation that separates verified facts from labelled estimates and assumptions: defer
      Option A until a real grounded-answer evaluation exists; the smallest honest next step is a scoped,
      out-of-CI from-scratch experiment producing the project's first genuinely measured number — and it
      is gated behind this P23 dependency ADR. Logged in the research log; it narrows the "which local
      model" open question without answering it.

## Unit before that (P21 — a triggered, explicitly-composed reassessment)

Implemented, tested, documented and green on Python 3.11–3.13 (recorded in
[ADR-0029](../decisions/ADR-0029-explicit-reassessment-trigger.md)).

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
      is generated (there is no model).

**What P21 did not do (and will not pretend to):** no autonomous decision (nothing revises an answer
*unbidden*), no revised wording, no correction record auto-persisted to disk, and the development double
gained no capability.

## Ordering after this unit

1. A minimal from-scratch training experiment as an out-of-CI `experiments/<id>/` artifact, gated behind
   the P23 ADR: a char-level model, fixed seed and config, a held-out set, reporting the project's first
   measured number (validation loss over steps) with the explicit limitation that it is a learning-loop
   demonstration, not a claim of language capability.
2. A `LanguageModel` adapter over the trained artifact; only after that, a genuinely runnable pretrained
   adapter (Option A) and a real `SemanticJudge`, each gated on its own dependency-feasibility decision.

## Known gaps that bound this unit

- K-001: offline and synthetic; a real workload and any real model have still not been run — the
  frontier of the whole project, and exactly what P22 assessed rather than assumed.
- K-006: the invented-reference condition is persisted per answer, but nothing aggregates it into a
  rate, and it is only durable when the SQLite backend is selected.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- Automatic persistence of `CorrectionRecord`s and `ExperimentResult`s: produced and surfaced, not
  written to disk unless a caller wires a store in.
- The planner uses an anaphoric-cue heuristic, not a discourse resolver; and the reassessment trigger
  reads an explicit identifier, not natural-language intent (P21 was explicit composition, not autonomy).
