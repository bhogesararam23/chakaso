# Active task

## Current unit: P24 — the smallest scientifically honest from-scratch training experiment

**Goal.** ADR-0030 accepted the numerical dependency (`numpy` in an optional `[train]` extra). Build the
minimal trainable language-model experiment it now gates, as an out-of-CI `experiments/<id>/` artifact
(`docs/internal/experiments/README.md` format), producing the project's **first genuinely measured
number** — held-out validation loss over training steps against a trivial baseline. The purpose is to
demonstrate a real learning loop and record it honestly, *not* to build a useful chatbot.

**Scope of this unit (each is a coherent commit when it is meaningful on its own).** Reusable mechanics
live in `chakaso.training` (the module ADR-0005 reserved for training code): a documented corpus with a
licence note, a deterministic train/validation split with leakage prevention (the split is taken before
any vocabulary is fit), a clearly-specified character tokenizer, a tiny causal model with an explicit
architecture, a training objective, a real optimizer step whose gradient is verified by finite
differences, checkpoint save/load, a held-out evaluation, a uniform/unigram baseline, and a reproducible
run config with a fixed seed. Tests are `numpy`-gated (`importorskip`) so the core CI matrix never
installs it; the numerical path is exercised by a separate optional CI job. The experiment record is
written *before* the run (status `running`) and completed with only measured values.

**What this unit is not.** Not a `LanguageModel` adapter over the artifact (that is the following unit,
gated on this one producing an artifact and the evidence supporting it); not a pretrained model; not any
claim of language capability. If the run cannot execute in the environment, record the exact blocker and
the unexecuted steps rather than inventing a number.

## What the previous unit completed (P23 — the dependency decision, ADR before code)

Delivered as a decision record; no model code was written and no dependency was installed.

- [x] [ADR-0030](../decisions/ADR-0030-minimal-optional-training-dependency.md) names the minimal offline
      numerical stack: **`numpy`, isolated in a new optional `[train]` extra** (`numpy>=1.26,<2.5`; the
      `<2.5` cap keeps the Python 3.11 floor working), with **hand-derived gradients verified by finite
      differences** rather than a framework's black-box autograd. It evaluates the candidate stacks on
      the Python floor, Windows, offline reproducibility, CPU-only operation, memory, footprint,
      licensing, determinism/seed control, verifiability, and CI isolation, and cites PyPI metadata for
      the packaging facts.
- [x] A decision that **the base install stays dependency-free and the core CI matrix is unchanged** —
      training tests skip unless the extra is installed, and the numerical path is covered by a separate
      optional CI job. A heavyweight autograd framework (PyTorch) is **rejected for now** with explicit
      revisit conditions (a concrete larger-model workload, or the manual-gradient route becoming a
      correctness risk). Training code is placed in `chakaso.training`, the module ADR-0005 reserved.

## Unit before that (P22 — a real-model feasibility assessment)

Delivered as a research note; no experiment was run and no dependency was added.

- [x] `docs/research/model-feasibility.md` inspects the available machine directly (Python 3.13.5,
      18 logical CPUs, ~15.4 GB RAM with ~1.4 GB free at inspection, Intel Arc — no CUDA, no ML library
      installed, ~160 GiB free disk) and assesses a runnable pretrained model (Option A) against a
      from-scratch training experiment (Option B) across hardware, licence, reproducibility, fit with the
      boundary, and maintenance.
- [x] A recommendation that separates verified facts from labelled estimates and assumptions: defer
      Option A until a real grounded-answer evaluation exists; the smallest honest next step is a scoped,
      out-of-CI from-scratch experiment producing the project's first genuinely measured number — and it
      was gated behind this P23 dependency ADR, now accepted. Logged in the research log; it narrowed the
      "which local model" open question without answering it.

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

1. A `LanguageModel` adapter over the trained artifact — *only if* the experiment produces an artifact
   and the evidence supports integrating it (it may honestly not be ready; then record the blocker). It
   loads the actual weights, uses the exact tokenizer/vocabulary version, validates checkpoint metadata,
   satisfies the model contract, is deterministic at zero temperature, declares itself *not* the
   development double, and fails explicitly when weights, config or the `[train]` extra are unavailable
   — never silently falling back to the double, and never imported into application logic (ADR-0002).
2. Only after that, a genuinely runnable pretrained adapter (Option A) and a real `SemanticJudge`, each
   gated on its own dependency-feasibility decision.

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
