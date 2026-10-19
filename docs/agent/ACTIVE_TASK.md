# Active task

## Current unit: P28 — a retrieval evidence-sufficiency decision (justified by E2603)

**Goal.** P27 built the grounded-answer evaluation workload and E2603 ran it. The single largest
correctness problem that measurement demonstrated is not in the answer layer but the retrieval layer:
lexical `RetrievalService.search` returns its full top-k *even when nothing supports an answer*, so
all three abstention cases produced a **false retrieval** and three cases surfaced a passage they had
marked **forbidden**. A grounded system cannot abstain correctly if retrieval always hands it
evidence. This unit designs and, where the evidence supports it, implements a narrow, tested fix: a
retrieval evidence-sufficiency signal — a documented score floor or relevance test that lets a
retrieving turn distinguish "I found nothing relevant" from "I found k things" — driven *only* by
those specific E2603 failure cases and covered by regression tests that show the failure first.

**Scope.** A change to the retrieval boundary or the `RetrievalService`/runtime composition, with an
ADR or an update to `docs/retrieval.md`/`docs/evaluation.md` in the same commit, plus the workload
re-run (E2603's successor or a reassessment) confirming the false-retrieval count moved. It must not
weaken recall on the answerable cases to win on abstention; both dimensions are reported separately.

**What this unit is not.** Not a new benchmark, not a model, not a retrained anything. If a clean
sufficiency rule cannot be shown to help without hurting recall, the finding is recorded and the
change deferred rather than forced.

## What the previous unit completed (P27 — the grounded-answer evaluation workload)

Designed, built, tested, run and recorded (ADR-0031; E2603).

- [x] `chakaso.benchmark.grounded_*` — a grounded-answer evaluation sub-module: a provenance-bearing
      `GroundedCase`/`GroundedSource`/`GroundedProvenance` schema (question, expected disposition,
      required facts, prohibited claims, gold/forbidden evidence by label, rationale, split), a
      `GroundedDataset` that refuses dangling evidence, duplicate cases and development/held-out split
      leakage, and a strict manifest+JSONL loader with the same validation.
- [x] A **self-authored Apache-2.0** workload (`ga-kestrel-library`, 13 passages / 14 cases, 9 dev /
      5 held-out, pinned content fingerprint `d2d2ee00d117b88a`) spanning direct, multi-source,
      distractor, conflicting, ambiguous, missing-evidence and unsupported-premise situations. Every
      required fact is asserted (by test) to be literally present in its gold evidence.
- [x] `GroundedEvaluationRunner` composing the real `RetrievalService` and `evaluate_answer` over an
      injected answer provider, classifying retrieval / generation / evaluation failures distinctly
      (a failure is recorded, never scored as success); a **reference oracle** (an oracle, explicitly
      not a model) and a non-answering provider; hand-checkable metric primitives over separate
      dimensions (no merged score); deterministic text and JSON reports that name the provider and
      whether it was an oracle; and `chakaso benchmark grounded` (`--provider`, `--split`, `--json`).
- [x] **E2603 measured result:** reference-oracle run over the workload — retrieval recall@k 0.8182 /
      MRR 0.8636 (held-out recall@k 1.0), zero stage failures, grounded-answer rate 0.8182, and honest
      negatives kept visible: 3 forbidden hits and 3 false retrievals. Recorded as a harness/retrieval
      measurement, never answer capability (no model answers; K-001 stands).

## What the unit before that completed (P26 — E2602, the training result reproduced across seeds)

Implemented, run, and recorded as a measured result
([`../../experiments/E2602/record.md`](../../experiments/E2602/record.md)).

- [x] `data.split_three_way` + `ThreeWaySplit` — a deterministic, disjoint train/validation/test
      separation (units wholly on one side, every unit exactly once) so early-stopping selection on
      validation never touches the reported test number; `fit_vocabulary` made structural to serve both
      split shapes. 5 numpy-free split tests.
- [x] `chakaso.training.controlled.run_controlled_experiment` — a fixed early-stopping protocol over
      pre-declared seeds, aggregating mean + sample stdev of held-out test loss against baselines.
- [x] **E2602 measured result:** mean test loss at the validation-selected step **1.7978 ± 0.1767 nats**,
      below both the unigram (mean 2.9808) and uniform (~3.3) baselines on **all five** seeds; the
      un-early-stopped step-300 loss (mean 2.7804) is worse on every seed — E2601's overfitting is
      systematic, not one lucky partition. Reported honestly as a small-corpus, high-variance toy result.

## What the unit before that completed (P25 — a trained artifact behind the `LanguageModel` boundary)

Implemented, tested, and honest about its limits; it integrates E2601's artifact without distorting the
interface.

- [x] `chakaso.models.trainedchar.TrainedCharacterModel` — a `trained_char` adapter selected through
      `model.adapter`/`model.checkpoint_path`: loads a checkpoint, rebuilds its bound vocabulary, generates
      greedily within `max_new_tokens` (reports `LENGTH` honestly — no learned stop token), declares
      `development_double=False` and implements `TOKENIZE`, and fails explicitly on an empty/missing/
      corrupt checkpoint rather than falling back to the double.
- [x] The checkpoint now embeds the vocabulary characters (schema/2, validated against the stored
      fingerprint) so a checkpoint can decode on its own — the measured E2601 losses are unchanged.
- [x] Lazy registry import keeps `import chakaso.models` and the base install numpy-free; the adapter
      inherits the full shared `LanguageModel` contract over a real trained checkpoint (numpy-gated tests).
- [x] Honest scope: the model is a tiny toy that produces incoherent text; the value is a real
      training-and-inference path, not usefulness, and that is stated in the adapter, the docs and the
      changelog.

## What the unit before that completed (P24 — the from-scratch training experiment, E2601)

Implemented, tested, run, and documented as the project's first measured result
([`../../experiments/E2601/record.md`](../../experiments/E2601/record.md)).

- [x] `chakaso.training` built end to end (the module ADR-0005 reserved): a self-authored Apache-2.0
      corpus, a deterministic character vocabulary (codepoint-ordered ids, reserved unknown id, content
      fingerprint), a **leak-free** line-unit split with the vocabulary fit on training text only, a
      validated `TrainingConfig`, a tiny causal character model with **hand-derived gradients verified
      against central finite differences for every scalar** (this caught and fixed a real numpy
      fancy-index scatter bug), an in-project Adam optimizer, a full-batch training loop that records a
      held-out curve, a content-checksummed JSON checkpoint that refuses corruption and incompatibility,
      uniform/unigram baselines, held-out evaluation, and an `E2601` experiment driver.
- [x] **E2601 measured result:** from random initialization, held-out next-character cross-entropy falls
      from 3.3118 nats (step 0) to **1.6650** (best, step 50) and ends at **2.4946** (step 300) — below
      **both** the unigram (3.0531) and uniform (3.3322) baselines; final bits/char 3.5989, perplexity
      12.12, next-char accuracy 0.6026; 984 parameters, seed 7, Windows/CPython 3.13.5/numpy 2.4.6, CPU
      only. The measured **overfitting** after step 50 is reported, not hidden.
- [x] The dependency-free layers (corpus/tokenizer/split/config/baselines) run in the core CI matrix;
      the numpy layers are `importorskip`-gated and pass with `[train]` installed. **A dedicated `[train]`
      CI job was NOT delivered** — the session's GitHub token lacks the `workflow` scope, so the
      workflow file cannot be pushed; recorded as K-009 rather than claimed as done.

## What the unit before that completed (P23 — the dependency decision, ADR before code)

Delivered as a decision record; no model code was written and no dependency was installed.

- [x] [ADR-0030](../decisions/ADR-0030-minimal-optional-training-dependency.md) names the minimal offline
      numerical stack: **`numpy`, isolated in a new optional `[train]` extra** (`numpy>=1.26,<2.5`; the
      `<2.5` cap keeps the Python 3.11 floor working), with **hand-derived gradients verified by finite
      differences** rather than a framework's black-box autograd. It evaluates the candidate stacks on
      the Python floor, Windows, offline reproducibility, CPU-only operation, memory, footprint,
      licensing, determinism/seed control, verifiability, and CI isolation, and cites PyPI metadata for
      the packaging facts.
- [x] A decision that **the base install stays dependency-free and the core CI matrix is unchanged** —
      training tests skip unless the extra is installed. A heavyweight autograd framework (PyTorch) is
      **rejected for now** with explicit
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

1. A controlled follow-up experiment to E2601, justified by its measured overfitting: early stopping /
   capacity selection on held-out loss under a fixed protocol (not a cherry-pick), and held-out loss
   across a few seeds so the result carries an error bar rather than a point.
2. Only after a trustworthy training/inference path exists, a genuinely useful pretrained adapter (Option
   A) and a real `SemanticJudge`, each gated on its own dependency-feasibility decision — and a real
   grounded-answer evaluation workload (K-001's stronger form) that none of the toy work yet touches.

## Known gaps that bound this unit

- K-001: offline and synthetic; a *useful* model on a real workload has still not been run. E2601 trained
  and measured a real (but toy) model, narrowing the training-loop half of this gap, but the project has
  still met no real corpus or real question — the frontier of the whole project.
- K-009: the numpy-gated training tests do not run in remote CI, because the session's GitHub token
  lacks the `workflow` scope and the `[train]` CI job cannot be pushed. Verified locally instead.
- K-006: the invented-reference condition is persisted per answer, but nothing aggregates it into a
  rate, and it is only durable when the SQLite backend is selected.
- K-008: the fetcher's destination screen is partial; any acquisition work must confirm no default path
  reaches the network and record the limitation rather than fix it silently.

## What previous units knowingly left undone

- Automatic persistence of `CorrectionRecord`s and `ExperimentResult`s: produced and surfaced, not
  written to disk unless a caller wires a store in.
- The planner uses an anaphoric-cue heuristic, not a discourse resolver; and the reassessment trigger
  reads an explicit identifier, not natural-language intent (P21 was explicit composition, not autonomy).
