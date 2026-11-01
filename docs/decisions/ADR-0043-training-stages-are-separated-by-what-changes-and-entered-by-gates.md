# ADR-0043: base pretraining, mid-training and post-training are separated by what changes, and each is entered through a gate

**Status:** Accepted
**Date:** 2026-11-01
**Related:** [ADR-0033](ADR-0033-model-centred-product-contract.md) (the model is the product, trained from random
initialization), [ADR-0039](ADR-0039-three-level-model-release-gate.md) (a release is gated on evidence),
[ADR-0041](ADR-0041-next-training-target-general-base-with-technical-mixture.md) (data before architecture),
[ADR-0042](ADR-0042-held-out-means-held-out-works.md) (held out by work),
[`open-model-landscape.md`](../research/open-model-landscape.md) (how other projects stage training),
[`chakaso-next-training-design.md`](../research/chakaso-next-training-design.md), K-020.

## Context

The word "training" currently covers three different things in the repository's documents. The curriculum calls
instruction and preference data "post-training, later"; the next-training design says SFT on a non-converged base
"would measure the data, not the model"; E2605–E2612 are all one thing — next-token pretraining from random
initialization — and no document says what would make a later stage a *different* stage rather than a longer run.

Other open projects have converged on a vocabulary for this. As recorded in the landscape document (sourced there,
not re-derived here), OLMo 3 stages its flow as pretraining → mid-training → long-context → SFT → DPO → RLVR, and
SmolLM3 as three pretraining stages → a long-context extension → a reasoning mid-training phase → SFT → an
anchored preference optimisation. They use the terms differently from each other: what one calls a pretraining
stage another calls mid-training. A project that adopts the words without a definition inherits the ambiguity, and
in this project the ambiguity is expensive, because a run that *continues* a checkpoint can look like a longer
base run, a mid-training run or a fine-tune depending on who is describing it.

## Problem

Define the stages so that (a) a run belongs to exactly one, decided by a property that can be checked rather than
by what it is called; (b) entering a later stage requires evidence from the earlier one, so a stage cannot be used
to hide a weak base; and (c) every released weight still has a provenance chain that starts at random
initialization.

## Decision

### 1. What separates the stages is what changes, and each stage changes exactly one thing

| Stage | Starts from | The objective | The data distribution | The schedule | What it is for |
| --- | --- | --- | --- | --- | --- |
| **B — base pretraining** | random initialization | next-token cross-entropy | the declared mixture, as pinned | its own warmup-and-decay | acquiring the language model |
| **M — mid-training** | a Chakaso base checkpoint | **unchanged** | **changed** (re-weighted, higher quality, domain-weighted, longer sequences) | a new, usually shorter decay | shifting what the base knows without changing what it is optimised to do |
| **P — post-training** | a base or mid checkpoint | **changed or reformatted** (assistant-turn loss masking; a preference or reward objective) | instruction, preference or verifiable-task data | its own | changing how the model behaves when asked |

Two consequences are normative:

* **A stage is defined by the change, not by the name a run is given.** A run that continues a checkpoint on the
  same data with the same schedule is a *resume* of stage B, not stage M. A run that continues a checkpoint on a
  different distribution with the same objective is stage M however it is labelled. A run that changes the loss is
  stage P. The existing resume check already encodes the first half
  (`nn_training.verify_resume_compatibility`: any change to learning rate, batch, schedule or seed makes the continuation "a
  different protocol" and is refused), so **mid-training must not be implemented by loosening `resume_from`**.
* **Long-context extension is stage M** under this definition (same objective, changed sequence length and data),
  not a fourth stage.

### 2. Each stage is entered through a gate, and the gate is a measurement that exists before the stage's data does

* **B → M.** A base checkpoint whose held-out loss has a *measured noise floor* (a seed spread at the reference
  configuration; ACTIVE_TASK T4; *measured for the E2612-T configuration by E2614 on 2026-11-01, and not for the released `chakaso-base@0.1.0` checkpoint, whose corpus and split differ*), so that a mid-training difference can be told from seed noise, and whose
  held-out partitions are whole works that no stage will train on (ADR-0042). Mid-training data is chosen on a
  hypothesis the base's measurements pose — a register it is weak on, a length it cannot use — not because mid-training
  is a stage other projects have.
* **B or M → P.** (i) A **coherence or usefulness instrument** that is not a repetition statistic (K-020 is open
  precisely because none exists); (ii) a **behaviour evaluation for the post-trained behaviour, defined before its
  training data** (instruction following, correct refusal, grounded answering), with its floor and contamination
  controls ([`chakaso-capability-benchmarks.md`](../research/chakaso-capability-benchmarks.md)); (iii) the
  base must beat its baselines on whole-work held-out text. **Stage P is not entered to repair a base** — the
  next-training design's rule that SFT must not be run "to make a non-converged base look better" is promoted
  here to a gate.

**Today:** stage B is Implemented and has been run; stages M and P are **Planned**, and **neither gate is open**.
The first is blocked by T4, the second by K-020 and the absence of any behaviour evaluation.

### 3. Rules that hold at every stage

1. **A provenance chain to random initialization.** Every non-base artifact names its parent checkpoint by SHA-256
   and its own pinned data; a chain that cannot be followed to a stage-B run from random initialization is not a
   Chakaso artifact (ADR-0033). No pretrained weights enter at any stage, including as a teacher.
2. **No leakage across stages.** A work held out by any stage's split is held out by every later stage. The held-out
   works are a property of the corpus, fixed once, not of the run.
3. **Each stage is evaluated on its parent's instruments plus its own.** A stage-M result is reported on the base's
   held-out partitions *and* the partitions the mid-training data targets, so a gain on one cannot hide a loss on
   the other (E2612 is the working example: −0.27584 technical and +0.03026 prose).
4. **Later stages never relabel earlier ones.** A good stage-P model does not make its base "Level 3", and a poor
   one does not lower the base's recorded level. ADR-0039's levels attach to the artifact that earned them, each
   with its own card.
5. **Synthetic data is labelled by its source.** Data generated by a Chakaso checkpoint (rejection sampling,
   self-distillation) is admissible and carries `synthetic-self` in its card. Data generated by a hosted or
   pretrained model is **not admissible in a released line**; it may exist only as a separately named research arm
   that is never merged into a released artifact (ADR-0033, and the decision matrix's exclusion of teacher-derived
   data in `chakaso-research-campaign-2026-11.md`).
6. **A stage that cannot be reproduced is not entered.** The pinned-snapshot rule of ADR-0036 applies to every
   stage's data, and the tokenizer is fixed across stages unless the change is itself the experiment (a new
   vocabulary invalidates the parent's embeddings, so it is a new stage-B run, not stage M).

### 4. What this decision does not do

It adds no code, no flag and no directory. The checkpoint format has no `parent` field and this ADR does not add
one: that is justified when a first stage-M run is preregistered, not before (AGENTS.md: no empty scaffolding). It
does not claim stage M or P is a good idea at this scale. The staged recipes the
[landscape document](../research/open-model-landscape.md) records are for models of 1.7B parameters and above
(SmolLM2 1.7B, SmolLM3 3B, OLMo 2 and 3 at 7B and up); it records no staged result at or below 10M, so whether
mid-training transfers to a 3.4M decoder is an open question, not a supported expectation.

## Alternatives considered

| Alternative | Why not |
| --- | --- |
| **Adopt another project's stage names unchanged** | They disagree with each other; Chakaso would inherit the ambiguity without the compute that makes the distinctions matter |
| **Treat any continued run as "fine-tuning"** | Hides whether the objective or only the data changed, which is exactly what decides whether the result is comparable to the base |
| **One stage, "training", with a flag per change** | A flag per change lets a run change three things at once and report one finding; the one-change rule is what keeps a stage result attributable |
| **Define the stages by token count or learning-rate phase** | Both are properties of one run's schedule and say nothing about whether the model's objective or data shifted |
| **Open stage P now, on the grounds that users want an assistant** | ADR-0033 and the next-training design reject it: with no behaviour instrument, SFT would measure its data, not the model |

## Consequences

* ACTIVE_TASK and the roadmap may refer to "stage B / M / P" and mean these definitions.
* A future mid-training preregistration must name its parent checkpoint by digest, the hypothesis that selects its
  data, and the base measurement that gates it; a future post-training preregistration must name its behaviour
  evaluation and floor.
* The decision is revisited when (a) T4 closes (the first gate becomes checkable), (b) a coherence instrument exists
  (K-020), or (c) a source measures mid-training or post-training below 10M parameters.
