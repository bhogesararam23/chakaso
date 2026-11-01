# Open-model research: Pythia, and two TinyLlama facts the earlier notes missed

**Status:** Research — a sourced reading of primary materials. Nothing here is measured in this repository, and
nothing is reproduced by Chakaso. Figures are as the authors report them unless marked *computed*, which means this
note did the arithmetic and shows it.
**Date:** 2026-11-01. **Access date for every source below:** 2026-11-01.
**Why this note exists.** The 2026-11-01 campaign read nine families
([`open-model-landscape.md`](../../research/open-model-landscape.md)) and gave Pythia one passing mention. Pythia
is the family closest to the instrument Chakaso lacks: a controlled suite with 14M–410M members, released
intermediate checkpoints, a replayable data order, and ten seeds per small size. The campaign's sharpest open
problem is that no instrument here has a measured noise floor
([ACTIVE_TASK](../../agent/ACTIVE_TASK.md) T4), so the one family that publishes seed variants is worth reading
properly.

## Sources

| # | Source | What was read |
| --- | --- | --- |
| P1 | Biderman et al., *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, [arXiv:2304.01373](https://arxiv.org/abs/2304.01373), first version 2023-04-03 | the abstract, via arXiv's API |
| P2 | EleutherAI, [`pythia` repository README](https://github.com/EleutherAI/pythia), branch `main` | the full README |
| T1 | Zhang et al., *TinyLlama: An Open-Source Small Language Model*, [arXiv:2401.02385](https://arxiv.org/abs/2401.02385), first version 2024-01-04 | the abstract (API) and the paper text (ar5iv rendering) |
| T2 | TinyLlama [repository README](https://github.com/jzhang38/TinyLlama), branch `main` | the release table and notes |

The Pythia paper's body was **not** read; only its abstract and the project's own README were. Everything
attributed to Pythia below is therefore a claim of the README or the abstract, and the README is a maintained page
that postdates the paper (it records edits through July 2024).

## 1. Pythia, as published

* **What it is** (P1): "a suite of 16 LLMs all trained on public data seen in the exact same order and ranging in
  size from 70M to 12B parameters", with "154 checkpoints for each one of the 16 models, alongside tools to download
  and reconstruct their exact training dataloaders."
* **Later additions** (P2): 14M and 31M models were added on 2 November 2023 "at the request of some researchers".
* **Training budget** (P2): every model saw 299,892,736,000 tokens (~300B) at a batch of 2,097,152 tokens for
  143,000 steps — just under one epoch of the Pile, and ~1.5 epochs of its 207B-token deduplicated version.
  *Computed:* 2,097,152 × 143,000 = 299,892,736,000, so the README's two figures agree exactly.
* **The small configurations** (P2):

  | Params | layers | `d_model` | heads | `d_head` | batch | peak LR |
  | --- | ---: | ---: | ---: | ---: | --- | --- |
  | 14M | 6 | 128 | 4 | 32 | 2M | 1.0e-3 |
  | 31M | 6 | 256 | 8 | 32 | 2M | 1.0e-3 |
  | 70M | 6 | 512 | 8 | 64 | 2M | 1.0e-3 |

* **Checkpoints** (P2): steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, then every 1,000 steps — log-spaced early,
  linear after. *Computed:* 11 log-spaced + 143 linear = 154, matching the stated count.
* **Seeds** (P2): the suite's default seed is the GPT-NeoX default, 1234. Models at **14M, 31M, 70M, 160M and 410M**
  were additionally trained with seeds 1 through 9 ("standard" Pile only, not deduplicated), which with the original
  makes "ten otherwise identical variants using different random seeds". Their training curves are in a public
  Weights & Biases project named in the README.
* **Data order** (P2): a tool, `utils/batch_viewer.py`, reconstructs any step's exact batch from the released
  pre-tokenised data; documents are concatenated and separated by an end-of-document token, and "each sample or batch
  may not start with an EOD token".
* **Licence** (P2): Apache-2.0 for the code, the models and "any other copyrightable artifacts" in the repository.

## 2. What Chakaso can take from it, and what it cannot

### 2.1 The seed-spread design: a published precedent for ten, not for three

Chakaso has a five-seed spread for a 984-parameter character model (E2602, seeds 7, 21, 1337, 2024, 42) and **none for the decoder**; the 0.02
bits-per-character "band" every decoder record leans on is a convention inherited from E2611, not a measured noise
floor (T4, [K-001](../../agent/KNOWN_ISSUES.md)). Pythia chose **ten** variants at every size from 14M to 410M. That
is a reported design choice, not a derivation — the README gives no reason for ten — but the arithmetic of what
three seeds can establish is checkable.

*Computed* — the 95 % interval for a standard deviation estimated from *n* runs, assuming the seed-to-seed
variation of a held-out loss is approximately normal (an assumption, not a measurement), is
s·√(df/χ²<sub>0.975,df</sub>) to s·√(df/χ²<sub>0.025,df</sub>):

| runs *n* | df | interval for the true σ, as a multiple of the sample s |
| ---: | ---: | --- |
| 3 | 2 | **0.52 s to 6.29 s** |
| 5 | 4 | 0.60 s to 2.87 s |
| 7 | 6 | 0.64 s to 2.20 s |
| 10 | 9 | 0.69 s to 1.83 s |

(χ² quantiles for df 2, 4 and 6 were computed from the closed-form CDF of the even-df χ² distribution; the df 9
values, 2.700 and 19.023, are the standard table entries.) So T4 as planned — three seeds — would return a standard
deviation that may be a half or six times the true one. That is enough to say whether a spread is *wildly* larger
than 0.02, and not enough to say whether 0.02 sits above noise, which is the question the band exists to answer.
**Recommendation (hypothesis, not result):** run T4 with five seeds if the budget allows, and report the interval,
not the point estimate. Three is a pilot.

### 2.2 What "seed" varies must be said

Chakaso's training seed is "the master seed for initialization and for the batch plan"
([`nn_training.py`](../../../src/chakaso/training/nn_training.py)): initialisation and data order change together,
and the split seed is separate. A Chakaso seed spread is therefore a spread over *init + order jointly*, with the
split and the tokenizer held fixed. Pythia's README does not say whether its extra seeds change the data order — it
says the models were "trained on public data seen in the exact same order", which describes the original suite — so
**the two spreads are not known to measure the same thing**, and neither should be quoted as the other's noise.

### 2.3 Replayable data order: Chakaso already has the property, and tests it

Pythia's headline instrument is that any batch can be reconstructed. Chakaso's batch plan is a pure function of
`(seed, update_index, data size)` (`plan_batch`, `nn_training.py:288`), tested as a pure function
(`test_the_batch_plan_is_a_pure_function_of_the_schedule`) and the reason a resumed run lands on the batches the
uninterrupted run would have drawn. What Chakaso lacks is the *viewer* — a way to ask "what text was in step 1,200's
batch?" — which would let a regression be traced to the documents that caused it. That is a debugging tool, not a
measurement; nothing here argues for building it now.

### 2.4 Intermediate checkpoints: the one genuine gap

Pythia keeps 154 checkpoints per run, log-spaced early. Chakaso keeps the best-validation state, the final state and
a resumable file, and its curve has **seven evaluation points** per 2,400-step run (every 400). Every decoder record
is truncated at the step ceiling, so *whether loss is still falling* is the question that decides whether a floor is
close to a result — and seven points cannot show a knee. A log-spaced evaluation schedule is cheap relative to a
90-minute arm (each evaluation is one pass over a 250k-token pack), but it changes a protocol field and is not
justified until a preregistered question needs it. **Verdict: ADAPT later, on demand; do not add now.**

### 2.5 Scale: Pythia's smallest member is ~4× Chakaso's largest and was trained ~15,000× longer

| | Pythia 14M | Chakaso E2608 decoder |
| --- | --- | --- |
| layers / `d_model` / heads | 6 / 128 / 4 | 4 / 256 / 8 |
| tokens seen | 299,892,736,000 | 19,660,800 |
| batch (tokens) | 2,097,152 | 8,192 |
| optimiser steps | 143,000 | 2,400 |

*Computed:* the token ratio is 299,892,736,000 / 19,660,800 ≈ **15,253**; the step ratio is ≈ 59.6; the batch ratio
is 256; and the label "14M" over Chakaso's 3,413,760 parameters is ≈ 4.1. Pythia's smallest model is not near
Chakaso's regime — it is larger and was trained for an order of magnitude more steps on four orders more tokens. Any Pythia number (loss, seed spread, checkpoint behaviour) describes a converged-ish
small model on a 300B-token mixture and **says nothing direct about a 20M-token truncated run**. What transfers is the
*design*, not the values.

## 3. Verdicts for Chakaso

| # | Mechanism | Verdict | Reason |
| --- | --- | --- | --- |
| 1 | Seed variants per size | **ADOPT as design; size by the interval above** | the only published small-model seed suite; five runs, not three, if affordable |
| 2 | Same data, same order across a suite | **Already true** | the batch plan is a tested pure function |
| 3 | Dense, log-spaced checkpoints | **ADAPT on demand** | would show whether truncated loss is still falling; not yet a preregistered need |
| 4 | Batch viewer | **DEFER** | a debugging aid; no open question needs it |
| 5 | Train on ~300B tokens | **REJECT as a target** | infeasible on this host and not the question |
| 6 | Pythia's configs as a baseline to beat | **REJECT** | different tokenizer, data and ~15,000× the tokens; no comparison is meaningful |

## 4. Two TinyLlama facts the earlier notes did not record

The 2026-11-01 notes cover TinyLlama's overtraining regime. Two further facts from its own paper and README bear on
how Chakaso reads its own results.

**4.1 A training-pipeline defect forced a from-scratch retrain, and the authors say so** (T1, §2.5). During the open
pre-training run, "a few implementation issues were identified within the training framework" — the paper names the
learning-rate scheduler and data loading, each with a tracked issue — and "we therefore re-trained a new model from
scratch after fixing those issues", calling the result version 1.1. This is the same event Chakaso lived on
2026-11-01 (the byte-pair merge loop, K-024): a defect in plumbing that no metric flagged, discovered by someone
looking, repaired by re-deriving the artifact. It is a published precedent for the position ADR-0036 takes — a
result whose pipeline was wrong is *re-run*, not footnoted.

**4.2 The release table's own non-monotonic commonsense average, and a named cause** (T2). Averaged zero-shot
commonsense scores for the base checkpoints read 46.11 (105B tokens), 48.28 (503B), 50.22 (1T), 51.28 (1.5T), 51.64
(2T), **53.86 (2.5T)**, **52.99 (3T)** — rising, then falling by 0.87 at the end. The README attaches a note to the
2T→2.5T jump, "related to [a] bos_id issue": a begin-of-sequence handling defect, in the project's own words.
*Interpretation (mine, not the authors'):* a benchmark average moved by about two points for a reason that was a
pipeline artefact rather than a learning event, and the 3T checkpoint, trained longest, scored below the 2.5T one.
Neither is evidence against TinyLlama; both are evidence that **a benchmark average on a single run is not a
learning curve**, which is the position Chakaso's E2612 (held-out loss and repetition moving in opposite directions)
reached independently.

**4.3 MMLU is at chance for a 1.1B model** (T1, Table 4). 5-shot MMLU: TinyLlama v1.0 **25.34**, Pythia-1.0B 25.70,
Pythia-1.4B 25.41, OPT-1.3B 24.90. MMLU is a four-choice test, so uniform guessing scores 25 %; every model in the
table is within one point of it. *Computed:* the largest gap to 25 is 0.70 points. A knowledge benchmark that cannot
separate a 1.1B model trained on 3T tokens from a random guesser cannot be the instrument at 3.4M parameters —
which is what [`chakaso-capability-benchmarks.md`](../../research/chakaso-capability-benchmarks.md) already
concludes by other routes, now with a published scale-matched data point.

## 5. What this changes in the repository

* [ACTIVE_TASK](../../agent/ACTIVE_TASK.md) T4 now says what three seeds can and cannot establish and recommends
  five.
* Nothing else is implemented. No run follows from this note; it is a sourced reading and a piece of arithmetic.

## 6. Limits

* The Pythia **paper's body** was not read, so its findings (memorisation, term-frequency effects, gender-bias
  interventions) are not summarised here; only the suite's design and the README's statements are.
* The seed interval assumes approximately normal seed-to-seed variation. Held-out loss across seeds often is, but
  **Chakaso has no decoder seed data to check it against**, which is the point of T4.
* TinyLlama's README is a live page; the release table above is as read on 2026-11-01.
