# E2606 — How much of E2605's result is the seed?

- Status: running (preregistered, not yet executed)
- Date: 2026-10-21
- Hypothesis: [docs/research/hypotheses.md](../../docs/research/hypotheses.md) — none of H1–H7. This experiment
  tests the *stability* of E2605's measured claim, which is a prerequisite for treating it as evidence at
  all. Recorded as a pre-declared study in `docs/research/next-model-experiment.md`.
- Owner: the project
- Git commit: this record is committed before any study seed runs. The commit that runs them is named in the
  Results section when they exist, and nowhere before.
- Config hash: the protocol is the table below; each run also writes its own resolved configuration into
  `data/experiments/E2606/E2606/seed-<n>/report.json`, whose digest is recorded per run.

## Motivation

E2605 measured 1.8657 bits per character for a 643,840-parameter subword Transformer against 3.1343 for the
character model on the same held-out documents ([`../E2605/record.md`](E2605/record.md)). It did that with
**one seed**. A single run cannot separate an effect from the initialization and batch order that produced
it, so the project's headline number currently has no spread attached to it, and nothing in the repository
says how large that spread is.

E2602 measured a five-seed reproduction for the 984-parameter character model (mean test loss 1.7978 ± 0.1767
nats, below both baselines on every seed). This is the same question asked of the model path that E2605
built: does the Transformer's advantage over the character architecture, over unigram and over uniform hold
across seeds, and how much does it move?

The decision this informs: whether the subword Transformer is worth scaling, and whether E2605's number can
be quoted at all without an interval. If the spread is large relative to the gap, the next experiment is not
a bigger model — it is a larger corpus, because a 780-kilobyte self-authored corpus would then be the
binding constraint rather than the architecture.

## Reproduction, which this study performs as a side effect

Seed 7 with split seed 7 **is** E2605's protocol. That run is therefore compared against E2605's recorded
report, field by field, at four distinct levels:

1. **Artifact and provenance reproduction** — the corpus fingerprint, the document split fingerprint, the
   tokenizer fingerprint, the architecture and the parameter count must be *equal*, with no tolerance. These
   are declared identities, not measurements.
2. **Procedure reproduction** — the recorded validation curve must have the same evaluation steps, the same
   learning-rate schedule, the same pre-clip gradient norms at each step, and the same selected step. Equal
   step-for-step, or the procedure differed.
3. **Numerical reproduction** — the held-out metrics must agree to within the tolerances declared below.
4. **Deviation report** — anything that fails 1, 2 or 3 is listed with the field, the recorded value, the
   reproduced value and the difference. E2605's record is not edited to match; this record is the deviation
   report.

**Tolerances, fixed here before any run of this study produced a number:**

| Quantity | Tolerance | Why that number |
| --- | --- | --- |
| corpus / split / tokenizer fingerprint, architecture, parameter count, `best_step`, `stopped_early` | exact | identities, not measurements |
| Transformer held-out bits per character | 0.0005 | half of the last decimal the records report |
| Transformer held-out nats per token | 0.0005 | same reported precision |
| Transformer perplexity | 0.005 | precision of the loss it exponentiates |
| Transformer token accuracy | 0.0005 | same reported precision |
| Character model bits per character and nats per token | 0.0005 | same reported precision |
| Unigram and uniform baselines | 0.0 | deterministic counts over fixed targets; any movement is a data change |
| Wall-clock times | none | not a reproduction claim; recorded and compared descriptively |

These live in code as `chakaso.training.seed_study.REPRODUCTION_TOLERANCES` and were written into this
record before the run. The known sources of nondeterminism they cover: torch's intra-op thread count (now
recorded in each artifact's environment block, and *not* recorded by E2605's), BLAS reduction order, and
library version drift between the two runs. E2605's block names torch 2.13.0+cpu, Python 3.13.5 and
Windows-11-10.0.26200-SP0; this study runs on the same machine, so the declared environment for this study
is the same, with the thread count additionally recorded per run.

Exact bit-for-bit reproduction is expected — same machine, same library versions, single-process CPU
deterministic kernels — but it is not *required* for a pass, because the thread count alone can reorder a
float reduction. If it does, the tolerance is what decides, and the tolerance was declared first.

## Research question and hypothesis

**Question.** Across pre-declared random seeds, with the corpus, split, tokenizer, architecture and budget
held fixed, how much does the Transformer's held-out bits-per-character move, and does its advantage over the
character architecture and over both trivial baselines persist on every seed?

**Hypothesis.** The Transformer's held-out bits per character is lower than the character model's, lower than
the subword unigram baseline and lower than the uniform baseline on every declared seed, and the spread
across seeds is small relative to the Transformer-to-character gap.

The third clause is the one that can fail, and failing it is the interesting result: it would mean E2605's
comparison was a single draw from a wide distribution.

## Primary metric

**Held-out bits per character** (`chakaso.training.model_metrics.metrics_from_scores`), computed over the
decoded whole target stream of the *test* documents, with separator and unknown-character targets excluded
from the denominator and reported by count. This is the only metric that is comparable across the two
tokenizations, so it is the only primary metric.

## Secondary metrics

- Transformer held-out nats per token and perplexity — **tokenizer-bound**, labelled as such wherever printed.
- Transformer held-out token accuracy.
- Character model held-out bits per character and nats per token (same architecture, its own vocabulary).
- Subword unigram and uniform bits per character on the same test positions.
- Selected optimizer step, whether early stopping fired, and wall-clock seconds per stage.
- Checkpoint checksum per run, so a reported model can be re-identified.

## Baseline definitions

| Baseline | Definition | Scored on |
| --- | --- | --- |
| uniform | `log(vocab_size)` nats at every position | the same test token positions as the Transformer |
| subword unigram | add-alpha (`alpha = 1`) counts fit on the **train** side only | the same test token positions |
| character MLP | E2601/E2602's architecture, `CharCausalMLP`, 16-character context, fit on train, selected on validation | the same test **documents**, in characters |

Every model and baseline is scored on the same test documents, from the same split, with the same
normalization. Architectural asymmetry that is *not* removed and must be read with the numbers: the
Transformer conditions on 128 subword tokens (about 266 characters at E2605's measured 2.08 characters per
token) while the character model conditions on 16 characters. The character baseline is neither re-tuned nor
enlarged for this study; changing it would require a separately declared experiment.

## Dataset and immutable fingerprints

| Field | Value |
| --- | --- |
| name / version | `chakaso-repo-text-v1` / `1.0.0` |
| documents | 71 |
| characters | 780,223 |
| corpus fingerprint | `e45993f9badf706c` |
| manifest checksum | `3255f98d9bdface2…` ([`E2605/corpus-manifest.json`](E2605/corpus-manifest.json)) |
| licence | Apache-2.0, the project's own licence; text authored for Chakaso |
| recovery revision | `9b8a478` → commit `9b8a47876de7e13c60aefca0b9c7fe501bc03def` |
| snapshot checksum after recovery | recorded per run from the file it loaded |

The corpus is this repository's own tracked markdown plus `LICENSE` and `AGENTS.md`, as it stood at
`9b8a478`. Every run loads it through a **declared snapshot** (`--corpus-snapshot`) and declares
`--expected-corpus-fingerprint e45993f9badf706c`, so a run that resolved a different corpus stops before
training rather than reporting a number against it. The tracked text has since been edited
(`chakaso corpus verify` names the drifted files), which is exactly why the snapshot is the input here and
not the working tree.

## Split, and what is held fixed across seeds

| Field | Value |
| --- | --- |
| split function | `chakaso.training.data.split_text_units`, whole documents, deduplicated by content first |
| split seed | **7 for every run** (`--split-seed 7`), which is E2605's assignment |
| split fingerprint | `87067cca96275def` — every run must report it |
| fractions | validation 0.15, test 0.15 |
| documents per side | 49 train / 11 validation / 11 test, 0 duplicates dropped |

**The split does not vary with the seed.** `ExperimentPlan.split_seed` exists for this study. Without it,
varying the master seed would re-draw the held-out documents, and the runs would differ in their data as
well as their initialization — the aggregate would then describe a mixture of experiments rather than the
effect of a seed. This is a protocol decision, recorded in
[ADR-0036](../../docs/decisions/ADR-0036-corpus-snapshot-recovery-from-history.md)'s consequences and here.

The tokenizer is re-derived per run from the same training documents; it is deterministic, so the recorded
`tokenizer_fingerprint` must be identical across runs. `build_study_report` refuses to aggregate runs whose
protocol fields disagree, which is the mechanical form of this claim.

## Model and complete configuration

Identical for every run except `seed`:

```text
architecture   vocab_size 1024, context_length 128, d_model 128, num_layers 3, num_heads 4,
               feedforward_dim 384, dropout 0.0, tie_embeddings true, initialization_std 0.02
optimizer      AdamW-style: learning_rate 6e-4, weight_decay 0.01, warmup_steps 60,
               gradient_clip_norm 1.0 (pre-clip norm recorded)
budget         batch_size 32, max_steps 2400, eval_every 60
stopping       patience 8, min_delta 0.001 nats, select on validation
packing        context_length + 1 per sequence, targets shifted by one, dropped tail reported
seed           the declared study seed (initialization and batch plan only)
```

Parameter count 643,840 (analytic from the configuration and cross-checked against the built model in
`chakaso.training.transformer`); recorded per run, must agree across runs.

The character baseline keeps E2605's own configuration: context 16 characters, embed 32, hidden 64, batch
256 windows, 4,000 steps, evaluation interval clamped to its budget, learning rate 2e-2, 44,084 parameters,
hand-derived gradients (ADR-0030).

## Random seeds — declared, not chosen

Seven, eleven, thirteen, seventeen, nineteen.

The rule was fixed before any run of this study produced a number: **seed 7 is E2605's own seed** (so the
study contains its own reproduction), and the remaining four are **the next four primes after 7**. Nothing
about these values was selected using a measured result; they are the output of that rule. Five seeds is
E2602's precedent, so the two multi-seed records in this repository are comparable in sample size.

Number of runs: five, one per seed, each in its own directory
(`data/experiments/E2606/E2606/seed-<n>/`).

## Hardware and software requirements

CPU only, offline, no GPU, no network, no provider API. Measured machine: Windows 11 (10.0.26200), 18
logical CPUs, ~15.4 GB RAM, ~160 GiB free disk, CPython 3.13.5, torch 2.13.0+cpu, numpy as installed. Each
run's `environment` block records the torch and numpy versions, the platform, the device and the torch
thread count; those are compared, not assumed.

**Runtime, measured rather than estimated.** E2605's full protocol took 513.6 s (Transformer, stopped at
step 1440 of 2,400) + 69.9 s (character baseline) + 13.2 s (preparation) = **596.7 s per run** on this
machine. A five-seed study is therefore ~50 minutes of training, which fits the machine and the session, so
the study runs at the full declared budget with all five seeds.

### Technical pilot, measured before this study ran

A pilot on **a different experiment id (`E2606P`) and a reduced budget of 120 optimizer steps**, two seeds,
on the recovered snapshot. It is a check of the machinery and of the time cost; its numbers are **not** part
of this study's results, are not included in any aggregate below, and did not influence the declared seeds,
the budget or the tolerances.

| What the pilot checked | Measured |
| --- | --- |
| corpus resolved from the declared snapshot | fingerprint `e45993f9badf706c`, 71 documents, 780,223 characters, snapshot checksum `84ecbc61cf1e634b…`, acquisition revision `9b8a47876de7e13c60aefca0b9c7fe501bc03def` |
| split and tokenizer identity | split `87067cca96275def`, tokenizer `2605bf8f50183a6e` — both equal to E2605's recorded values |
| parameter count | 643,840, equal to the recorded value |
| test denominators | 46,208 positions, 96,246 characters, 10 separator targets — all three equal to the recorded values |
| character baseline | selected step **3800** and held-out **3.134278645735399** bits per character: **bit-for-bit equal to E2605's recorded numbers** |
| unigram and uniform baselines | **3.322429516434608** and **4.80103069218461**, bit-for-bit equal to the recorded values |
| per-step throughput | 120 steps in 45.33 s = **0.378 s/step** (E2605 measured 0.357 s/step), preparation 10.34 s, character baseline 62.46 s |
| projected full-budget run | 1,440 steps × 0.378 s + 62 s + 10 s ≈ **610 s per seed**, so five seeds ≈ **51 minutes** |
| output completeness | each run wrote its own manifest, tokenizer, resumable checkpoint (8,309,847 B), usable checkpoint (3,124,304 B) and `report.json` in its own directory |
| resume behaviour | a second pass over both completed reports took **3.3 s** and printed the identical aggregate; deleting seed 11's report and re-running rebuilt **only** that seed (3m42s) and returned the same aggregate, `transformer 2.9182 bits/char (sample stdev 0.0006), character 3.1156` |
| defect the resume test found | the first rebuild attempt was recorded as `failed`, not resumed: the interrupted pass had left a checkpoint and the run would not overwrite it. That is the ADR-0035 guard firing where it should not have, since the *report* is what makes a run recorded and a seed with no report has nothing to protect. Fixed before this study ran: a seed with no report may replace its own artifacts, while a seed that does have one still requires an explicit declaration to be rerun |
| environment block | `torch 2.13.0+cpu`, `numpy 2.4.0`, `python 3.13.5`, `device cpu`, `torch_threads 14` |
| study aggregate | `E2606P: 2/2 seeds, transformer 2.9182 bits/char (sample stdev 0.0006), character 3.1156`, report digest `86402f111ee5dacd…` |

Two things the pilot established that change how this record should be read.

First, **the non-Transformer parts of E2605's protocol reproduce exactly**. The character model, the unigram
baseline and the uniform baseline are numpy-only paths and they returned identical numbers to the last
decimal, from a snapshot recovered from history rather than from the files the run originally read. So the
reproduction question in this study is confined to the torch path and its reduction order, and any deviation
that appears will be attributable there rather than in the data.

Second, **the recorded environment is not comparable in one respect, and it is a deviation to state now
rather than explain later**: E2605's artifact block carries torch, python, platform and device only. This
study's runs additionally record `numpy` and `torch_threads`, because the same field was added between the
two. E2605's thread count is therefore unknown and cannot be compared; the declared tolerances above, not an
assumption of identical threads, are what decide whether a numerical difference is a reproduction.

## Training budget and stopping rule

Unchanged from E2605, and for the same reason: `max_steps 2400`, validation every 60 steps, early stopping
when eight consecutive evaluations fail to improve the best validation loss by `min_delta = 0.001` nats.
The reported model is the validation-selected checkpoint, never the last step, and the loop's incremental
selection is cross-checked against the pure `choose_validation_best` rule over the recorded curve — the
run refuses to return if the two disagree.

**Test data plays no part in selection.** Selection reads the validation side; the test side is scored once,
after the selected artifact exists. `train_transformer` receives no test examples, so there is no path from
the test documents to early stopping, hyperparameters, or which checkpoint is reported.

## Missing-run and failed-run policy

- A run that raises is recorded with `status: "failed"`, the exception type and its message, and its elapsed
  seconds. The study continues with the remaining seeds; nothing is deleted and no other run is disturbed.
- A failed run is **rerun only if the cause is infrastructure** (the process was killed, the disk filled,
  the machine rebooted) and only at the same seed and the same configuration. The rerun replaces nothing:
  both attempts are described in this record, and the aggregate uses the completed attempt.
- A run that completes with a bad score is **never** dropped. The aggregate includes all five seeds.
- Any seed with no completed run appears in `missing_seeds`, and every statistic states the number of runs it
  was computed from. No result is padded, interpolated, or replaced by another seed.
- A reused report is accepted only after its configuration, corpus fingerprint and seed are checked against
  this declaration; a mismatch stops the study rather than silently mixing protocols.

## Aggregation method

`chakaso.training.seed_study.build_study_report` recomputes everything from the per-run `report.json` files
on disk. For each metric: **n**, **mean**, **sample standard deviation** (`statistics.stdev`, n−1),
**median**, **min**, **max**, **range**. A single run reports no sample SD rather than zero.

**No confidence interval is published.** Five seeds give a descriptive spread of the seed-to-seed variation
of this metric under this protocol. The sampling distribution of held-out bits per character across seeds is
not established to be normal, the corpus is one author's prose, and a Student-t interval over n=5 would read
as an inferential claim about a wider population that this experiment does not sample from. The reported
uncertainty is the sample standard deviation and the observed range, labelled as such. E2605's number, and
any number in this record, must be quoted with the range it came from, not as a point.

Two independent recomputations are performed before this record is closed: the aggregate is rebuilt from the
per-run reports by `--collect-only`, and the same statistics are computed a second time directly over the
five reported values outside the study code. The two must agree; a disagreement is a defect, and it is
reported here rather than resolved silently.

## Planned outputs

- `data/experiments/E2606/study.json` — the aggregate, with its digest cited in this record.
- `data/experiments/E2606/E2606/seed-<n>/report.json` — one per seed, plus that seed's checkpoint.
- Committed to this directory: `results.json` (the aggregate) and the five per-run reports if their size
  allows under the artifact policy; the 824 KB corpus snapshot and the 3 MB checkpoints stay in `data/`.
- One table: seed × metric, every seed shown, no aggregation only.
- One table: E2605's recorded numbers against the seed-7 reproduction, with the tolerance and the verdict.
- A validation-curve summary per run (selected step, step at which stopping fired), not plotted curves: a
  five-curve plot of a 2,400-step toy run would illustrate nothing the tables do not already say.
- The three greedy continuations each run already generates, quoted as they come out. Repetition and looping
  are analysed by the generation evaluation suite described in `docs/evaluation.md`; that analysis is a
  separate section of this record and is not one of the metrics the interpretation rules use.

## Known limitations and threats to validity

- **The corpus is this project's own documentation.** One author, one domain, 780 KB, and the model trained
  partly on the record of the experiment before it. No claim about general language is available from this
  study at any level of confidence.
- **Five seeds is a small sample.** It supports a descriptive spread, not an inferential interval, and it
  cannot separate a rare catastrophic seed from a merely unlucky one.
- **The split is fixed by design**, so this study measures initialization and batch-order variance only.
  Split-to-split variance is a different experiment and is not covered here.
- **The architectures are not capacity-matched**; the asymmetry in the baselines section is real and is not
  corrected by this study.
- **Bits per character is computed over decoded text**, so it inherits the tokenizer's behaviour. It is
  chosen because it is the only cross-tokenizer metric available, not because it is a good metric of
  language quality — it is a compression number.
- **Same machine, same library versions.** Portability of these numbers to another CPU or a different torch
  build is untested; the environment block is recorded so a later reader can tell.
- **No human judgement of the generated text** is part of this study's protocol; the continuation samples
  are recorded, not rated.

## Interpretation rules, fixed in advance

- **Success:** all five seeds complete; the Transformer's mean bits per character is below the character
  model's mean and below both baselines; and it is below each of them on **every** seed.
- **Inconclusive:** one or more seeds fail for infrastructure reasons and cannot be rerun, or fewer than five
  complete. Reported with the runs that exist and the spread they show, not scaled up to five.
- **Failure of the hypothesis:** any seed on which the Transformer is *not* below the character model, or the
  sample standard deviation is larger than the Transformer-to-character mean gap. In that case E2605's result
  is restated in this record as a single draw, and the next experiment is about the corpus, not the model.

A negative result here is a valid outcome and will be written as one.

## Follow-up

Whichever way this lands, the next question is the one this study cannot answer: whether the advantage
survives on text that is not Chakaso's own documentation. That is **E2607**, on an independently licensed
corpus pinned by the same manifest machinery.
