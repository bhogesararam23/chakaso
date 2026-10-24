# E2606 — How much of E2605's result is the seed?

- Status: completed (measured; five of five declared seeds)
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
character model on the same held-out documents ([`../E2605/record.md`](../E2605/record.md)). It did that with
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
| manifest checksum | `3255f98d9bdface2…` ([`E2605/corpus-manifest.json`](../E2605/corpus-manifest.json)) |
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

**Runtime, measured rather than estimated.** E2605's full protocol took 513.6 s (Transformer, selected at
step 1380 of 2,400) + 69.9 s (character baseline) + 13.2 s (preparation) = **596.7 s per run** on this
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

## Results

**All five declared seeds completed. None failed, none was dropped, none was rerun.** The study process
started at commit `fc5f8ff` (the commit that contains this record) and ran to completion; two commits after it
changed how a run names its corpus records and added a generation statistic, neither of which altered these
runs' inputs — the identity fields below are the evidence for that, taken from the runs themselves.

Every run loaded corpus fingerprint `e45993f9badf706c` from snapshot checksum `84ecbc61cf1e634bbfd92d0bd…`
(recovered from revision `9b8a47876de7e13c60aefca0b9c7fe501bc03def`), and every run reported split fingerprint
`87067cca96275def`, tokenizer fingerprint `2605bf8f50183a6e`, 643,840 parameters, 46,208 scored positions,
96,246 scored characters and 10 separator targets. The aggregate rebuild refuses to run if any of those
disagrees across seeds; it did not have to.

| Seed | Transformer bits/char | nats/token | perplexity | token acc | selected step | character bits/char | gap to character | checkpoint checksum |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7 | 1.865734 | 2.693647 | 14.7855 | 0.450095 | 1380 | 3.134279 | 1.268545 | `262b01a5611f…` |
| 11 | 1.881652 | 2.716629 | 15.1292 | 0.447195 | 1320 | 3.096885 | 1.215233 | `bdbf8c5c072f…` |
| 13 | 1.882356 | 2.717645 | 15.1446 | 0.441331 | 1320 | 3.065195 | 1.182839 | `6414802f18ed…` |
| 17 | 1.884429 | 2.720638 | 15.1900 | 0.445204 | 1260 | 3.074580 | 1.190151 | `87079b5107d1…` |
| 19 | 1.882783 | 2.718261 | 15.1540 | 0.446200 | 1320 | 3.097405 | 1.214622 | `c152f3481b1d…` |

| Statistic (n=5) | Transformer bits/char | Character bits/char | Gap | Unigram | Uniform |
| --- | --- | --- | --- | --- | --- |
| mean | 1.879391 | 3.093669 | 1.214278 | 3.322430 | 4.801031 |
| sample standard deviation | 0.007702 | 0.026686 | 0.033602 | 0.0 | 0.0 |
| median | 1.882356 | 3.096885 | 1.214622 | 3.322430 | 4.801031 |
| min – max | 1.865734 – 1.884429 | 3.065195 – 3.134279 | 1.182839 – 1.268545 | 3.322430 | 4.801031 |
| range | 0.018695 | 0.069084 | 0.085706 | 0.0 | 0.0 |

Secondary metrics over the same five runs: nats/token mean 2.713364 (SD 0.011120, range 0.026991),
perplexity mean 15.080663 (SD 0.166506), token accuracy mean 0.446005 (SD 0.003189), selected step mean 1320
(SD 42.4, min 1260, max 1380). **Every run early-stopped**; none used its full 2,400-step budget.

Per-run wall clock: preparation 11.49–29.61 s, character baseline 102.19–151.23 s, Transformer
655.78–983.22 s (mean 890.11, range 327.44). These are **not comparable to E2605's recorded 513.63 s**: the
study ran while this session's own test suites and pilots competed for the same 18 logical CPUs, and every
run recorded `torch_threads: 14`. No timing claim is made here; the numbers are recorded so a later reader
can see that the wall clock is the noisiest quantity in the study by an order of magnitude (range/mean 37%
against 0.41% for the metric that matters).

The aggregate is `data/experiments/E2606/study.json`, digest
`3f8f0175c510ff680cf1772b2644fce9ec7ee16e16b577bb5e96f17364e52831`, rebuilt from the five per-run reports by
`chakaso experiment study … --collect-only`. The same statistics were then computed a second time, directly
over the five reported values, outside the study code (`statistics.fmean`/`stdev` on the values in the table
above) and agree with the aggregate on every field it reports.

## Reproduction of E2605 (seed 7, pre-declared tolerances)

**Verdict: reproduced, with zero deviation on every measured field.** The comparison is
`chakaso.training.seed_study.compare_with_reference` run against
[`../E2605/results.json`](../E2605/results.json), at the tolerances declared above:

| Field | Recorded (E2605) | Reproduced (E2606 seed 7) | Difference |
| --- | --- | --- | --- |
| corpus / split / tokenizer fingerprint | `e45993f9badf706c` / `87067cca96275def` / `2605bf8f50183a6e` | identical | — |
| architecture, parameter count | 9 fields, 643,840 | identical | 0 |
| selected step, early-stopped | 1380, true | 1380, true | — |
| held-out bits per character | 1.8657340418089126 | 1.8657340418089126 | **0.0** |
| held-out nats per token | 2.693647205504745 | 2.693647205504745 | **0.0** |
| perplexity | 14.785503470353591 | 14.785503470353591 | **0.0** |
| token accuracy | 0.4500952216066482 | 0.4500952216066482 | **0.0** |
| character model bits/char, nats/token, selected step | 3.134278645735399 / 2.1724045178920877 / 3800 | identical | **0.0** |
| unigram, uniform bits/char | 3.322429516434608 / 4.80103069218461 | identical | **0.0** |

The tolerances were not needed: the agreement is exact at full double precision, so the four reproduction
levels collapse into one. Artifact/provenance reproduction, training-procedure reproduction (33 recorded
evaluations, same schedule, same selected step) and numerical reproduction are all satisfied. **The
declared tolerance therefore played no part in this verdict** — which is worth stating, because a tolerance
that ends up deciding is a weaker claim than one that turns out to be unnecessary.

Two follow-ups were measured rather than assumed:

- **The weights themselves are identical, not merely their scores.** Loading both artifacts and comparing all
  41 stored tensors with `torch.equal` returns true, and a SHA-256 over the concatenated float32 bytes of the
  state dicts is `7e8407f7132d6097053f899a` for both.
- **The artifact checksums differ, and the cause is the metadata block, not the model.** E2605's usable
  artifact is `95c849b3cffa5cb0…`, this run's is `262b01a5611f5973…`. Between the two, the environment block
  gained `numpy` and `torch_threads` (commit `d2051ac`), and the checksum covers metadata. This is the one
  recorded deviation of the reproduction, it is a format-of-the-record change rather than a change in the
  trained object, and it is listed here instead of being smoothed over.
- **The generations are identical too.** The greedy continuations from the two checkpoints reproduce the same
  strings, including the same looping.

## Generation behaviour, measured over the same five checkpoints

`chakaso.training.generation_eval` was run over each seed's checkpoint at temperature 0 on five fixed
continuation prompts and four hostile inputs (`data/experiments/E2606/generation-E2606-seed-*.json`). All
five models behave the same way in kind, and the raw text of all 45 samples is retained in those files:

| Seed | samples with a periodic run of ≥3 (of 5) | tail loop detected | mean bigram repetition | samples that hit the length bound (of 5) |
| --- | --- | --- | --- | --- |
| 7 | 1 | 0 | 0.4870 | 4 |
| 11 | 4 | 0 | 0.6561 | 5 |
| 13 | 4 | 1 | 0.5255 | 5 |
| 17 | 5 | 1 | 0.4868 | 5 |
| 19 | 4 | 0 | 0.6904 | 5 |

Every seed emitted text like `' model that is a model that is a model that that is not a model to a model
that it.'` and `' model to model model and a model model and a model model to model s a'` — that is what
1.8794 bits per character looks like when read aloud. Four of the five seeds produce a sample that repeats
the same block at least three times back to back; on seed 17 every sample does. Four or five of five samples
per seed stop because the token budget ran out rather than because the model produced a separator: there is
no learned stop, so no sample here ended by deciding to stop.

Two things this measurement does **not** support, and the record says so because the numbers are tempting:
the repetition ratio is not a quality score (a heading-style artifact of this corpus inflates it and a short
sample deflates it), and the seed-to-seed differences in it are not ranked — four of five samples versus five
of five is not an effect, at n=5. The empty-prompt probe raises a typed `ModelError` naming the model and the
reason ("produced no input identifiers from the supplied text") on every seed; the boundary refuses, the
tokenizer does not crash, and non-ASCII input encodes.

## Interpretation

The hypothesis stands on all five seeds, in all three of its parts.

1. The Transformer beat the character model on **every** seed, by between 1.1828 and 1.2685 bits per character
   — never by less than a third of the character model's own distance from the unigram baseline.
2. It beat the unigram baseline on every seed (mean gap 1.4430 bits/char) and the uniform baseline on every
   seed (mean gap 2.9216). A model that does not beat a frequency count has learned nothing worth recording;
   this one clears it on every draw.
3. The spread across seeds is **small relative to the gap**: a sample standard deviation of 0.0077 bits/char
   for the Transformer against a mean gap of 1.2143 to the character model — the gap is roughly 158 times the
   seed-to-seed standard deviation, and the whole observed range (0.0187) is 2.4 SDs wide, i.e. the five runs
   sit inside a band one part in 65 of the measured difference.

So E2605's result was not the luck of one seed. It was, however, the luck of the *best* one: seed 7 produced
the lowest Transformer loss **and** the highest character-model loss of the five, which is why its gap
(1.2685) is the maximum and the honest figure to quote is the mean, **1.8794 against 3.0937 bits per
character, a gap of 1.2143 with a sample SD of 0.0336 across seeds**. Quoting 1.8657 without the spread
overstates the model by about one part in a thousand and the gap by 4.5%.

A second finding worth stating on its own: **the smaller model is the less stable one.** The character
architecture's spread across seeds (SD 0.0267, 0.86% of its mean) is 3.5× the Transformer's in absolute bits
and 2.1× in relative terms, and its selected step was 3800 on every seed while the Transformer's moved
between 1260 and 1380. E2601 and E2602 already showed the character model overfitting early; across seeds it
is the noisy one, which is consistent with a 44,084-parameter model with a 16-character window having less
structure to fall back on than 643,840 parameters with 128 tokens.

The early-stopping result replicates too: five for five runs stopped before their budget, selecting between
step 1260 and 1380 of 2400. A run that used its whole budget would have been reporting an overfit model, and
this is the fifth and sixth independent observation (E2601, E2602's five seeds, E2605, E2606) that the
un-stopped end of the curve is not the model you want.

What none of this touches is capability. The generations loop on every seed. The corpus is one author's
technical prose about the project itself, and the model has been shown to predict *it* better than a
frequency count predicts it. Bits per character is a compression measurement, and the tight spread says the
compression estimate is stable, not that what is being compressed is language in any general sense.

## Deviations from the preregistration

Three, none of which changes a measurement:

1. **Runtime was underestimated** because the pilot ran on a quieter machine. The projection was ~610 s per
   seed; measured runs took 655.78–983.22 s of Transformer time against E2605's recorded 513.63 s for the
   same work, with this session's own test suites and pilots competing for CPU. The declared budget, seeds and
   protocol were unchanged.
2. **The environment block grew between the two records** (`numpy` and `torch_threads` added by `d2051ac`),
   which is why the reproduced artifact's metadata checksum differs while its weights are bit-identical. The
   reproduction section above states this as the single recorded deviation rather than presenting identical
   checksums.
3. **The recorded run's code version is `fc5f8ff`, and the tree moved during the study.** Later commits changed
   corpus-record naming for snapshot runs and added the whole-output periodic-run statistic; neither affects a
   number in this record, and the identity fields each run reported are the evidence that the five runs share
   one protocol.

No seed was added, dropped, rerun or substituted. No tolerance was widened after the fact — every comparison
came out exact, so the declared tolerances were never the deciding criterion.

## Limitations

- **Five seeds, one machine, one corpus.** The spread described here is seed-to-seed variation of this
  protocol on this data. It is not a confidence statement about seeds in general, and no interval is claimed.
- **The split was held fixed by design**, so nothing here measures split-to-split variation. That would be a
  different study, and E2605's number and this one share a single 11-document test side.
- **Seeds 7 and 11 bracket the outcome asymmetrically**: 7 is the best Transformer draw and the worst
  character draw of the five, which is exactly the coincidence a single-seed result cannot see and a
  five-seed study can. Two more seeds would not have shown it.
- **Wall-clock numbers are contention-dominated** and should be read as "this is what it cost on a busy
  laptop", not as a performance figure.
- **The generations were not judged by a human.** Every statement above about them is a statement about
  repeated substrings, lengths and termination.

## Conclusion

Believed, at the strength the five runs support: **on this corpus, with the held-out documents fixed, the
subword Transformer's advantage over the character architecture and over both trivial baselines is real and
stable — mean held-out 1.8794 against 3.0937 bits per character, a gap of 1.2143 ± 0.0336 (sample SD across
five seeds), with every seed below every baseline.** E2605's single figure is confirmed as a reproducible
result, reproduced here bit-for-bit from a corpus recovered out of the repository's own history, and corrected
in emphasis: it was the most favourable of five draws, and the spread belongs beside it.

Also believed, and unchanged: the model loops, its text is not usable, and none of this measures language
ability on any material but this repository's own prose.

## Next experiment, from the measured result

The question this study cannot answer is the one E2607 asks: does the same protocol, on text nobody wrote
about this project, still beat the same baselines? If it does, the next decision after that is capacity, and
the budget that held here (2,400 steps) becomes the thing to vary — on the external corpus's 2.13× the data,
the same step count is proportionally fewer epochs, which E2607's record declares as a limitation rather than
a comparison. If it does not, then every number in this record describes how well a model learned
Chakaso's own documentation, and the next experiment is about the data, not the model.

## Corrections applied after the run (found 2026-10-22, committed 2026-10-24)

Two statements in this record were wrong against its own artifacts, and a documentation audit found them.
They are corrected in place with the original wording kept here, because a silent edit to a measured record is
exactly what the reproducibility rules exist to prevent.

| Where | Said | Correct | How it was caught |
| --- | --- | --- | --- |
| *Expected compute* | E2605 "stopped at step 1440 of 2,400" | E2605's **selected** step is 1380; 1440 was the curve's argmin, which lost to `min_delta` (E2605's own record, its *Training and selection* paragraph) | recomputed against `experiments/E2605/results.json` |
| *Interpretation* item 3 | "the whole observed range (0.0187) is 11 SDs wide" | 0.018695 / 0.007702 = **2.4 SDs** | arithmetic against `results.json` (now committed here, digest `3f8f0175c510ff68…`, the value this record already cited) |

The pilot's projected 610 s per seed was kept as written: it is labelled a projection, and *Results* reports
the measured 655.78–983.22 s beside it. The pilot's step basis (1,440) was the wrong expectation for a run that
early-stopped, which the measured wall clock in this record already discloses.
