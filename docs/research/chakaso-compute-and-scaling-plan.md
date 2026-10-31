# Chakaso compute and scaling plan: what this machine can train, and what it cannot

**Status:** Research and proposal. The measured rows below are from committed reports; every row labelled
*estimate* is arithmetic over those measurements or a labelled assumption, never a measurement.
**Date:** 2026-10-29.
**Companions:** [`chakaso-next-training-design.md`](chakaso-next-training-design.md),
[`chakaso-data-curriculum.md`](chakaso-data-curriculum.md).

## The measured machine

| Quantity | Value | Source |
| --- | --- | --- |
| OS | Windows 11 (10.0.26200), x64 | read from the OS, 2026-10-29 |
| logical CPUs | 18 | `os.cpu_count()` |
| CUDA device | none | `torch.cuda.is_available()` |
| RAM | 15.43 GB total, **0.68 GB available, 95% load** at inspection | `GlobalMemoryStatusEx` |
| free disk | 154.1 GB | `shutil.disk_usage` |
| Python | 3.13.5 | `.venv-nn` |
| torch | 2.13.0+cpu, 14 intra-op threads | import |
| numpy | 2.4.0 | import |

The RAM figure is the binding constraint, and it is transient: the machine is shared with the author's
session. A training run that needs 2 GB of headroom should not be started when 0.68 GB is free.

## Measured training throughput

All numbers are from committed experiment reports on this CPU, not from a benchmark harness. "Positions/s"
is the report's own rate over gradient time only; wall clock additionally includes validation, checkpointing
and the run's post-training evaluation.

| Run | Parameters | Context | Batch | Positions/s | Wall clock | Tokens seen |
| --- | --- | --- | --- | --- | --- | --- |
| E2605 Transformer | 643,840 | 128 | 32 | not reported in the same field | 513.6 s | ~2,400 steps × 4,128 |
| E2608 arm A decoder | 3,413,760 | 256 | 32 × 1 | **4,821.2** | 6,024 s | 19,660,800 |
| E2608 arm B transformer | 1,037,056 | 128 | 32 × 2 | **12,098.1** | 2,322.9 s | 19,660,800 |
| E2611 decoder (capacity-matched) | 1,018,240 | 128 | 32 × 2 | 5,363.1 | 5,169.4 s | 19,660,800 |
| E2612 smoke (multi-register corpus) | 2,699,264 | 256 | 32 × 1 | **4,493.7** | 169.05 s (60 steps) | 491,520 |

The E2608 rates imply that pure gradient work was ~4,108 s of arm A's 6,024 s and ~1,638 s of arm B's
2,323 s; the remaining ~30–40% is evaluation and checkpoint overhead. Any wall-clock estimate below must add
that overhead, and the estimate is labelled.

The E2612 smoke row is the only one measured on the multi-register corpus, and it is a rate measured with the
machine also running the operator's session: 60 steps at 4,493.7 positions/s and 1,710 MB peak resident set,
with preparation (tokenizer fit on a 12M-character sample plus packing) costing 70.4 s. That vocabulary was
1,024 identifiers rather than the 4,096 the arms declare, so its parameter count is smaller; the E2612 arm
reports are the rate to use for a 3.4M model on this corpus.

## What the arithmetic says about model size

Chinchilla's compute-optimal point is roughly **20 tokens per parameter** (Hoffmann et al., 2022). The
current corpus is ~6.72M identifiers (E2608 report). The following table is arithmetic, not a measurement:

| Parameters | Tokens at 20/param | Epochs of the 6.72M-token corpus | Wall clock at 5,000 positions/s, +35% overhead | Wall clock at 10,000 positions/s, +35% |
| --- | --- | --- | --- | --- |
| 3.4M | 68M | ~10 | 5.1 h | 2.6 h |
| 5M | 100M | ~15 | 7.5 h | 3.8 h |
| 10M | 200M | ~30 | 15.0 h | 7.5 h |
| 30M | 600M | ~90 | 45.0 h | 22.5 h |
| 100M | 2B | ~300 | 150 h | 75 h |

The epoch column is the reason the current corpus is not a scaling path: beyond roughly four epochs,
repeated data has diminishing value (Muennighoff et al., 2023), and a 10M model would need ~30 epochs.
**A 5–10M model is the largest scale for which a new 50–150M-token corpus would be a genuine data-to-parameter
match; a 30M model needs a ~600M-token corpus and days of CPU; a 100M model is a different project.**

## Memory arithmetic

Training memory is dominated by parameters, gradients and AdamW moments. In float32:

- weights: `4P` bytes;
- gradients: `4P` bytes;
- AdamW first and second moments: `8P` bytes;
- total state before activations: **`16P` bytes**.

At 5M parameters that is 80 MB; at 10M, 160 MB; at 30M, 480 MB; at 100M, 1.6 GB. Activations add a
batch-and-context term. A rough estimate for a pre-norm Transformer is
`B × T × d_model × layers × ~20 float32 values`; at batch 32, context 256, `d_model` 256, 4 layers that is
~335 MB, and at batch 32, context 512, `d_model` 512, 6 layers it is ~4.0 GB. These are estimates, not
measurements; the arithmetic is shown so one can be checked.

A CPU run with 15.4 GB total RAM and less than 1 GB free at inspection cannot assume a 4 GB activation
budget. The practical ceilings here are a ~10M-parameter model at context 256–512 with a small batch, or a
~30M model at a short context and gradient accumulation. Anything larger needs a machine with free RAM
before it needs a GPU.

Inference memory is much smaller: a 3.4M-parameter decoder at context 256 has a KV cache of
`2 × 4 layers × 256 × 256 × 4 bytes ≈ 2.1 MB` for multi-head attention; a 30M-parameter, 6-layer,
`d_model` 512 model at context 1024 is ~24 MB, and grouped-query attention reduces it further. The project's
inference constraint is CPU latency, not memory.

## Storage

| Artifact | Estimate |
| --- | --- |
| Raw corpus, 1B tokens at ~2.15 characters/token | ~2.15 GB of text |
| Tokenized identifiers, 4 bytes each | ~4 GB |
| Packed input+target pairs, 8 bytes per position | ~8 GB |
| Resumable checkpoint, 3.4M parameters | ~45 MB (measured) |
| Resumable checkpoint, 10M / 30M / 100M | ~160 / 480 MB / 1.6 GB (arithmetic) |
| Release artifact, usable checkpoint | ~18 MB for 3.4M; ~40 / 120 / 400 MB at 10M / 30M / 100M |

Free disk is 154 GB, so storage is not the constraint at these sizes. The artifact policy is: bytes stay out
of Git, the catalogue and digests are committed, and a release is rebuilt from the record.

## Wall-clock conditions, stated honestly

Every timing here comes from a machine that was also running the author's session. E2608's arm A took 100.4
minutes and arm B 38.7 minutes on the same corpus and budget; the ratio is not only the model's size, it is
also the decoder implementation and the load. E2606's five runs ranged 655.78–983.22 s for the same
architecture, a range/mean of 37% against 0.41% for the metric, so **the clock is the noisiest quantity in
this repository**. A wall-clock plan is a planning aid, not a promise.

## Experiment ladder and stop conditions

Each rung is a new preregistered record. No rung is entered because the previous one "looks good"; it is
entered because the previous record's result justifies it.

| Rung | Question | Cost on this machine | Stop condition |
| --- | --- | --- | --- |
| E2611 | Is the E2608 architecture gap present at matched capacity/context? | ~1 new 2,400-step arm, ~100 min | If the result is inside the declared ±0.02 band, stop citing an architecture effect; move to data |
| E2612 (proposed) | Does an independent model assign lower likelihood to human continuations than to the released model's? | Scoring only, minutes | If the judge prefers the model's text, the instrument is invalid; report and stop |
| Data arm A | Does a technical mixture improve technical held-out loss without damaging prose loss, at 5M parameters and ~100M tokens? | ~8 h per arm at 5k positions/s; two arms ~16 h | If validation loss is still falling at the ceiling, the arm is a floor; do not compare levels as converged |
| Data arm B | Does the same mixture help at 10M parameters and ~200M tokens? | ~15 h per arm | If the 5M result is a regression on prose, do not spend the 10M budget |
| Decoder spread (E2609) | What is the seed spread of the decoder family at 3.4M? | 5 × ~100 min ≈ 8.3 h | If the spread is larger than the difference being claimed, the claim is not established |
| Instruction stage | Can a measured SFT stage improve instruction following without degrading language modelling? | not before a base checkpoint exists | If the base is not measurably better than baselines on a held-out task, SFT is premature |

**Hard stop conditions for any run:** non-finite loss or gradient norm above the declared clip for several
consecutive steps; free RAM below a declared floor; wall clock beyond the preregistered budget; a data-drift
or contamination check failure; or a report that would overwrite a measured run's artifact. A stop is
recorded as a completion class, not edited out.

## What an accelerator would change, and what it would not

No GPU is available here, and no GPU run is measured. A rented A100/T4 would reduce the wall clock, not the
data requirement; it would also add a cost and a dependency that the project's local-first runtime does not
have. Training on a rented accelerator is compatible with ADR-0001 (the runtime stays local), but it needs
its own record: hardware identity, cost, software versions, and a reproducible environment. The estimate
"a 10× faster device makes a 15-hour CPU run a 1.5-hour run" is arithmetic, not a measurement, and should
not be used to promise a capability result.

## What this plan does not claim

It does not claim that a 10M-parameter model on 200M tokens will be useful. The literature's small models
are 1–3B parameters and 3–11T tokens; a 10M model is a *learning-loop* scale, not a capability scale. The
plan's purpose is to make the next measurement affordable and honest: enough data that repetition is not the
binding constraint, enough tokens that the curve can turn, and a report that states what was actually spent.

## Hardware and compute feasibility, measured 2026-11-01

The E2612 arms are the first runs whose cost was recorded end to end on this machine rather than projected from
a smoke. Numbers are from the two reports on disk; the arithmetic that uses them is labelled as derived.

| Measured | Arm P (prose) | Arm T (mixture) |
| --- | ---: | ---: |
| Training throughput | 3,851.8 positions/s | 3,976.1 positions/s |
| Wall clock, 2,400 steps including evaluation | 5,426.02 s (90.4 min) | 5,246.90 s (87.4 min) |
| Preparation (tokenizer fit + packing) | 145.55 s cold | 57.02 s on a warm cache |
| Peak resident set | 2,040.1 MB | 2,560.8 MB |
| Identifiers packed for training | 20,000,402 | 20,000,103 |
| Greedy generation | 59.61 characters/s | 78.34 characters/s |

**C-01 — Result.** The smoke's projection was optimistic by 11–14 %. The record predicted ~73 min per arm from
4,493.7 positions/s; the arms reported 3,851.8 and 3,976.1 and took 90.4 and 87.4 min. A throughput measured on
a 60-step smoke with a 1,024-identifier vocabulary is not a rate, and planning against it costs 20 % of every
subsequent estimate. Use the arm-level number.

**C-02 — Observation.** Peak memory, not speed, is the tighter constraint on this host. 16.6 GB total with
0.68 GB available at the start of the campaign, and one 3.4M arm peaked at 2,560.8 MB. Anything that multiplies
the packed corpus in RAM — a larger tokenizer sample, a wider evaluation pack, several arms resident at once —
bumps against the ceiling before it bumps against the clock.

**C-03 — Result, and the campaign's most expensive surprise.** Exact byte-pair fitting is not free, and the
number is bigger than expected: on the multi-register corpus where nearly every pre-tokenized piece is unique,
1,002,483 characters fit in **6.6 s** and 4,014,828 characters in **494 s** — against the 29.2 s at 4.0M that
the optimising commit measured. On the prose corpus, 10,129,113 characters fit in 25.6 s against 12.3 s. The
heap may no longer discard work, so a tokenizer sample sized for the fast-but-wrong loop is a different budget
decision now. See the timing table below.

**C-04 — Result, and the number that changes the plan.** The curve is superlinear and it was measured to the end:
1.0M characters in 6.6 s, 4.0M in 494 s, and the **12,663,402-character E2612 sample in 2,116.4 s — 35.3 minutes**
for 3,839 merges. E2612 itself reported 145.55 s of *total* preparation, tokenizer fit and packing together. So
exact fitting costs roughly **24× more preparation** than the run that produced the current record spent on all of
it, and a tokenizer sample is now a budgeted experimental parameter rather than a default. Three mitigations are
available and each trades something different: shrink the sample (the vocabulary is a property of the sample, so
this changes comparability across records), fit once and cache it (which makes a git-ignored file load-bearing for
reproducibility — [K-024](../agent/KNOWN_ISSUES.md)), or hold the vocabulary fixed across a whole experiment
series so the fit is paid once. The third is the only one that costs no methodological clarity, and it should be
the default for any multi-arm programme.

**C-05 — Result.** An arm is ~90 minutes of CPU and ~2.5 GB of RAM at 3.4M parameters, 256 context and a
20M-token budget. **Three arms — the E2613 exposure control — is ~4.5 hours, affordable in one evening, with no
new hardware.** This is the cheapest place in the whole plan to buy a real answer, and it is the campaign's
first recommendation for exactly that reason.

**C-06 — Observation.** At the measured 3,900 positions/s, the ~9.84M-parameter step in
[`chakaso-next-training-design.md`](chakaso-next-training-design.md) is not a 2× cost: parameters grow ~2.9×,
and the same token budget with a wider model means fewer steps per second per parameter. Any capacity step
should be preceded by a short throughput measurement at the new configuration, because every wall-clock figure
in this repository has so far been inherited from a different model size.

**C-07 — Result, and a limit no CPU workaround removes.** No amount of scheduling makes 200M+ tokens affordable
here. At 3,900 positions/s a 200M-token run is ~14.3 hours of continuous CPU *before* evaluation, and the
data review found no licence-clean corpus at that size that this project could prepare in RAM. GPU access is
therefore a **data-quantity** requirement, not a speed preference: the point of a GPU is to afford tokens, not to
finish sooner. Until then the honest plan stays inside the 20–80M-token band.

**C-08 — Interpretation, and the trap in the current numbers.** Every trained artifact so far is truncated:
E2608, E2611, E2612-P and E2612-T all completed `budget_ceiling` with `stopped_early: false` and validation
still falling at the last evaluation. That means the repository's *absolute* bits-per-character values are floors,
and comparisons between two truncated runs at equal tokens are still the most valid comparisons available. It
also means "extend the budget" is a measurement, not a fix — and E2612's own arms show why: extending budget on a
fixed pool means repeating it.

**C-09 — Result, and the constraint that caps C-08.** Muennighoff et al. (arXiv:2305.16264) report that
"training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique
data". Chakaso's arms each packed 20,000,000 identifiers and trained 19,660,800 positions — **0.98 epochs of its
own pool**. Reaching ~20 tokens per parameter by adding steps therefore means ~4 epochs of a fixed pool, which
the literature says buys little. Xue et al. (arXiv:2305.13230) name dataset size and parameter count as the
drivers of multi-epoch degradation, so tolerance is plausibly *lower* at 3.4M, not higher. More steps is not a
substitute for more data; it is a way of measuring how much repetition costs.

**C-10 — Observation.** The ~20 tokens-per-parameter figure has no authority at this scale and should stop
being quoted as a target. It is arithmetic on Hoffmann's own Table 3 (400M→8.0B; 1B→20.2B — the literal phrase
"tokens per parameter" does not appear in the text read), re-derived as ≈20 by Besiroglu et al.
(arXiv:2404.10102) while disputing Hoffmann's confidence intervals, and contested by Schaeffer et al.
(arXiv:2509.23963) as partly a unit-convention artefact. Chakaso is more than 20× below the smallest model in
every one, and Hoffmann writes that "projections from very small models lead to different predictions than those
from larger models". **Use the direction; do not use the number.**

**C-11 — Result, and the one source at genuinely this scale.** Shi et al. (arXiv:2502.01481) ran a six-layer
transformer over 12M/25M/50M-token slices and conclude that "deploying expansive context windows on constrained
datasets guarantees suboptimal convergence". At context 256 on a 20M-token budget, Chakaso's context may be
**too long rather than too short** — the opposite of the intuition the roadmap had. It is the most
scale-appropriate source found in the campaign and it argues for a context-arm experiment, not a context
increase.

**C-12 — Result, and a schedule rule that is actually size-relative.** Porian et al. (arXiv:2406.19146;
900+ runs, 55M–901M parameters) find that a long fixed warmup leaves "models suboptimally trained" and give the
rule "set the number of warmup tokens to be identical to the model size N". Chakaso warms up for 200 steps ×
8,192 positions = **1.64M tokens ≈ 0.48 N**, so its warmup is already near the recommended side. Wen et al.
(arXiv:2410.05192) place the decay at ~10 % of steps, robust between 8 and 12 %. Both are cheap to test because
they change no other variable — and E2612's still-descending curves are the symptom this class of result
addresses.
