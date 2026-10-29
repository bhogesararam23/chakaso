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
| E2611 decoder (capacity-matched) | 1,018,240 | 128 | 32 × 2 | reported by its own report | running at the time of writing | 19,660,800 |

The E2608 rates imply that pure gradient work was ~4,108 s of arm A's 6,024 s and ~1,638 s of arm B's
2,323 s; the remaining ~30–40% is evaluation and checkpoint overhead. Any wall-clock estimate below must add
that overhead, and the estimate is labelled.

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
`2 × 4 layers × 256 × 256 × 4 bytes ≈ 2.1 MB` for multi-head attention; at 30M parameters and context 1024
it is ~33 MB, and grouped-query attention reduces it further. The project's inference constraint is CPU
latency, not memory.

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
