# E2611 — the architecture question without the confound: a capacity- and context-matched decoder against E2608's transformer

**Status:** running — this file was committed before the arm ran
**Type:** Research experiment, **preregistered**
**Date preregistered:** 2026-10-29
**Decision record:** [ADR-0037](../../docs/decisions/ADR-0037-reference-model-sized-by-measured-throughput.md)
**Depends on:** E2608 (the control arm and its stated confound), E2610 (what decoding can and cannot explain),
ADR-0036 (a measured corpus stays re-trainable)

## Motivation

E2608 compared two Transformer-family architectures at a matched token budget and reported the rotary decoder at
**1.6865** bits per character against the transformer at **1.9043**. Its own record refuses the architecture
reading: the arms differ **3.3× in parameters** (3,413,760 against 1,037,056) and **2× in context** (256 against
128), and their learning-rate schedules differ (cosine to 0.1 against constant). What that experiment measured
is "the larger rotary decoder at equal tokens does better", not "the decoder architecture is better".

The project's roadmap names the missing experiment as Phase 3: a capacity-matched arm, so the difference is the
architecture rather than the scale. This record is that experiment. It is deliberately the cheapest arm that can
answer the question — one new run, one seed, one corpus, no new machinery — and it reuses the committed E2608
transformer report as its control instead of retraining an identical arm.

## What was known before this was written (disclosure)

The control arm's numbers are known, because the point of a controlled comparison is to compare against a
measured baseline: E2608's transformer report gives 1,904,315.1128 × 10⁻⁶ bits per character, 1,037,056
parameters, context 128, 2,400 selected steps (the ceiling), constant schedule, seed 7, corpus fingerprint
`e9fc2e92f59d88ea`, tokenizer `b597eb34f13d9058`, report digest `fd6af2c8e6e3ec7f…`.

The new arm has **not** been run in any form, exploratory or recorded. Its parameter count is arithmetic over
the configuration, not a measurement (the run will report the count from the built model as well). E2610's
decoding diagnosis is known and motivates the generation hypothesis below; it says nothing about this arm's
held-out loss.

## Hypotheses, stated before the run

The treatment is a decoder with the **same width, depth, heads, feed-forward width, vocabulary, context, batch
shape and token budget** as the control transformer. The two designs differ in their architecture bundle —
rotary positions rather than a learned position table, RMSNorm rather than LayerNorm, bias-free q/k/v/o rather
than the fused biased projection, and the two implementations' feed-forward details — not in scale or context.

- **H1 (the confound was the explanation).** At matched capacity and context, the decoder's held-out bits per
  character differs from the transformer's 1.9043 by **less than 0.02 bpc**. If so, E2608's 0.2178-bit gap is
  consistent with parameters, context and schedule rather than with the architecture bundle, and the project
  must stop citing it as an architecture result.
- **H2 (the direction survives the match).** The decoder is lower by **at least 0.02 bpc**, so an architecture
  effect remains at this scale. This is the alternative H1 is written to fail into; it is not the hoped outcome,
  and a failure of H1 in this direction would be the stronger finding.
- **H3 (the reverse).** The decoder is **higher by at least 0.02 bpc**, so whatever helped the larger decoder is
  scale/context, not the design.

H1, H2 and H3 are mutually exclusive and exhaustive at the declared threshold. The threshold **0.02 bpc** is a
declared materiality band, not a confidence interval: it is about 2.6× the only seed standard deviation this
project has measured for a Transformer-family arm (0.0077, E2606, on a different corpus and a different
architecture), and it is small against E2608's 0.2178.

- **H4 (budget).** The 1.018M-parameter decoder will also select the 2,400-step ceiling, as every
  1M-parameter-family run on this corpus has. If it early-stops instead, capacity rather than budget was the
  binding constraint for the control arm, and that is a result about the *control*, not a decoration of this
  arm.
- **H5 (generation).** Its greedy continuations will remain non-coherent and repetitive on the surface
  statistics the project measures. No mechanism here is designed to fix degeneration, and E2610 already showed
  that decoding settings do not repair a model of this scale. A lower loss with still-degenerate samples is
  the expected outcome, not a contradiction.

**What would count against a hypothesis:** any result outside the declared band is reported as the observed
direction and magnitude, whether or not it was expected. The band is not widened after the number is read.

## Control arm

E2608 arm B, as committed in [`../E2608/record.md`](../E2608/record.md) and its report
`data/experiments/E2608/transformer/report.json` (digest `fd6af2c8e6e3ec7f…`). The training and scoring modules
(`nn_training.py`, `nn_evaluation.py`, `decoder.py`, `transformer.py`, `nn_checkpoint.py`) are **unchanged**
between the control's code revision `260ab8d` and this record's commit; `subword_experiment.py` gained report
wiring for generation, latency and peak memory after the control ran, but not a change to the training or
scoring path. The control is therefore a valid comparator, and the treatment's report will carry the newer
blocks while the control's does not — a shape difference stated here rather than presented as a measurement
difference.

## Dataset, licence and provenance

Identical to E2608, and read from the same pinned snapshot.

| Field | Value |
| --- | --- |
| name / version | `gutenberg-english-prose-v2` / `1.0.0` |
| sources | 24 Project Gutenberg texts, declared in [`configs/corpora/gutenberg-english-prose-v2.toml`](../../configs/corpora/gutenberg-english-prose-v2.toml) |
| licence | Public domain in the United States, distributed under the Project Gutenberg License; the licence statement travels in the catalogue and the data card |
| documents / characters | 919 documents, 14,490,005 characters |
| corpus fingerprint | `e9fc2e92f59d88ea` |
| snapshot | `data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json`, checksum `8eacb49f8d244e65db63f80796d76a026f532252899b8f85c7631ac3b9a2415f` |
| verification | `chakaso corpus verify --manifest data/experiments/E2608/transformer/gutenberg-english-prose-v2-manifest.json --snapshot …` reports 919 documents / 14,490,005 characters at the fingerprint above, before this record was committed |
| split | document-granular three-way split, validation 0.15, test 0.15, **split seed 7**, the control's split |
| tokenizer | byte-pair, 4,096 identifiers, `min_frequency` 3, fitted on the training side only, fingerprint `b597eb34f13d9058`, reused through the control's cache |

No new data is introduced. Nothing here changes the corpus's duplicate screen, licence or provenance.

## Model and configuration — matched in capacity and context

| Field | Treatment: decoder | Control: transformer (E2608 arm B) |
| --- | --- | --- |
| family / schema | `decoder` / `chakaso-decoder-checkpoint/1` | `transformer` / `chakaso-transformer-checkpoint/1` |
| position encoding | rotary, `rope_theta` 10,000 | learned table, 128 rows |
| normalization | RMSNorm, pre-norm | LayerNorm, pre-norm |
| attention | explicit q/k/v/o, no biases, KV cache | fused biased input projection |
| embeddings | tied | tied |
| `d_model` / layers / heads / ff | **128 / 3 / 4 / 384** | **128 / 3 / 4 / 384** |
| context length | **128** | **128** |
| vocabulary | 4,096 | 4,096 |
| parameters, declared from configuration | **1,018,240** | **1,037,056** |
| parameters, reported by the built model | recorded in the results | 1,037,056 |
| batch × accumulation | **32 × 2** | **32 × 2** |
| positions per optimizer step | 8,256 (block width 129) | 8,256 |
| steps | 2,400 | 2,400 |
| total training positions | 19,814,400 | 19,814,400 |
| schedule | **constant** (the control's) | constant |
| seed | 7 | 7 |

The parameter counts differ by **18,816 (1.8%)**, which is almost entirely the control's learned position table
(128 × 128 = 16,384) and its LayerNorm parameters. The match is not exact, and is stated as 98.2% rather than
as "equal": a reader who requires an exact count should read this as a capacity-matched comparison with a
1.8% residual, not as an exact one. The feed-forward expansion ratio is identical (3× `d_model`), so no shape
knob was traded to reach the count.

Both arms: AdamW, learning rate 6e-4, weight decay 0.01, warmup 200, gradient clipping 1.0, `eval_every` 100,
`patience` 3, `min_delta` 0.0005, `max_steps` 2,400, float32 on CPU.

The treatment uses the **constant** schedule because the control does. E2608's arm A used cosine; reproducing
that schedule here would confound the architecture comparison with the schedule difference E2608 already
stated. The cost is that this arm is not a reproduction of arm A and cannot be compared to it as one.

## Environment

Measured at the start of this session, 2026-10-29: Windows 11, CPython 3.13.5 in `.venv-nn`, `torch
2.13.0+cpu`, `numpy 2.4.0`, 18 logical CPUs, **no CUDA device**, 15.43 GB RAM total with 0.68 GB available
(95% load), 154.1 GB free disk. The machine is shared with the author's session, so wall-clock and throughput
are reported as observed conditions, not as intrinsic rates.

## Commands, exactly as they will be run

```bash
.venv-nn/Scripts/python.exe -m chakaso corpus verify \
  --manifest data/experiments/E2608/transformer/gutenberg-english-prose-v2-manifest.json \
  --snapshot data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json

.venv-nn/Scripts/python.exe -m chakaso experiment model \
  --root . --experiment E2611 \
  --corpus-snapshot data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json \
  --expected-corpus-fingerprint e9fc2e92f59d88ea \
  --tokenizer-cache-dir data/experiments/tokenizers \
  --tokenizer-min-frequency 3 --vocab-size 4096 \
  --no-character-baseline --sample-prompts held_out --samples-per-prompt 64 \
  --batch-size 32 --learning-rate 0.0006 --weight-decay 0.01 \
  --warmup-steps 200 --eval-every 100 --max-steps 2400 \
  --patience 3 --min-delta 0.0005 --seed 7 --split-seed 7 \
  --model-family decoder --context-length 128 --d-model 128 --num-layers 3 --num-heads 4 \
  --feedforward-dim 384 --rope-theta 10000 --grad-accum-steps 2 \
  --schedule constant \
  --work-dir data/experiments/E2611/decoder-small \
  --write-report data/experiments/E2611/decoder-small/report.json \
  --verbose
```

The verify command is run before the training command; its output is quoted in the results. Reports and
checkpoints are written under git-ignored `data/`; the record cites their digests.

## Metrics and decision rule

- **Primary:** `transformer.test.bits_per_character` from the treatment report, compared with the control's
  1.9043151128. The test side is read once, after selection, as in every experiment here.
- **Secondary:** validation bits per character at the selected step, the validation curve, selected step and
  completion class, positions per second, wall clock, peak resident memory, the report's own generation
  statistics, and the parameter count reported by the built model.
- **Decision rule, fixed now:** if |Δ| < 0.02 bpc, H1 holds and the project records that no architecture
  advantage is measurable at matched capacity and context on this corpus and budget; E2608's gap may not be
  cited as an architecture result. If Δ ≤ −0.02 bpc (decoder lower), H2 holds and the decoder path is the
  candidate for the next seed spread. If Δ ≥ +0.02 bpc (decoder higher), H3 holds and ADR-0037's architecture
  choice is reopened on the result. In every case the direction and magnitude are reported, and no threshold is
  changed after the run.
- **H4 and H5** are checked from the report's `best_step`, `completion`, `curve` and `generation` blocks.

## Failure and missing-run policy

A run that raises is reported with its error and not aggregated. The report writer refuses to overwrite an
existing report; the training artifact's overwrite guard is left off. If the run is interrupted, the partial
resumable checkpoint stays on disk and is named in the results; finishing it under a changed budget is a new
experiment id, not an edit to this one. If the run cannot be completed in the session, this record states that
and the hypothesis remains untested rather than becoming a conclusion.

## Limitations, declared before the result

1. **One seed.** No spread exists for either arm at this scale. The 0.02 band is a declared materiality
   threshold, not a confidence interval, and a one-seed difference inside it is not evidence of no effect in
   general.
2. **One corpus, one register.** 19th- and early-20th-century English narrative prose. Nothing here speaks to
   other data.
3. **Both arms may be truncated.** The control selected the ceiling; if the treatment does too, both numbers
   are floors and the comparison is of two truncated models at equal tokens.
4. **1.8% parameter residual** and a different position/normalization/attention bundle. The architecture effect
   is a bundle effect; this experiment cannot separate RoPE from RMSNorm from the bias-free attention
   implementation.
5. **The control is a committed report, not a re-run.** Its code path for training and scoring is unchanged,
   but it was produced in a different session on the same machine. The report's environment block is quoted in
   the results.
6. **Surface generation statistics are not coherence.** H5 is about repetition, not meaning; Level 3 of
   ADR-0039 remains withheld whatever this run measures.
