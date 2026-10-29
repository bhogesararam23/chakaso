# E2611 — the architecture question without the confound: a capacity- and context-matched decoder against E2608's transformer

**Status:** completed 2026-10-29 — preregistered in `e957167` before the arm ran; results below
**Type:** Research experiment, **preregistered**
**Date preregistered:** 2026-10-29
**Code that ran the arm:** the working tree at `a2416d1` (started 05:16 local, report 06:39). The
document-boundary masking feature (ADR-0040) landed while the arm was running and was not imported by the
process; the report has no `document_masking` field and the arm trained with full causal attention, which is
the protocol this record declared. The control arm's training and scoring modules are unchanged between
`260ab8d` and `a2416d1`; only the report wiring differs.
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
| model input positions per optimizer step | 8,192 (32 × 2 × 128) | 8,192 |
| steps | 2,400 | 2,400 |
| total model input positions | 19,660,800 | 19,660,800 |
| schedule | **constant** (the control's) | constant |
| seed | 7 | 7 |

The batch shape and step count are the control's exactly. (The control's table also quotes 8,256 *packed
stream* tokens per step, because a block is 129 identifiers wide and the last one is a target rather than an
input. The model processes 8,192 input positions per step; 19,660,800 is the control's recorded total and the
number this record matches.)

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

## Results

All runs: Windows 11, Python 3.13.5, `torch 2.13.0+cpu`, numpy 2.4.0, CPU only, 18 logical CPUs, no CUDA
device. The machine was shared with the author's session; wall clock is an observed condition, not an
intrinsic rate. Artifacts stay under git-ignored `data/`; the figures below are the run's own report.

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Treatment report | `data/experiments/E2611/decoder-small/report.json` | `bdcb49862787eaf01df6fd5383bb11194057c99213b6b4db6ac5341af6c11e88` |
| Treatment checkpoint | `data/experiments/E2611/decoder-small/E2611-decoder.pt` | `27bfb69fc54ad661e6b77972b6e86533834d6af670fc1ecd43fdf2a8b149e20f` |
| Quality suite (secondary) | `data/experiments/E2611/quality.json` | `d77ecfc9279245d58cee308262a5a7d4cd5ec60fd8809d53f53bad8824f653bc` |
| Aggregate committed beside this record | [`results.json`](results.json) | `414fdb1c9249520d4b47cbca9afc172d0845a2f2c4c51da100b5634dead2060b` |

The aggregate contains both arms' test and validation blocks, both curves, the treatment's generation and
resource blocks, and each source report's SHA-256, so the comparison can be re-derived without the
git-ignored tree.

### Held-out result

| | Treatment: decoder | Control: transformer (E2608 arm B) | Difference |
| --- | --- | --- | --- |
| Parameters | 1,018,240 | 1,037,056 | −18,816 (−1.8%) |
| Context / vocabulary | 128 / 4,096 | 128 / 4,096 | matched |
| Corpus fingerprint / split seed | `e9fc2e92f59d88ea` / 7 | same | matched |
| Tokenizer fingerprint | `b597eb34f13d9058` | same | matched |
| Seed / schedule / batch | 7 / constant / 32 × 2 | same | matched |
| **Test bits per character** | **1.8077426492** | **1.9043151128** | **−0.0965725 (−5.071%)** |
| Test nats per token | 2.7011719645 | 2.8454728313 | −0.1443009 |
| Test perplexity | 14.8972 | 17.2097 | −2.3125 |
| Test token accuracy | 0.479679 | 0.461311 | +0.018368 |
| Test positions / characters scored | 996,608 / 2,148,397 | same | matched |
| Validation bits per character | 1.8348230548 | 1.9356927020 | −0.1008696 |
| Uniform / unigram bits per character | 5.566614 / 3.226482 | 5.5666 / 3.2265 | matched |
| Selected step | 2,400 (ceiling) | 2,400 (ceiling) | both truncated |
| Completion class | `budget_ceiling` | `budget_ceiling` | both truncated |
| Throughput / wall clock | 5,363.1 positions/s / 5,169.44 s | 12,098.1 positions/s / 2,322.89 s | decoder slower |
| Peak resident memory | 1,062.3 MB | not reported in the control's shape | — |

The validation comparison is the cleaner curve because both arms evaluated at the same steps:

| Step | Treatment validation nats/token | Control validation nats/token | Treatment train | Control train |
| --- | --- | --- | --- | --- |
| 0 | 8.3507 | 8.3496 | 8.3430 | 8.3471 |
| 600 | 3.2255 | 3.5405 | 3.1780 | 3.4903 |
| 1,200 | 2.9297 | 3.1690 | 2.8792 | 3.1144 |
| 1,800 | 2.7996 | 2.9809 | 2.7322 | 2.9425 |
| 2,400 | **2.7290** | **2.8790** | 2.6450 | 2.7935 |

(Validation bits per character in the treatment report is 1.8348230548; the control report's validation block
is in [`results.json`](results.json). The control's test block in the table is the committed
`data/experiments/E2608/transformer/report.json`.)

### Verdict on the declared hypotheses

- **H1 (within ±0.02): false.** The difference is −0.0966 bits per character, about 4.8× the declared band.
- **H2 (decoder materially lower): true.** The treatment is lower by 0.0966 bits per character, 5.1% relative.
- **H3 (decoder materially higher): false.**
- **H4 (budget): true.** The smaller decoder also selected step 2,400, the ceiling, with validation still
  falling (2.7342 → 2.7290 over the final 100 steps). Capacity did not turn the curve inside this budget;
  both arms are floors.
- **H5 (generation): true on the surface statistics.** The run's `generation` block reports five greedy
  samples, **all bound-reached**, mean bigram repetition **0.654745**, zero tail loops and zero periodic runs,
  one refusal, zero crashes. The secondary calibrated suite (`chakaso model quality`, default configuration
  set, 24 prompts drawn from the same held-out documents) measures word distinct-1/2/3:

| Configuration | distinct-1 | distinct-2 | distinct-3 | Loops | Periodic | Gates |
| --- | --- | --- | --- | --- | --- | --- |
| human continuation | 0.853 | 0.969 | 0.982 | — | — | reference |
| greedy | 0.721 | 0.855 | 0.930 | 0.00 | 0.00 | fail all three distinct gates |
| greedy_rp1.2 | 0.931 | 1.000 | 1.000 | 0.00 | 0.00 | fail (over-diverse) |
| t0.7_p0.9 (3 seeds) | 0.886 | 0.980 | 0.998 | 0.00 | 0.00 | **pass** |
| t0.8_k40 (3 seeds) | 0.860 | 0.979 | 0.996 | 0.00 | 0.01 | **pass** |
| t1.0 (3 seeds) | 0.964 | 1.000 | 1.000 | 0.00 | 0.00 | fail (over-diverse) |

The greedy sample from the report, verbatim (its three lines are the generated line breaks; the one U+FFFD is
the model's):

```text
ers
of the house, and I was going to see the house and the king�s room. I
was going to see the door and then I saw that I was
```

It is less repetitive than the released 3.4M decoder's greedy output (E2610: distinct-1 0.505 against the same
human 0.853) and still not text a reader would keep; the two passing sampled settings are surface passes, not
coherence, and K-020 still applies.

## Interpretation — which hypothesis held

**H2 is supported, and it changes what E2608's gap may be called.** At nearly equal parameters, the same
context, corpus, split, tokenizer, seed, batch shape, token budget and schedule, the rotary decoder reaches
**1.8077** bits per character against the transformer's **1.9043**. The difference is not the 3.3× parameter
and 2× context confound E2608 had to state: those were removed. What remains is the architecture bundle —
rotary positions rather than a learned table, RMSNorm rather than LayerNorm, bias-free q/k/v/o rather than a
fused biased projection, the decoder's initialization and implementation details — and that bundle is worth
about **0.10 bits per character** at this scale on this corpus.

**What it does not say.** The bundle is not decomposed: this experiment cannot say whether RoPE, RMSNorm,
the attention bias removal or the initialization is responsible, and it does not claim any single component
is. It does not say the decoder is better at every scale; the tested scale is ~1M parameters and the
control is the 1M transformer, not the 3.4M decoder. It does not say either model converged: both selected
the 2,400-step ceiling, so both numbers are floors and the comparison is of two truncated models at equal
tokens. And it does not say the text is good: the greedy samples are bound-truncated and repetitive, and the
sampled settings that pass the surface gates still write invented words (K-020).

**What it changes.** E2608's 0.2178-bit difference can no longer be attributed to scale/context alone; a
controlled arm at matched capacity and context shows an architecture-bundle difference of about half that
size. The decoder family remains the release path, now for a measured reason at two scales rather than one.
The next lever is still data and tokens: the treatment's own validation curve is still falling at the
ceiling, and the corpus is 6.72M identifiers seen ~4.2 times (ADR-0041, K-022). The result strengthens the
case for the data experiment; it does not replace it.

## Deviations from the preregistration

1. **The masking feature landed during the run.** The arm started from the tree at `a2416d1`, before ADR-0040's
   code existed. The process imported the pre-masking modules and trained with full causal attention; the
   report has no `document_masking` field. This is the declared protocol, not a deviation in method, but a
   reader comparing the report's shape to the current tree should know why the field is absent.
2. **The control's report lacks the generation, latency and resource blocks.** They were added to the driver
   after E2608 arm B ran. The treatment has them; the control's comparison is therefore on training and
   scoring metrics only, and the control's peak RSS is not reported here.
3. **The quality suite is a secondary measurement**, run after the primary result at the default
   configuration set, not part of the preregistered success rule. It uses the same held-out split, prompt
   procedure and prompt offset as E2610's tuning set, so the human reference is the same; its two passing
   sampled settings are reported as observations, not as the hypothesis test.
4. **The parameter match is 98.2%, not exact.** The 18,816-parameter difference is declared in the model table
   and is almost entirely the control's learned position table and LayerNorm parameters.

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

All six declared limitations held. The result adds one post-run nuance: the secondary quality suite's default
configuration set is not E2610's 19-configuration sweep, so the greedy distinct-1 comparison (0.721 here
against 0.505 for the released 3.4M decoder) is on the same prompts but not the same sweep. It is an
observation, not a controlled decoding comparison.

## Conclusion

**Believed, at this scale and on this corpus:** at nearly equal parameters, the same context, data, split,
tokenizer, seed, batch shape, token budget and schedule, the rotary-decoder bundle reaches **1.8077** bits per
character against the transformer's **1.9043**, a difference of **0.0966** bits (5.1% relative) that is
outside the preregistered ±0.02 band. E2608's 0.2178-bit gap therefore cannot be attributed to scale and
context alone; about 44% of it survives a controlled match at ~1M parameters. The decoder family is the right
release-path architecture for a measured reason at two scales, not one.

**Not believed:** that a single component of the bundle is responsible (the experiment is a bundle test);
that the decoder is better at every scale or on every corpus; that either arm converged (both selected the
2,400-step ceiling with validation still falling); that the model's text is coherent or useful (the greedy
samples are bound-truncated, the mean bigram repetition is 0.655, and settings that pass the surface gates
still invent words); or that this closes K-001 or earns Level 3 of ADR-0039. It does not.

## Follow-up

1. **Put a seed spread on the decoder family (E2609).** E2608's ±0.0077 spread belongs to the transformer;
   the 1.018M decoder's 0.0966 advantage needs a spread at its own scale before the number is quoted as a
   magnitude. Use E2606's method unchanged.
2. **Run the equal-token data experiment (ADR-0041).** The next unit is the 3.4M decoder on a larger, mixed,
   licensed corpus against an equal-token prose control; the data curriculum and compute plan are already
   written. This result strengthens that ordering because the architecture question now has a measured answer
   and the budget question does not.
3. **Decompose the bundle only if a component decision is needed.** A RoPE-vs-learned-position,
   RMSNorm-vs-LayerNorm or bias-removal ablation at 1M parameters would say which part carries the 0.0966;
   it is not required to choose the family, so it is ranked below the data experiment.
4. **Measure document-boundary masking (ADR-0040).** The paired masked/unmasked decoder experiment is
   independent of this result and remains unrun.
5. **Do not call this a capability result.** It is a held-out loss comparison on one corpus, one seed,
   one budget, with no question asked or answered. K-001 stands.
