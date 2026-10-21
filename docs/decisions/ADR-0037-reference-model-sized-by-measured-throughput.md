# ADR-0037: the reference model is a rotary decoder, sized by measured throughput rather than by ambition

**Status:** Accepted
**Date:** 2026-10-22
**Amends:** [ADR-0033](ADR-0033-model-centred-product-contract.md) (the model is the unit of work),
[ADR-0034](ADR-0034-optional-pytorch-model-extra.md) (torch stays an optional extra), and it is the first
decision taken under [ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md)'s rule that a measured
result must stay re-trainable.
**Related:** E2605, E2606, E2607, and the experiment it sizes: E2608.

## Context

E2605 measured a 643,840-parameter Transformer; E2606 put a spread on it (1.8794 ± 0.0077 bits per character
over five pre-declared seeds) and reproduced the original run exactly; E2607 repeated the protocol on an
externally licensed corpus and found two things. First, the gap to the character architecture survives a
change of data (1.2042 against 1.2143). Second, and less comfortable: **no E2607 run early-stopped**. All
three selected step 2,400, which was the ceiling of the declared budget, with the validation loss still
falling at the last evaluation. Its reported 2.0176 bits per character is therefore a floor, not a converged
level. And every checkpoint from every run of every experiment so far generates text that loops: the held-out
loss improvement never became usable text.

The release mission asks for a reference model trained from random initialization that a developer can
actually load, run and integrate. Before choosing its size, three things were unknown and had to be measured
rather than estimated: what this machine can train per second, whether the new architecture's key/value cache
is correct and worth it, and how much memory the pipeline holds while it runs.

## Problem

Choose the reference architecture and training budget for E2608 using measured evidence, on a machine with no
GPU, 18 logical CPUs, 16.6 GB of RAM and other processes using most of it, and do so without scaling blindly
on the assumption that a bigger model stops looping.

## Evidence (measured 2026-10-22, this machine, on `gutenberg-english-prose-v2`)

Corpus: 919 documents, 14,490,005 characters, fingerprint `e9fc2e92f59d88ea`, snapshot checksum
`8eacb49f8d244e65…`. Split seed 7 → 643 train / 138 validation / 138 test documents. Tokenizer: byte-pair,
4,096 identifiers, `min_frequency` 3, fingerprint `b597eb34f13d9058`, fitted on the training side only.
Environment: Python 3.13.5, `torch 2.13.0+cpu`, 14 torch threads, 18 logical CPUs, **no CUDA device**, 166.8 GB
free disk, 16.6 GB RAM with 1.5 GB free at measurement time. Peak resident set was read from the operating
system (`K32GetProcessMemoryInfo`), which is new: no number in this section is an estimate of memory.

| Configuration | Parameters | Tokens per step | Step time | Tokens per second | Peak RSS | Minutes per 1,000 steps |
| --- | --- | --- | --- | --- | --- | --- |
| transformer d128-L3-H4 ctx128 (E2605 geometry) | 1,037,056 | 4,128 | 0.300 s | 13,761 | 928 MB | 5.0 |
| decoder d128-L4-H4 ctx256 | 1,182,848 | 8,224 | 0.867 s | 9,492 | 1,636 MB | 14.4 |
| decoder d192-L4-H6 ctx256 | 2,167,232 | 8,224 | 1.238 s | 6,641 | 1,836 MB | 20.6 |
| decoder d256-L4-H8 ctx256 | 3,413,760 | 8,224 | 1.618 s | 5,081 | 2,126 MB | 27.0 |
| decoder d256-L4-H8 ctx512 | 3,413,760 | 16,416 | 1.930 s | 4,252 | 2,249 MB | 32.2 |
| decoder d384-L6-H12 ctx512 | 9,548,544 | 16,416 | 2.670 s | 1,537 | 2,249 MB | 44.5 |

Each row is 12 timed steps after 3 untimed warm-up steps, batch 32 (16 at ctx 512, 8 for the largest row),
real AdamW updates with clipping, over the packed training side. Losses moved as expected (the 3.4M row fell
from 6.37 to 4.96 across those 12 steps), which is a sign the timing ran on a training loop and not on a
no-op.

Data preparation, measured on the same corpus: snapshot read 0.2 s (peak RSS 268 MB); document split 0.0 s;
byte-pair vocabulary at 4,096 identifiers **467 s**, at 2,048 **222 s**; packing into 256-token blocks 16.6 s
(18,252 / 3,981 / 3,908 blocks train/validation/test). One epoch at batch 32 with 256-token blocks is **570
steps**.

Decoding throughput, the same 3.4M-parameter decoder, 48 greedy tokens from a 24-token prompt: re-running the
whole prefix each step took **630 ms (76 tokens/s)**; stepping with the key/value cache took **382 ms (126
tokens/s)**, a **1.7× speed-up** at this prompt length, and the cached path's per-step cost does not grow
with the answer.

## Decision

1. **The reference architecture for E2608 is the decoder family at `d_model` 256, 4 layers, 8 heads,
   feed-forward 640, context 256, vocabulary 4,096, tied embeddings — 3,413,760 parameters.** It is the
   largest configuration in the table whose step time keeps a declared budget of a few thousand steps inside
   about an hour and a quarter per run, and whose measured peak RSS (2,126 MB) this machine has been
   observed to hold. The 9.5M row is excluded by measurement, not by taste: 44.5 minutes per 1,000 steps and
   1,537 tokens/s means a converging budget would occupy the rest of the day, and the same table says its
   extra capacity buys throughput it cannot use.
2. **The training budget is declared to converge, not to fit the clock: 2,400 steps with cosine decay to
   0.1 of the peak rate, warmup 200, evaluation every 100 steps, patience 3, `min_delta` 0.0005.** E2607's
   finding was that a constant schedule with a ceiling stopped while still improving. Cosine decay is the
   cheapest declared change that addresses it, and it is a *new experiment identity* — E2608's numbers are
   never restated as E2607's, and E2605–E2607 keep the `constant` schedule that produced them. 2,400 steps at
   8,224 tokens per step is 19.7M tokens, about 4.2 epochs of the training side: enough to test whether the
   curve turns, which is what the record has to answer.
3. **The comparison arm is the measured geometry on the same corpus, matched in tokens rather than in
   steps.** The transformer arm runs batch 32 with two accumulated micro-batches at context 128, so both arms
   consume 8,256 tokens per optimizer step and both stop at 2,400. Comparing 128-token steps against
   256-token steps would compare two different amounts of text, and the arm would lose before it started.
4. **The character-model baseline is skipped on this corpus, and the report says so.** It materializes one
   window per character position; on 14.5M characters that is an order of magnitude beyond the free memory
   measured above. The comparison set for E2608 is therefore unigram, uniform, and the transformer arm.
5. **The key/value cache ships and is used by generation.** It is correct before it is fast: the test suite
   asserts that stepping with a cache matches a full forward pass to 1e-6 on every position, and that a cache
   asked to continue from the wrong position is refused. The 1.7× measured at 48 tokens understates its
   benefit, because the uncached cost grows with the answer and the cached cost does not.
6. **Nothing about generation quality is claimed in advance.** This decision sizes a model; it does not
   predict that the model will stop looping. E2608's generation section is measured with the same suite that
   recorded E2606's and E2607's failures, its raw samples are retained, and if the runs still repeat, the
   release is Level 1 and Level 2 with useful behaviour recorded as **not achieved**.

## Reasoning

- **Sizing by throughput per epoch rather than by parameter count** is what the table makes possible. 570
  steps per epoch says how much text 2,400 steps is; a 9.5M-parameter model at 4.2 epochs on 4.7M tokens of
  training text is a model that will be *under-fed*, and the machine cannot check whether it was the capacity
  or the data that limited it. At 3.4M parameters the same corpus is 4.2 epochs and 19.7M tokens of gradient
  signal, which is the largest configuration this dataset can actually train rather than the largest that
  fits in RAM.
- **Token-matched arms** are the only version of this comparison that measures the architecture. Step-matched
  arms would confound "the decoder is better" with "the decoder saw twice the text per step", and E2606
  already showed this project how easy it is to quote the flattering comparison.
- **The cosine schedule is declared before the run**, with a stated failure condition. If it converges, the
  record will have shown the ceiling was the limit. If it does not, the record will have shown the schedule
  was not either, which is a smaller space of explanations than before.
- **Skipping the character baseline is a loss of information, taken openly.** The alternative was to run it on
  a subsample, which would produce a number that looks comparable and is not.

## Trade-offs

- Slower runs mean fewer seeds. The declared arms are two; a multi-seed study of *this* configuration is
  E2609's job, and E2606 established the seed-spreading method that it will reuse.
- Cosine decay changes the trajectory mid-protocol relative to three prior records; that is why it is a new
  experiment id and not a rerun of an old one.
- A 256-token context on a 14.5M-character corpus is still small. The measurement says why the next one up
  (512) costs 32.2 minutes per 1,000 steps for the same parameters: longer sequences, not more weights.
- Peak RSS 2.13 GB on a machine with 1.5 GB free at measurement time risks paging, which would show up as
  inflated wall-clock rather than as a failure. The record reports the measured RSS and the measured step
  time, and does not present a timing number as a property of the model.

## Rejected alternatives

1. **Scale parameters and hope looping stops.** The E2607 evidence says the loss was still falling; nothing
   measured so far connects parameter count to non-repeating text. Adopting capacity as the fix before
   measuring it is the definition of scaling blindly.
2. **Keep the 128-token measured architecture and train it longer.** Cheapest, and it answers nothing about
   the release path, whose context handling and cache the runtime now depends on. It also confounds the
   comparison the other way: a longer run of the old geometry against a new one is two changes at once.
3. **Subsample the corpus so the character baseline fits.** A baseline computed on different data is not a
   baseline; it is a different experiment wearing an old name.
4. **Train on the repository's own 780 KB text because it is already pinned.** That corpus exists to keep
   E2605 reproducible. A reference model for release should not be one author's documentation.
5. **Declare 8,000 steps to be safe.** At 27 minutes per 1,000 steps that is 3.6 hours per run, which would
   consume the session and still not be a declared convergence criterion — the schedule, not the clock, is
   what decides whether a run converged.

## Consequences

- E2608's preregistration states this configuration and budget before any arm runs, and its pilot
  (`E2608SMOKE`) is a separate experiment id whose numbers are not aggregated with the study.
- `ExperimentPlan` now carries a `model_family`, a `schedule`, accumulation, a shared tokenizer cache and a
  `--sample-prompts held_out` option; `docs/training.md` documents them, and the measured row-by-row table
  above lives in the experiment record rather than only here.
- Peak memory is now a recorded field. Every prior record's sentence "peak memory is not stated anywhere"
  stays true of those records; from E2608 on, the number is read from the operating system.
- Creates one follow-up decision it does not make: whether the decoder becomes the *only* architecture the
  product path trains on, which needs E2608's held-out numbers and its generation measurements in front of
  it.

## Conditions to revisit this decision

1. If E2608's arms both hit the step ceiling with validation still falling, the schedule is not the lever and
   the budget or the optimizer must be reconsidered with the recorded curves in hand.
2. If the decoder beats the transformer arm on held-out loss but its generations still loop, capacity and
   schedule are both excluded as the explanation for degeneration, and the next decision is about data and
   decoding behaviour, not size.
3. If a run's measured peak RSS exceeds the machine's headroom and timings degrade, the configuration drops
   to the next row in the table and the record states which row and why.
