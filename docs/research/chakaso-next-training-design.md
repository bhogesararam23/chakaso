# Chakaso next training design: data and tokens before architecture

**Status:** Research and proposal. The measurements cited are from committed records; every budget marked
*estimate* is arithmetic over measured throughput, not a promise.
**Date:** 2026-10-29.
**Companions:** [`open-model-landscape.md`](open-model-landscape.md),
[`chakaso-data-curriculum.md`](chakaso-data-curriculum.md),
[`chakaso-compute-and-scaling-plan.md`](chakaso-compute-and-scaling-plan.md),
[`chakaso-capability-benchmarks.md`](chakaso-capability-benchmarks.md).

## Recommendation in one paragraph

Hold the rotary-decoder architecture fixed and make the next unit an **equal-token data experiment** at the
measured 3.4M-parameter scale, then scale to a ~9.8M-parameter decoder only if that experiment shows the data
lever works. The binding constraints, in order, are the **size and diversity of the corpus**, the **token
budget**, and only then capacity or architecture. Decoding has been measured and exhausted (E2610). A 3.4M
model trained on 19.66M positions is at ~5.8 tokens per parameter against Chinchilla's ~20 and has seen its
6.72M-token corpus 4.2 times; the next run should spend its compute on new tokens, not on a new block.

## The evidence for that ordering

| Finding | Source | Consequence |
| --- | --- | --- |
| E2608 arm A: 3,413,760 parameters, 19,660,800 positions (8,192 model input positions per step over 2,400 steps), 4.2 epochs of the training side, validation still falling at the 2,400-step ceiling | E2608 report and record | The current result is a budget floor, not a converged level |
| Corpus: 6,718,455 subword identifiers (4,690,865 train), one register (19th-century English prose), 24 books | E2608 report | ~5.8 tokens/parameter at 3.4M; ~30 epochs for a 10M model at Chinchilla's 20 |
| E2610: greedy decoding produces the repetition; the model's sampled distribution has human-level word diversity; no decoding setting produces coherent text | E2610 record | Decoding is not the lever for usefulness; the model/data is |
| Small open models are trained at 560–6,500 tokens per parameter (OLMo 2 7B 3.90T, OLMo 3 7B 6T, SmolLM3 3B 11.2T, TinyLlama 1.1B 3T) | open-model landscape, reported figures | The 5.8 ratio is not "small"; it is an order of magnitude below what the small-model literature spends |
| Chinchilla: parameters and tokens scale together; ~20 tokens/parameter at the compute-optimal point | Hoffmann et al. 2022 | A 3.4M model's compute-optimal budget is ~68M tokens, ~10× the current corpus |
| Repeating data beyond a few epochs has diminishing value | Muennighoff et al. 2023 | "Train longer on the same corpus" is not a substitute for new data |
| E2611: capacity- and context-matched decoder vs transformer | running at the time of writing; its record is the preregistration | Decides whether the E2608 0.2178-bit gap is architecture or scale; either way the next lever is data |

The ordering is falsifiable: if an equal-token data experiment at fixed architecture does **not** improve
held-out loss over the current corpus, the data hypothesis is wrong and the next question is capacity or
optimization. If it does improve, the following run scales the winning recipe.

## The next model, concretely

### Architecture — keep the measured decoder family

Keep the E2608/E2611 decoder: rotary positions, RMSNorm pre-norm, explicit bias-free q/k/v/o with a key/value
cache, GELU feed-forward, tied embeddings, depth-scaled residual initialization. It is measured, loadable,
streamable and checkpointed under `chakaso-decoder-checkpoint/1`. Do not add MoE, MLA, FP8, multi-token
prediction or a second attention variant for the next run; none has a measured reason at this scale, and each
would confound the data experiment.

| Stage | `d_model` | Layers | Heads | FF | Context | Vocab | Parameters | Why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Control (E2608/E2611)** | 256 | 4 | 8 | 640 | 256 | 4,096 | 3,413,760 | The measured reference; do not move it in the data experiment |
| **Data experiment** | 256 | 4 | 8 | 640 | 256 | 4,096 | 3,413,760 | One variable: the corpus |
| **Scale step (if data works)** | 384 | 6 | 12 | 1,024 | 256 | 4,096 | 9,843,840 | ~2.9× capacity, still trainable on this machine; fits the ADR-0037 table's 9.5M class |

A 9.84M model at Chinchilla's 20 tokens/parameter needs ~197M tokens. At the measured decoder rate of ~4,800
positions/s, that is ~11.4 hours of gradient work plus ~35% evaluation/checkpoint overhead, ~15 hours — a
one-day run, not a one-session run. A 6.73M model (`d_model` 320, 6 layers, 8 heads, FF 768) at 135M tokens
is a smaller intermediate if the 9.84M run is too long; the arithmetic is in the compute plan.

### Context and tokenizer — hold them fixed for the data experiment

- **Context 256.** E2610's positioning check found teacher-forced loss flat from position ~16 to 255, so the
  model uses the window; 512 is a later controlled comparison, not a free improvement. At 256 the measured
  throughput and memory are known.
- **Vocabulary 4,096.** At a 9.84M model, a 4k vocabulary costs 1.57M embedding parameters (16% of the model);
  8k would cost 3.15M (32%) and 16k would cost 6.3M (64%). A larger vocabulary may help code and mathematics,
  but at this scale it buys fewer tokens at the price of a large fraction of the parameters. Hold 4k for the
  data experiment; test 8k only when a code/math mixture is measured.

### Optimization — the measured recipe, with one declared change at a time

- AdamW, learning rate 6e-4, weight decay 0.01, warmup 200, gradient clipping 1.0, float32 on CPU.
- **Cosine decay to 0.1** for new long runs, because it is implemented and E2608 used it for the decoder arm;
  keep `constant` for control arms that reproduce E2605–E2608. Do not add a WSD schedule without a separate
  controlled experiment and a code change.
- Batch 32 × 2 accumulation at context 256 = 16,448 positions per step (8,224 per micro-batch); a 68M-token
  run is ~4,130 steps, a 197M-token run ~12,000 steps. State the effective batch and the token budget in the
  record before the run.
- Evaluation every 250 steps, patience 3, `min_delta` 5e-4. The evaluation set should be a fixed, whole-work
  held-out set, not the full validation side every time; if a subset is used, declare it and keep a full pass
  for the final report.
- **Document-boundary masking is off** in the control and is the subject of its own paired experiment
  (ADR-0040). Do not enable it and change data in the same run.

### Data — the one variable

The data experiment compares two arms at the same model, token budget, split, seed and schedule:

| Arm | Corpus | Purpose |
| --- | --- | --- |
| P (prose control) | Expanded general English prose, ~68M tokens | The new data volume, one register |
| T (technical mixture) | ~68M tokens at 50% general prose / 25% technical-scientific / 10% mathematics / 10% code / 5% reference | The mixture hypothesis |

Both arms are selected to the same token count from a prepared pool; the tokenizer is fitted on the training
side of each arm (or, better, on a shared sample with the same target size and a declared rule). The primary
metric is held-out bits per character on **two whole-work sets**: a prose set and a technical set. The
success condition is declared before the run:

- **T improves technical held-out bpc** over P by a material margin (the exact band is declared in the
  record, in the same spirit as E2611's 0.02), **and**
- **T does not regress prose held-out bpc** by more than the same band, **and**
- both arms beat their uniform and unigram baselines on both sets.

If T improves technical but damages prose, the mixture is a trade-off, not a win; the record says so and the
next mixture reweights. If neither improves over the current E2608 baseline at equal tokens, the data
hypothesis is not supported at this scale.

### Evaluation — loss is the primary, but not the only, gate

- Held-out bits per character on whole-work prose and technical sets, with uniform/unigram baselines.
- The calibrated repetition suite (E2610) on the same prompts; surface statistics only.
- A coherence measure that is not a repetition statistic, when one exists: a human rubric on a sample, or the
  cross-model likelihood proxy proposed in the capability-benchmarks document. Until then, no coherence claim.
- Raw samples, seeds, configuration and artifact digests retained.

### Post-training — not in this unit

The prerequisites are a base checkpoint that beats baselines on a whole-work held-out set, a documented
instruction dataset, assistant-turn loss masking, and a separate evaluation that measures instruction
following rather than next-token loss. None exists. Do not run SFT to make a non-converged base look better;
that is the failure mode ADR-0033's ordering exists to prevent.

## Alternatives considered

| Alternative | Why not now |
| --- | --- |
| **MoE / MLA / FP8 / multi-token prediction** | Cluster techniques whose overhead exceeds their benefit below ~100M parameters; no measurement here supports them; adding one would confound the data experiment |
| **Jump to 100M+ parameters** | Needs ~2B tokens and ~150 hours of CPU at measured rates; the corpus is 6.72M tokens. Capacity without data is the mistake E2608 already made at a smaller scale |
| **Web-scale corpus (FineWeb, The Stack)** | Licence and provenance review, disk and compute; not a one-session acquisition. Gutenberg + public-domain science is the legal first step |
| **Instruction tuning now** | The base cannot follow instructions; SFT would measure the data, not the model |
| **Distillation from a hosted teacher** | Forbidden as the product by ADR-0033; a teacher-derived dataset would need its own provenance and experiment, and it would not teach the project whether its own pretraining path works |
| **A new architecture from the frontier** | E2608/E2611 are the controlled comparisons; adopting a bundle would make the next result uninterpretable |
| **Decoding-only fixes** | E2610 measured them; they change surface statistics, not the text |
| **A larger vocabulary now** | At 9.84M parameters, 16k vocabulary would spend 64% of the model on embeddings; the trade-off is unmeasured and belongs after the data question |

## Expected benefits, costs and risks

**Benefits.** A 3.4M model on 68M new tokens, if the data is diverse enough, gives the project its first
result where the corpus is not the binding constraint. It produces a clean answer to "does data help at this
scale?" and a corpus/recipe that a 9.84M model can reuse. It also exercises the catalogue, tokenizer, packing
and evaluation path at a scale an order of magnitude beyond E2608.

**Costs.** Data acquisition and preparation (network, licence checks, tokenizer fit, packing) is days of work
before the first training step. The two-arm experiment is ~10 hours of CPU; the 9.84M follow-up is ~15 hours.
The machine has 0.68 GB free at inspection, so it needs a quiet window and a declared memory floor.

**Risks.** The expanded corpus may be dominated by the same register (Gutenberg is still mostly prose), so the
"technical" arm may be too small to move technical loss. The tokenizer may fragment mathematics and code
badly. Whole-work held-out splits may leave too few documents. The loss may still be falling at the budget
ceiling, in which case the result is a floor and the next run is longer, not different. Each risk has a
declared check in the experiment record, not a hope.

## What would falsify this plan

1. The equal-token data experiment shows no improvement over the current corpus at fixed architecture, seed,
   budget and split. Then the next unit is capacity or optimization, not more data.
2. The technical arm improves technical loss only by damaging prose beyond the declared band. Then the mixture
   weights are wrong, not the data hypothesis.
3. The expanded corpus cannot be licensed or is too small to reach 68M tokens. Then the plan stops at the
   corpus stage and the next unit is a licence-appropriate source, not a training run.
4. E2611 shows a large architecture effect at matched capacity (outside its ±0.02 band) in either direction.
   Then the architecture question reopens first; the data plan is unchanged in ordering but the winning
   architecture is the one the data experiment uses.

## Definition of done for the next unit

A preregistered record, a prepared and pinned corpus with a data card and whole-work splits, two equal-token
arms (prose vs technical) trained at the measured 3.4M configuration, held-out bits per character on both
sets with baselines, the repetition suite run on both, raw samples retained, a results section that states
the completion class and the budget floor, and a decision that names the next lever. No capability claim, no
Level 3, and no instruction tuning in the same record.
