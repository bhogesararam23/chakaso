# The open-model landscape: what five families report, and what a CPU-scale project can use

**Status:** Research — a sourced comparison, not a measurement of any model here.
**Date:** 2026-10-29. **Access date for every source:** 2026-10-29.
**Scope:** DeepSeek-V3/R1, Qwen2.5/Qwen3, OLMo 2/3, SmolLM2/3, TinyLlama, and the scaling, data,
architecture, optimisation and evaluation literature that bears on a from-scratch small model.
**What this document is not:** a claim that any recipe below would work at Chakaso's scale. Each section
separates what a report states from what this project can use, and the last section says what Chakaso has
actually validated.

## Why these families

They span the decisions a from-scratch project actually faces: a frontier mixture-of-experts system whose
techniques are mostly cluster-bound (DeepSeek), a family that documents dense and MoE trade-offs and a very
large multilingual tokenizer (Qwen), the most complete open *process* (OLMo), the closest published scale to
Chakaso's ambition (SmolLM), and a deliberately small, heavily overtrained baseline (TinyLlama). No hosted
model's answers are used anywhere in this repository; these are research references only.

## Comparison table

All figures are **as reported by the cited source**. "Data transparency" means whether the pretraining data
itself is released or only described. Licences are the model-weight licences, not the code licences.

| Family / model | Sizes | Pretraining tokens | Data transparency | Training stages reported | Weight licence | Reproducibility |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V3 | 671B total, 37B active per token (MoE) | 14.8T | No pretraining-data release; architecture and recipe described | Pretraining; no public SFT/RL detail in the technical report's abstract | DeepSeek Model License; not an OSI licence | Recipe described, data not reproducible |
| DeepSeek-R1 | 671B total, 37B active; distilled 1.5B–70B variants | Not stated in the abstract; R1 is post-trained from a V3-class base | 800k reasoning SFT samples described; no pretraining data | Cold start → reasoning RL (GRPO) → rejection sampling + SFT → all-scenario RL | DeepSeek Model License; distilled variants inherit base licences | Process described; data and compute not reproducible here |
| Qwen2.5 | 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B | 18T | No pretraining-data release | Pretraining; SFT on >1M samples; DPO + GRPO | Apache-2.0 for the open sizes (check each model card) | Weights and recipe described; data not released |
| Qwen3 | 0.6B–235B; dense and MoE; flagship 235B total / 22B active | 36T | No pretraining-data release | Long-CoT cold start → reasoning RL → SFT with thinking/non-thinking fusion → general RL → strong-to-weak distillation for small models | Apache-2.0 for the open sizes | Weights and recipe described; data not released |
| OLMo 2 | 7B, 13B, 32B | 3.90T (7B), 5T (13B) | Dolma released, with the training mix | Pretraining with late-stage curriculum; mid-training; RLVR | Apache-2.0 | Full data, code, logs and intermediate checkpoints |
| OLMo 3 | 7B, 32B | 6T | Dolma 3 released (see composition below) | Pretraining → mid-training → long-context → SFT → DPO (Delta Learning) → RLVR (OlmoRL) | Apache-2.0 | Every stage, checkpoint, data point and dependency released |
| SmolLM2 | 1.7B (family also documents smaller sizes) | ~11T | Data mixture described; SmolTalk/FineMath/Stack-Edu collections released | Multi-stage pretraining mixing web, math, code; SFT; DPO | Apache-2.0 | Weights, configs, logs and most data released |
| SmolLM3 | 3B | 11.2T | Pretraining/mid-training/SFT/preference data released as SmolTalk2; mixture described | 3-stage pretraining → 100B-token long-context → 35B reasoning mid-training → 1.8B SFT → APO preference optimisation → model merging | Apache-2.0 | Weights, configs, logs, intermediate checkpoints and data released |
| TinyLlama | 1.1B | 3T | Data mixture described; not released as a reproducible package | Single pretraining run; no post-training stage in the base release | Apache-2.0 | Weights and recipe described; data not reproducible |

### Reported facts behind the table

**DeepSeek-V3.** The technical report states 671B total parameters with 37B activated per token, 14.8T
training tokens, Multi-head Latent Attention (MLA), a DeepSeekMoE architecture with an auxiliary-loss-free
load-balancing strategy, multi-token prediction as a training objective, and 2.788M H800 GPU hours for the
full training run (abstract; the report's training-precision section describes an FP8 mixed-precision
framework). **DeepSeek-R1** reports a four-stage pipeline: a small long-chain-of-thought cold start, the same
large-scale RL as R1-Zero, rejection sampling and SFT, then a second RL stage for helpfulness and
harmlessness; the RL uses Group Relative Policy Optimization (GRPO), which drops the critic and estimates a
baseline from group scores, with rule-based accuracy and format rewards. R1-Zero is trained by RL without
SFT. The report states that R1 is optimised for Chinese and English, can mix languages, and is sensitive to
prompts (conclusion).

**Qwen2.5.** The report states 18T pretraining tokens; sizes 0.5B–72B; GQA, SwiGLU, RoPE, QKV bias and
RMSNorm; 128K-token context in the open models; over 1M SFT samples; and DPO plus GRPO in post-training
(§2–§4). **Qwen3** reports 36T tokens, 119 languages and dialects, six dense and two MoE configurations,
0.6B–235B parameters, a flagship with 235B total and 22B active, and a five-phase post-training pipeline:
long-CoT cold start, reasoning RL on mathematics and coding with verification, SFT that fuses
thinking/non-thinking modes, general-domain RL, and strong-to-weak distillation for compact models
(abstract and §2–§4).

**OLMo 2.** The 7B model was trained on 3.90T tokens and the 13B on 5T; the architecture changes include
RMSNorm, a reordered norm, QK-norm, Z-loss and RoPE with a larger base, and the release includes weights,
data, code and logs (abstract and architectural section). **OLMo 3** reports 7B and 32B models trained on a
6T-token mix drawn from a 9T-token cleaned collection; the released Dolma 3 mix is 76.1% Common Crawl,
13.6% academic PDFs linearised with olmOCR, 6.89% rebalanced GitHub code, 2.56% mathematics web content,
0.86% arXiv LaTeX and 0.04% encyclopedic entries. The model flow is pretraining → mid-training →
long-context extension → SFT → DPO (Delta Learning) → RLVR (OlmoRL), and the authors state that every stage,
checkpoint, data point and dependency is released. They also state that DPO and RL support in the core
training stack was planned but not complete at the paper's writing, and that RL methodology is still
immature and expensive to tune.

**SmolLM2.** The report describes a 1.7B model trained on ~11T tokens with a multi-stage mixture of web
text, mathematics, code and instruction data, using FineMath, Stack-Edu and SmolTalk. **SmolLM3** is a 3B
model trained on 11.2T tokens in six languages, with GQA (4 groups), NoPE applied to every fourth layer,
context extended from 4K to 32K to 64K tokens (and to 128K at inference with YaRN), a three-stage pretraining
mixture that raises code and mathematics from 12%/3% to 24%/13%, a 100B-token long-context phase, a 35B-token
reasoning mid-training phase, 1.8B SFT tokens, and Anchored Preference Optimization (APO, an off-policy DPO
variant) rather than on-policy RL. The blog states that weights, configs, logs, intermediate checkpoints and
the SmolTalk2 data are released, and names two limitations: the APO preference data was capped at 24K tokens,
which degraded long-context benchmarks until a model merge, and some reasoning traces were generated
synthetically by Qwen3-32B.

**TinyLlama.** The project page reports a 1.1B model, 3T tokens, 16 A100-40G GPUs, 90 days, 22 layers,
32 heads, 4 query groups, a 2048-token context, and the Llama 2 architecture and tokenizer.

## What the comparison actually shows

### 1. The small models are heavily overtrained by Chinchilla's standard

Chinchilla's compute-optimal result is that parameters and training tokens should be scaled together, and
that most models of its time were undertrained; the paper's 70B example used four times more data than the
prior Gopher recipe (abstract; the widely cited configuration is 70B parameters and 1.4T tokens). The
ratio implied is roughly **20 tokens per parameter** at the compute-optimal point.

The small open models sit far above that:

| Model | Parameters | Tokens | Tokens per parameter |
| --- | --- | --- | --- |
| TinyLlama | 1.1B | 3T | ~2,700 |
| SmolLM2 | 1.7B | 11T | ~6,500 |
| SmolLM3 | 3B | 11.2T | ~3,700 |
| OLMo 2 7B | 7B | 3.90T | ~560 |
| OLMo 3 7B | 7B | 6T | ~860 |
| **Chakaso E2608 arm A** | **3.41M** | **19.66M positions** | **~5.8** |

This is the single most important comparison in this document. E2608's arm A consumed about 5.8 tokens per
parameter, about 29% of Chinchilla's 20 and about 1/640th of SmolLM3's ratio. Its validation loss was
still falling at the 2,400-step ceiling. A model this undertrained cannot be expected to produce coherent
text, and no architecture change fixes a data-and-token budget that is one to three orders of magnitude
short. **Interpretation, not a reported fact:** the project's next model should be chosen by the data and
token budget it can afford, not by the architecture's novelty.

### 2. Data-constrained training has diminishing returns, so "more epochs" is not a substitute

The data-constrained scaling work finds that repeating data is useful up to a point and then decays: extra
epochs are not equivalent to new tokens, and the value of repetition falls as the number of repeats grows
([Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)). Chakaso's pinned corpus is ~6.72M subword
identifiers (E2608 report: 4,690,865 train, 1,023,146 validation, 1,004,444 test). E2608 trained ~19.66M
positions, about 4.2 epochs of the training side. A 10M-parameter model at 20 tokens/parameter would need
~200M tokens, which is ~30 epochs of this corpus — beyond the range where repetition is a good substitute for
new text. **The data bottleneck is not a preference; it is arithmetic.**

### 3. Full-process transparency is a separate achievement from scale

OLMo 3 releases every stage, checkpoint, data point and dependency; SmolLM3 releases weights, logs,
intermediate checkpoints and most data; DeepSeek and Qwen release weights and recipes but not pretraining
data. Chakaso's own standard (ADR-0036: a measured corpus is recoverable from the revision that pinned it;
ADR-0039: a release is gated on evidence) is closer to OLMo's than to the closed-data families. The useful
lesson is not "copy OLMo's 6T-token mix" — that is a cluster-scale artefact — but **copy the model flow**:
name each stage, release the checkpoints, and make the data mixture a versioned artefact. The project already
does much of this; what it lacks is a stage after pretraining.

### 4. Post-training is not a substitute for pretraining

Qwen3's five-stage post-training pipeline, DeepSeek-R1's four stages and OLMo 3's SFT/DPO/RLVR sequence all
begin from a pretrained base that already models language and has latent capabilities. The reports do not
claim that post-training creates the base capability; they claim it shapes and elicits it. SmolLM3's SFT and
APO stages are 1.8B and a preference set, not the 11.2T-token pretraining. **For Chakaso, an instruction
stage is premature while the base model is a 3.4M-parameter, 5.8-token-per-parameter model that writes
repetitive text.** The correct order is base data and tokens, then a measured base, then SFT.

### 5. The frontier architecture bundle is mostly a cluster technique

MLA, DeepSeekMoE, FP8 training, multi-token prediction and 671B-parameter routing are designed for clusters
and large batches. At 1–30M parameters on one CPU, the relevant choices are much narrower: rotary positions,
pre-norm RMSNorm, SwiGLU/GELU feed-forward, tied embeddings, a key/value cache, grouped-query attention only
if inference memory matters, and a stable optimiser. E2608 already measured a rotary decoder against a
transformer at one scale; E2611 is the capacity-matched test. The landscape does not justify adding MoE or
MLA to a model whose total parameter count is smaller than one attention head in a frontier model.

## What transfers, and what does not

| Technique | Where it is reported | Transfers to Chakaso? | Why |
| --- | --- | --- | --- |
| Chinchilla-style token/parameter scaling | Hoffmann et al. 2022 | **Yes, as a budget rule** | It is arithmetic, not a cluster feature; it says the current run is ~4× undertrained by tokens and the corpus is ~30× too small for a 10M model |
| Multi-stage data mixture (web → math/code → long context) | SmolLM3, OLMo 3 | **Yes, in a scaled-down form** | The project can afford a few million tokens per stage; the mixture ratios must be declared before the run |
| Deduplication | Lee et al. 2021 | **Yes** | The manifest already records digests and duplicate groups; near-duplicate screening exists in `dataset_report` |
| Document-level split and packing | OLMo/SmolLM practice | **Already implemented** | Packing and document-terminated examples exist; the split is by document |
| QK-norm, Z-loss, RMSNorm, RoPE | OLMo 2, Qwen2.5 | **Yes, but one change at a time** | These are small-model-compatible; each needs a controlled comparison, not a bundle adoption |
| GQA / MQA | Qwen2.5, TinyLlama | **Only if inference cost matters** | It reduces KV-cache size; at 3.4M parameters the cache is already tiny |
| MoE routing | DeepSeek-V3, Qwen3 | **No, not at this scale** | Routing overhead and load-balancing complexity exceed the capacity benefit below ~100M parameters; no measurement here supports it |
| MLA / FP8 / multi-token prediction | DeepSeek-V3 | **No** | Cluster-oriented; FP8 is not a CPU training path here |
| RL with verifiable rewards | DeepSeek-R1, OLMo 3, Qwen3 | **Not yet** | It needs a base model that can already produce a correct answer sometimes; Chakaso's base cannot |
| Distillation from a teacher | DeepSeek-R1, Qwen3, SmolLM3 | **Only as a labelled, teacher-derived experiment** | ADR-0033 forbids substituting a hosted model's answers; any teacher data needs provenance and its own record |
| Full checkpoint/data/log release | OLMo 3 | **Yes, as a process standard** | The project already pins data and cites artifacts by digest; the missing half is machine-readable evidence for early records (K-017) |

## Claim ledger

Each row is a claim this document uses, the source that states it, the lesson drawn, the limit on that lesson,
and whether Chakaso has implemented or validated anything like it. "Not applicable" means the technique is
real but out of reach at this scale; it is not a judgement that the source is wrong.

| Claim (as reported) | Source and section | Lesson for Chakaso | Limit | Status here |
| --- | --- | --- | --- | --- |
| DeepSeek-V3: 671B total / 37B active, 14.8T tokens, MLA, DeepSeekMoE, auxiliary-loss-free balancing, MTP, 2.788M H800 GPU hours | DeepSeek-V3 report, abstract and architecture/training sections, 27 Dec 2024 | MoE and MLA buy capacity per FLOP at cluster scale; they are not a small-model recipe | Needs a cluster, large batches and routing infrastructure; no measurement here | Not applicable at 3.4M; not implemented |
| DeepSeek-R1: cold start → reasoning RL (GRPO, rule-based accuracy/format rewards) → rejection sampling + SFT → all-scenario RL; R1-Zero is RL without SFT | DeepSeek-R1 report, abstract and pipeline sections, 22 Jan 2025 | Verifiable rewards can elicit reasoning from a base that already has latent capability; the base is the prerequisite | Post-training did not create the base; the report notes language mixing and prompt sensitivity | Not started; no base, no reward environment |
| Qwen2.5: 18T tokens, 0.5B–72B, GQA, SwiGLU, RoPE, QKV bias, RMSNorm, 128K context, >1M SFT samples, DPO + GRPO | Qwen2.5 report, abstract and §§2–4, 19 Dec 2024 | The architecture components are standard and small-model-compatible; the data and post-training are the differentiators | Pretraining data is not released; the recipe is not reproducible from the report | Decoder already uses RoPE/RMSNorm/GELU/tied embeddings; no GQA or QKV bias; no SFT |
| Qwen3: 36T tokens, 119 languages, 0.6B–235B dense and MoE, thinking/non-thinking modes, five-stage post-training, strong-to-weak distillation | Qwen3 report, abstract and §§2–4, 14 May 2025 | Post-training shapes behaviour; compact models can be distilled from a stronger teacher | Distillation is teacher-derived data and must be labelled; data is not released; long-context degradation is reported with thinking mode | Not started; no instruction stage, no teacher data |
| OLMo 2: 7B/13B/32B, 3.90T/5T tokens, RMSNorm, reordered norm, QK-norm, Z-loss, RoPE, full data/code/log release | OLMo 2 report, abstract and architecture sections, Jan 2025 | Stability tricks (QK-norm, Z-loss) and full-process release are adoptable ideas; the data scale is not | Cluster-scale tokens and compute; the release is the lesson, not the size | Full-process release is the project's standard (ADR-0036/0039); QK-norm/Z-loss not implemented |
| OLMo 3: 7B/32B, 6T tokens, Dolma 3 (76.1% web, 13.6% PDFs, 6.89% code, 2.56% math, 0.86% arXiv, 0.04% encyclopedic), pretraining → mid-training → long-context → SFT → DPO → RLVR, every stage released | OLMo 3 report, data and model-flow sections, Dec 2025 | A staged model flow with named mixtures and released checkpoints is the right process template | 6T tokens is cluster scale; the authors state parts of the RL/DPO stack were still maturing | Process template adopted in the data curriculum; stages beyond pretraining not started |
| SmolLM2: 1.7B, ~11T tokens, multi-stage web/math/code/instruction mixture, FineMath/Stack-Edu/SmolTalk | SmolLM2 report, abstract and data sections, Feb 2025 | Small models benefit from a staged mixture that raises math/code late; data curation is the lever | 11T tokens is ~6,500 tokens/parameter, far beyond this CPU; data collection is a project of its own | Mixture plan drafted; no corpus acquired |
| SmolLM3: 3B, 11.2T tokens, six languages, GQA, NoPE every fourth layer, 4K→32K→64K context, staged mixture, APO, model merging; weights/logs/intermediate checkpoints/data released | SmolLM3 blog, training and release sections, 8 Jul 2025 | A 3B model can be overtrained on ~11T tokens; long-context extension is a stage, not a flag; release the process | APO is off-policy, not on-policy RL; some reasoning traces are synthetic from Qwen3-32B; long-context degradation from preference data is reported | Not implemented; noted as a later-stage option |
| TinyLlama: 1.1B, 3T tokens, 16 A100-40G, 90 days, GQA, 2048 context | TinyLlama repository/paper, 2024 | A 1B model at 3T tokens (~2,700 tokens/parameter) is a multi-GPU, months-scale project; the token budget is the cost | No data release; a single 2023-era recipe; not a capability benchmark | Not applicable on this CPU; used only as a scale anchor |
| Chinchilla: parameters and tokens should scale together; ~20 tokens/parameter at the compute-optimal point | Hoffmann et al., abstract and scaling analysis, 2022 | The current 3.4M model at 5.8 tokens/parameter is ~3.4× below the compute-optimal token count; a 10M model needs ~200M tokens | Compute-optimal is not capability-optimal; small models may benefit from more tokens than 20/parameter | Known and stated; the next data experiment targets it |
| Repeating data has diminishing value beyond a few epochs | Muennighoff et al., 2023 | More epochs of the 6.72M-token corpus is not a substitute for new text | The exact decay depends on data and model; the paper's regimes are larger | Informs the data curriculum; not measured here |
| Deduplication improves training | Lee et al., 2021 | Remove exact and near duplicates before training; report the screen | Dedup does not replace quality filtering or licence review | Exact digests and near-duplicate screening exist; applied to the v2 corpus |
| Benchmark scores can be contaminated by training data | Data Provenance Initiative, 2023; GSM1k-style checks, 2024 | Hold out whole works and benchmark items; report overlap checks; do not tune on test | No check can prove a model never saw a paraphrase | Benchmark plan requires whole-work splits and overlap checks; not yet implemented |
| Full-process release (data, code, logs, checkpoints) is possible | OLMo 3, SmolLM3 | Release the process, not only the final weights | Requires the data to be redistributable and the logs to be kept | The project releases records, contracts and digests, but not weight bytes (artifact policy); early records lack machine-readable evidence (K-017) |

## What Chakaso has actually validated, and what it has not

**Validated by measurement here:** a from-scratch character model beats trivial baselines (E2601/E2602); a
subword Transformer beats the character architecture on held-out bits per character, across seeds and on
external licensed prose (E2605–E2607); a rotary decoder beats a larger transformer at a matched token budget,
with a stated scale/context confound (E2608); greedy decoding, not the data pipeline, produces the released
model's repetition (E2610); and a calibrated quality suite can separate some degenerate settings from others
without measuring coherence (E2610, K-020).

**Not validated here:** that any open-model recipe in this document transfers to a 3.4M-parameter CPU model;
that a larger Chakaso model would converge within an affordable budget; that any instruction-following,
factuality, mathematics, code, physics or research-synthesis capability exists; and that a benchmark score
from another project's harness would mean anything about this one. K-001 stands.

## References

All URLs accessed 2026-10-29. Publication dates are the versions read; a technical report may have later
revisions.

1. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 27 Dec 2024. <https://arxiv.org/abs/2412.19437>
2. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.*
   arXiv:2501.12948, 22 Jan 2025. <https://arxiv.org/abs/2501.12948>
3. Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115, 19 Dec 2024 (v2 Jan 2025).
   <https://arxiv.org/abs/2412.15115>
4. Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, 14 May 2025. <https://arxiv.org/abs/2505.09388>
5. OLMo Team. *OLMo 2: The Best Fully Open Language Model to Date.* arXiv:2501.00656, Jan 2025.
   <https://arxiv.org/abs/2501.00656>
6. OLMo Team. *Olmo 3.* arXiv:2512.13961, Dec 2025. <https://arxiv.org/abs/2512.13961>; announcement
   <https://allenai.org/blog/olmo3>, 20 Nov 2025.
7. Hugging Face. *SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model.*
   arXiv:2502.02737, Feb 2025. <https://arxiv.org/abs/2502.02737>
8. Hugging Face. *SmolLM3: smol, multilingual, long-context reasoner.* Blog, 8 Jul 2025.
   <https://huggingface.co/blog/smollm3>
9. Zhang, P. et al. *TinyLlama: An Open-Source Small Language Model.* arXiv:2401.02385, Jan 2024.
   <https://arxiv.org/abs/2401.02385>; repository <https://github.com/jzhang38/tinyllama>
10. Kaplan, J. et al. *Scaling Laws for Neural Language Models.* arXiv:2001.08361, Jan 2020.
    <https://arxiv.org/abs/2001.08361>
11. Hoffmann, J. et al. *Training Compute-Optimal Large Language Models.* arXiv:2203.15556, Mar 2022.
    <https://arxiv.org/abs/2203.15556>
12. Muennighoff, N. et al. *Scaling Data-Constrained Language Models.* arXiv:2305.16264, May 2023.
    <https://arxiv.org/abs/2305.16264>
13. Lee, K. et al. *Deduplicating Training Data Makes Language Models Better.* arXiv:2107.06499, Jul 2021.
    <https://arxiv.org/abs/2107.06499>
14. Penedo, G. et al. *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.*
    arXiv:2406.17557, Jun 2024. <https://arxiv.org/abs/2406.17557>
15. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* arXiv:2104.09864, Apr 2021.
    <https://arxiv.org/abs/2104.09864>
16. Zhang, B. and Sennrich, R. *Root Mean Square Layer Normalization.* arXiv:1910.07467, Oct 2019.
    <https://arxiv.org/abs/1910.07467>
17. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
    Checkpoints.* arXiv:2305.13245, May 2023. <https://arxiv.org/abs/2305.13245>
18. Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need.* arXiv:1911.02150, Nov 2019.
    <https://arxiv.org/abs/1911.02150>
19. Shazeer, N. *GLU Variants Improve Transformer.* arXiv:2002.05202, Feb 2020.
    <https://arxiv.org/abs/2002.05202>
20. Dao, T. et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.*
    arXiv:2205.14135, May 2022. <https://arxiv.org/abs/2205.14135>
21. Loshchilov, I. and Hutter, F. *Decoupled Weight Decay Regularization.* arXiv:1711.05101, Nov 2017.
    <https://arxiv.org/abs/1711.05101>
22. Yang, G. et al. *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter
    Transfer.* arXiv:2203.03466, Mar 2022. <https://arxiv.org/abs/2203.03466>
23. Ouyang, L. et al. *Training Language Models to Follow Instructions with Human Feedback.*
    arXiv:2203.02155, Mar 2022. <https://arxiv.org/abs/2203.02155>
24. Rafailov, R. et al. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.*
    arXiv:2305.18290, May 2023. <https://arxiv.org/abs/2305.18290>
25. Shao, Z. et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.*
    arXiv:2402.03300, Feb 2024. <https://arxiv.org/abs/2402.03300>
26. Lightman, H. et al. *Let's Verify Step by Step.* arXiv:2305.20050, May 2023.
    <https://arxiv.org/abs/2305.20050>
27. Eldan, R. and Li, Y. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
    arXiv:2305.07759, May 2023. <https://arxiv.org/abs/2305.07759>
28. Gunasekar, S. et al. *Textbooks Are All You Need.* arXiv:2306.11644, Jun 2023.
    <https://arxiv.org/abs/2306.11644>
29. Longpre, S. et al. *The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing and
    Attribution in AI.* arXiv:2310.16787, Oct 2023. <https://arxiv.org/abs/2310.16787>
30. Hendrycks, D. et al. *Measuring Massive Multitask Language Understanding.* arXiv:2009.03300, Sep 2020.
    <https://arxiv.org/abs/2009.03300>
31. Cobbe, K. et al. *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168, Oct 2021.
    <https://arxiv.org/abs/2110.14168>
32. Chen, M. et al. *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374, Jul 2021.
    <https://arxiv.org/abs/2107.03374>
33. Rein, D. et al. *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* arXiv:2311.12022, Nov 2023.
    <https://arxiv.org/abs/2311.12022>
34. Zhou, J. et al. *Instruction-Following Evaluation for Large Language Models.* arXiv:2311.07911,
    Nov 2023. <https://arxiv.org/abs/2311.07911>
35. Lin, S. et al. *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* arXiv:2109.07958, Sep 2021.
    <https://arxiv.org/abs/2109.07958>
36. Wei, J. et al. *Emergent Abilities of Large Language Models.* arXiv:2206.07682, Jun 2022.
    <https://arxiv.org/abs/2206.07682>
