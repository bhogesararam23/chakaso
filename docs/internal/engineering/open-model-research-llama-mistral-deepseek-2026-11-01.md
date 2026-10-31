# Llama, Mistral and DeepSeek, mechanism by mechanism: what transfers below 10M parameters on a CPU

**Status:** Research — a sourced reading of primary technical reports. It is not a measurement of any model
named here, and it is not evidence of any capability of Chakaso's own model.
**Date:** 2026-11-01. **Access date for every source below: 2026-11-01.**
**Method:** located with WebSearch, read with WebFetch against arXiv abstract and HTML pages, the Hugging Face
rendering of two reports whose arXiv HTML truncated mid-page, and one official DeepSeek model card. Every
quoted value is copied from the fetched text; where the fetch could not reach a section, that is written
down as *could not verify* rather than filled in.
**Scope:** Meta Llama (1, 2, 3), Mistral 7B, Mixtral 8x7B, DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 — and the
three attention/objective/schedule mechanisms they are known for (MQA, GQA, SWA, MoE, MLA, MTP), plus
tokenizer size, long-context RoPE changes, warmup-stable-decay versus cosine, FP8 training precision and R1's
reinforcement-learning curriculum.
**What this document is not:** a recipe. [`../../research/open-model-landscape.md`](../../research/open-model-landscape.md)
holds the family-level comparison (tokens, licences, transparency) and the scaling arithmetic; this is the
mechanism ledger behind it, written to answer one question per mechanism: **does this transfer below 10M
parameters and onto a CPU?** Where the honest answer is "the source never claims it at that scale, and the
reason it works is a cluster reason", that is said plainly. This is a research note, not a decision record;
the decisions it bears on are [ADR-0041](../../decisions/ADR-0041-next-training-target-general-base-with-technical-mixture.md)
and the design in [`../../research/chakaso-next-training-design.md`](../../research/chakaso-next-training-design.md).

---

## The configuration being compared against

Everything in the "Relevance at Chakaso scale" fields is arithmetic against this measured configuration
(E2608 arm A, re-declared unchanged in E2612):

| Field | Value | Source |
| --- | --- | --- |
| Architecture | rotary decoder, `d_model` 256, 4 layers, 8 heads, feed-forward 640, tied embeddings, RoPE θ = 10,000, RMSNorm, GELU, bias-free attention | `experiments/E2608/record.md`, `experiments/E2612/record.md` |
| Parameters | 3,413,760, counted from the built model | `experiments/E2608/record.md` |
| Context | 256 positions | same |
| Vocabulary | 4,096 byte-pair identifiers (3,839 learned merges at `min_frequency` 3 in E2612) | same |
| Measured compression | **2.0993 characters per token** on the test side of the multi-register licensed corpus, at vocabulary 4,096 (`experiments/E2612/record.md`); the older 1,024-identifier tokenizer measured 2.0829 on repository text and 1.8782 on external prose (`experiments/E2605/record.md`, `experiments/E2607/record.md`) | E2612, E2605, E2607 |
| Budget | 2,400 steps × 8,192 positions = 19,660,800 positions, packing capped at 20,000,000 training identifiers per arm; **both** E2608's arm A and both E2612 arms finished at the ceiling with validation still falling | E2608, E2612 records |
| Optimizer | AdamW, lr 6e-4, weight decay 0.01, 200 warmup steps, cosine decay to 0.1 of peak, gradient clipping 1.0 | E2612 record |
| Device / precision | float32 on CPU, ~3,900–4,800 positions/s, 18 logical cores, 16.6 GB RAM, no GPU, no CUDA | ADR-0037, E2608 record |
| Headline result | 1.6865 bits per character held out, against 1.9043 for the 1.04M transformer control; no usable text | E2608, E2610 |

Four derived quantities used repeatedly below. They are **arithmetic over the declared configuration, not
measurements**:

- **The embedding table is 1,048,576 of the 3,413,760 parameters (30.7%).** At a fixed `d_model` of 256, every
  extra 4,096 vocabulary identifiers costs 1,048,576 parameters. This is the single structural difference
  between Chakaso's situation and every model in these reports, where the vocabulary is a rounding error
  (Llama 3 8B: 128,256 × 4,096 ≈ 0.5B of 8B, ~6%).
- **Key/query/value projections are 524,288 parameters** (2 of 4 attention matrices, 256×256, 4 layers).
  Collapsing to a single key-value head would remove 458,752 parameters, **13.4% of the model** — so grouped
  attention here is a capacity decision, not an inference one. In the reports it is the reverse: at 7–405B
  the freed capacity is immaterial and the bandwidth is not.
- **The full 256-position key/value cache is ~2 MiB per sequence** (2 × 256 dims × 4 bytes × 4 layers × 256
  positions), against 13 MiB for one float32 copy of the model. Every bandwidth argument in the GQA/MQA/MLA
  literature is about a cache two to four orders of magnitude larger.
- **Scale gaps.** Llama 3 8B is ~2,344× this model's parameters and its 15T tokens are ~763,000× this run's
  19.66M positions. DeepSeek-V3's 2.788M H800 GPU-hours are roughly 2×10⁶ times E2612's measured ~73 minutes per
  2,400-step arm (the record's own figure at the measured rate), on hardware class that does not exist on this
  machine. DeepSeek-V2's stated 172,800 GPU-hours per trillion tokens times its 8.1T tokens is ~1.4M GPU-hours —
  again, against an hour per arm here.

---

## Observations about the source base itself

These are readings, not measurements, and they change how much weight the numbers below can carry.

**Observation 1 — every headline number here is self-reported.** Llama 2, Llama 3, Mistral, Mixtral,
DeepSeek-V2, V3 and R1 all report their own ablations. No artifact in this set has been independently
re-produced at its own scale (an 8B 15T-token run and a 671B MoE cannot be re-run by a third party, and none
of the reports cite a replication). The one mechanism on this list that **two independent groups** measured is
multi-token prediction: Meta's own 0.3B–13B study and DeepSeek's separate 15.7B/228.7B ablation, from different
groups, on different corpora, both reporting downstream-task gains. Where below this document says
"self-reported", it means exactly that and nothing stronger.

**Observation 2 — two Meta primaries disagree about Llama 2's token count.** Llama 2's own Table 1 gives
"2.0T" training tokens (`arXiv:2307.09288`); the Llama 3 paper states "We pre-train Llama 3 on a corpus of
about 15T multilingual tokens, compared to 1.8T tokens for Llama 2" (`arXiv:2407.21783v3`, §1/§3). Both are
Meta. The discrepancy is not resolvable from the fetched text. Consequence for this project: a second-hand
"known fact" about a frontier model is worth checking in the primary before it is repeated, including in this
repository's own documents.

**Observation 3 — Mistral 7B and Mixtral 8x7B report no training-token count in the text fetched.** Twice
(Mistral HTML, ar5iv rendering) the architecture table returned fully and the training-budget fields returned
absent: no total tokens, no batch size, no learning rate, no schedule, no GPU count. Mixtral's abstract likewise
gives only parameters and a 32k context. A paper that names its window size and its head count but not its
token budget cannot be used to reason about tokens per parameter; the landscape document already records this
as a transparency lesson, and the primary reading confirms it.

**Observation 4 — the "Llama 3 uses warmup-stable-decay" claim could not be verified, and the verifiable Meta
primary says cosine.** The Llama 3 herd paper states a "cosine learning rate schedule decaying to 8×10⁻⁷ over
1,200,000 steps" for the 405B run (§3.4.1) and a "cosine learning rate schedule with a linear warmup for 2,000
training steps" with a 90% reduction for the scaling trials (§3.2.1). It does **not** compare WSD with cosine in
the text fetched. A widely cited companion document ("Llama 3 Handbook", often given as arXiv:2404.07944) was
searched for and fetched: **arXiv:2404.07944 resolves to "On the Durdevic approach to quantum principal
bundles"**, a differential-geometry paper. Two further searches did not locate it on arXiv. So: *could not
verify*, and the argument "adopt WSD because Llama 3 does" has no primary source behind it in this reading. The
argument for WSD has to be WSD's own evidence, which is much smaller-scale — see the block below.

**Observation 5 — Llama 1 does have a paper.** `arXiv:2302.13971` (27 Feb 2023) is the primary for Llama 1;
Meta's publication page and the 24 Feb 2023 blog announcement are the release-side primaries. Nothing in the
fetched abstract or Table 2 mentions grouped-query attention, which is consistent with Llama 1 being
multi-head throughout; the GQA column first appears in Llama 2's Table 1.

---

## Mechanism ledger

Format, per mechanism: **Source | Mechanism | Reported evidence | Stated limitations | Reproducibility |
Relevance at Chakaso scale | Verdict.** Verdicts are for **this** configuration (3.4M parameters, 256-position
context, float32 CPU, 19.66M-position budget, archaic public-domain English) and for nothing else.

### 1. Multi-query attention (MQA)

- **Source:** Noam Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*,
  [arXiv:1911.02150](https://arxiv.org/abs/1911.02150), 6 Nov 2019 (single author, Google Brain).
- **Mechanism:** all query heads share one key/value projection, shrinking the cached tensors that autoregressive
  decoding must re-read every step. It is a bandwidth technique for incremental generation.
- **Reported evidence:** the abstract states "greatly reducing the size of these tensors" and reports only
  "minor quality degradation from the baseline". **No speed-up factor, no memory ratio and no model size appear
  in the fetched abstract** — the paper is a 13-page technical note whose numbers were not reached by this
  read. Self-reported.
- **Stated limitations:** the follow-on literature is where the limit is stated, not here. Ainslie et al.
  (next block) characterise MQA as "a more aggressive cut in both memory bandwidth and capacity" that "can lead
  to quality degradation".
- **Reproducibility:** self-reported, 2019, TPU-era encoder-decoder setting; not re-run here.
- **Relevance at Chakaso scale:** the mechanism's whole subject — decode bandwidth — does not exist yet at
  256 positions and 4 layers (~2 MiB of cache per sequence). Adopting it would spend 458,752 parameters of
  capacity to save a cache that is not the bottleneck, and would change the model's parameter count mid-
  comparison.
- **Verdict: REJECT** at this scale, for this run. Revisit only if a measured serving benchmark on this machine
  shows decode time dominated by cache traffic at long context.

### 2. Grouped-query attention (GQA)

- **Source (mechanism):** Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, *GQA: Training Generalized
  Multi-Query Transformer Models from Multi-Head Checkpoints*,
  [arXiv:2305.13245](https://arxiv.org/abs/2305.13245), v1 22 May 2023, v3 23 Dec 2023 (EMNLP 2023).
  **Source (usage):** Llama 2 Table 1, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288), 18 Jul 2023
  (v2 19 Jul 2023); Llama 3 Table 3 and §3.2,
  [arXiv:2407.21783v3](https://arxiv.org/abs/2407.21783), v1 31 Jul 2024, v3 23 Nov 2024; Mistral 7B Table 1,
  [arXiv:2310.06825](https://arxiv.org/abs/2310.06825), 10 Oct 2023.
- **Mechanism:** an intermediate number of key-value heads shared across groups of query heads; the recipe is
  to *uptrain* an existing multi-head checkpoint into GQA rather than train it from scratch.
- **Reported evidence:**
  - Ainslie et al., **Table 1**, T5.1.1 **Large and XXL** only (roughly 0.77B and 11B in that family): mean of
    summarisation ROUGE-1, translation BLEU and QA F1 is **MHA-XXL 47.2, GQA-8-XXL 47.1, MQA-XXL 46.6**;
    inference per sample **1.51 s (MHA) → 0.28 s (GQA-8) → 0.24 s (MQA)** at "the largest batch size that fits
    up to 32 per TPU". They "selected 8 groups as a favorable middle ground" and uptrained with **5% of original
    pre-training compute ("approximately 600 TPUv3 chip-days")**. Their stated quality language: GQA achieves
    "quality close to multi-head attention with comparable speed to MQA". Self-reported; batch size was *not*
    varied, so the speed-up is one operating point, not a curve.
  - **Llama 2 Table 1 is the decisive usage fact: 7B and 13B are marked as not using GQA; 34B and 70B are marked
    as using it.** Meta applied GQA only where decoding a 34B/70B model over a 4K window became bandwidth-bound,
    and explicitly did **not** apply it to the two smallest sizes.
  - Llama 3 §3.2: "with 8 key-value heads to improve inference speed", and Table 3 shows all three sizes (8B,
    70B, 405B) at 8 KV heads — the 8B having 32 query heads, i.e. 4 queries per KV head.
  - Mistral 7B Table 1: 32 query heads, `n_kv_heads` 8, `head_dim` 128.
- **Stated limitations (by the mechanism's own authors):** MQA "can lead to quality degradation", "may not be
  desirable to train a separate model just for faster inference", and MQA risks "training instability during
  fine-tuning, in particular combined with long input tasks". The paper offers **no analysis below T5.1.1 Large
  (≈0.77B)** — the fetched text states this directly.
- **Reproducibility:** self-reported; T5-family, single group. Note the direction of the evidence: GQA was
  invented as a *checkpoint conversion*, and its quality claim is a 0.1-point average difference at 11B.
- **Relevance at Chakaso scale:** three concrete mismatches. (a) Chakaso has **8 query heads**; "8 KV heads"
  would be identity, so the only choices are 4, 2 or 1 KV heads, i.e. 2×–8× capacity removal from the K/V
  projections (458,752 parameters at 1 head = 13.4% of the model). (b) The cache is not the bottleneck at
  256 positions, so the benefit's precondition is absent. (c) The training-throughput cost is zero and the
  training-quality effect is unmeasured below 0.77B — so it would be an *unmeasured capacity reduction* inside a
  run whose whole point is capacity-limited learning. Adopting it "because Llama 3" is doubly wrong: Llama 2's
  own table says the 7B did not use it, and the 8B's 8-KV-head configuration is incoherent transplanted onto an
  8-head model.
- **Verdict: REJECT for E2612 and the 9.84M step.** TEST EXPERIMENTALLY later only as a *capacity-neutral*
  paired arm (GQA arm given extra feed-forward width to restore the freed parameters, matched on the same
  budget), and only once a measured decode-time or RAM profile makes cache size a named constraint. If
  12+ heads exist at the ~9.8M configuration (the design table lists 12 heads at `d_model` 384), the arithmetic
  changes but the precondition — bandwidth-bound decode — still does not.

### 3. Sliding-window attention (SWA)

- **Source:** Mistral 7B §2 (mechanism and the rolling-buffer subsection),
  [arXiv:2310.06825](https://arxiv.org/abs/2310.06825), 10 Oct 2023.
- **Mechanism:** each token attends to the previous layer's hidden states with positions between *i−W* and *i*;
  stacked over *k* layers, a token's effective receptive field is **W×k** tokens, so full-context influence is
  obtained without quadratic cost. Mistral's Table 1 fixes `window_size` = **4096** with `context_len` = **8192**.
- **Reported evidence:** the fetched §2 text states that at **16K** sequence length the code changes "yield a 2x
  speed improvement over a vanilla attention baseline", and that at **32k** tokens the rolling buffer "reduces
  the cache memory usage by 8x, without impacting the model quality". Both figures are about long sequences and
  both are self-reported; the abstract names SWA as one of two efficiency choices but carries no numbers.
- **Stated limitations:** no limitations section was found in the fetched text of the paper (the fetch reported
  it absent), so the quality claim "without impacting the model quality" is asserted without an ablation table
  in the portion read. **Could not verify** whether the paper contains a window-size ablation at all.
- **Reproducibility:** self-reported. Note that the two subsequent large families in scope did **not** adopt it:
  Llama 2/3 and DeepSeek-V2/V3 use full attention (Llama 3 §3.2 and DeepSeek-V2 architecture sections as read).
  A mechanism that a 7B model advertised and a 405B model declined is at minimum not a universal win.
- **Relevance at Chakaso scale:** the window (4096) is **16× longer than the entire context** (256). SWA over a
  256-position context with W ≥ 256 is mathematically identical to full attention — it changes nothing while
  adding a masking code path and a second thing to keep correct in the document-boundary work
  ([ADR-0040](../../decisions/ADR-0040-document-boundary-attention-masking.md)). Every reported benefit sits at
  16K–32K.
- **Verdict: REJECT as cargo-culting at this scale.** It becomes a live question only if context grows past
  ~8K–16K, which the compute plan does not reach: quadratic cost at 256 is not what limits this project.

### 4. Sparse mixture of experts (Mixtral, DeepSeekMoE, auxiliary-loss-free load balancing)

- **Sources:** *Mixtral of Experts*, [arXiv:2401.04088](https://arxiv.org/abs/2401.04088), 8 Jan 2024;
  DeepSeek-V2 [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), v1 7 May 2024; DeepSeek-V3
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), v1 27 Dec 2024, v2 18 Feb 2025.
- **Mechanism:** per layer, *N* feed-forward blocks plus a router that picks *k* of them per token, with one
  always-on shared expert in the DeepSeek design. Mixtral: 8 experts, top-2. DeepSeek-V2: 162 experts per layer
  ("two shared, 160 routed"), 6 selected per token, constrained to at most three devices. DeepSeek-V3: 256
  routed experts plus one shared, **top-8**, "constrained to four compute nodes", with **auxiliary-loss-free
  load balancing** — routing decisions get dynamically adjusted bias offsets tied to real-time batch utilisation
  instead of an added balancing penalty (§2.1.2).
- **Reported evidence:** Mixtral is "47B parameters" total with "13B active" per token, trained at a "context
  size of 32k tokens", and the abstract claims it "outperforms or matches Llama 2 70B and GPT-3.5 across all
  evaluated benchmarks", strongest in mathematics, code and multilingual. DeepSeek-V2: 236B total / 21B active,
  8.1T tokens, "saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation
  throughput to 5.76 times" versus DeepSeek 67B. DeepSeek-V3: 671B total / 37B active, 14.8T tokens,
  **2.788M H800 GPU-hours** (2.664M pre-training, 119K context extension, 5K post-training) on a cluster of
  **2048 NVIDIA H800 GPUs**, and the balancing mechanism "achieves better performance than models that encourage
  load balance through pure auxiliary losses" (§2.1.2) — the fetched text gives no numbers for that claim.
- **Stated limitations:** Mixtral's paper reports **no training-token count** (Observation 3), so its capacity
  per token cannot be compared; the V3 limitations section (§6) was **cut off by fetch truncation and could not
  be verified**. The routing claims are self-reported at cluster scale; none of the three papers reports a
  controlled MoE-versus-dense comparison below tens of billions of parameters.
- **Reproducibility:** self-reported; all three are cluster artifacts. The auxiliary-loss-free balancing result
  in particular has no numbers in the portion read.
- **Relevance at Chakaso scale:** MoE buys *capacity per FLOP at fixed active parameters*, and the enabling
  condition is that total capacity is large enough for specialised experts to emerge. Splitting a 640-wide
  feed-forward into 8 experts gives 8 experts of 640 that each see 1/8 of the tokens — 3.4M parameters spread
  over 27M total, at 19.66M positions, i.e. less than one epoch per expert at best. Router parameters, a
  load-balancing objective and a capacity-factor code path would all be new machinery, with E2608's
  token-starved arms as the counter-evidence: the binding constraint measured here is tokens, not capacity per
  FLOP. Also on this machine: the benefit is throughput-shaped (dispatch to different devices), and a CPU has
  no such dispatch to save.
- **Verdict: REJECT at this scale, and reject as a next unit.** The only defensible small-scale version of this
  idea is a *measured* one, and no source read here measures MoE below ~1B total parameters. Chakaso's next
  capacity step is the design document's dense 9.84M arm, not a sparse one.

### 5. Multi-head latent attention (MLA)

- **Source:** DeepSeek-V2 §architecture and Table 1,
  [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), 7 May 2024; reused unchanged in DeepSeek-V3 §2.2,
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437).
- **Mechanism:** down-project the hidden state to a low-rank latent (V2 reports **key/value latent rank 512**,
  query compression dimension 1,536), cache *that* instead of per-head keys and values, and up-project during
  decoding; positional information is preserved by a **decoupled RoPE key** carried in a separate shared slot,
  concatenated before the softmax.
- **Reported evidence:** "reduces the KV cache by 93.3%" and "boosts the maximum generation throughput to 5.76
  times" relative to DeepSeek 67B (abstract), with deployment above 50,000 tokens/s; V3 §4.5.1 reads as
  confirming the same architecture at 671B. **Critically: the attention-mechanism comparison in V2's Table 1 is
  qualitative tiers, not numbers** — as fetched, it rates MHA "Strong capability", grouped-query "Moderate",
  multi-query "Weak" and MLA "Stronger", with the cache footprint as the measured axis. There is no perplexity
  or loss delta for MLA-versus-MHA in the text read.
- **Stated limitations:** none for MLA in the portion fetched (V2's discussion was not reached). The absence of
  a quality ablation with numbers is itself the limit on how strongly "MLA also improves quality" can be
  claimed from the primary.
- **Reproducibility:** self-reported, single group, 236B/671B models; no independent replication.
- **Relevance at Chakaso scale:** the mechanism is a *compression*: it pays only when the un-compressed cache is
  large, i.e. when `n_heads × head_dim × layers` greatly exceeds the latent rank. Chakaso's per-position cache
  is 2 × 256 dims × 4 layers = 2,048 values; the reported V2 latent is **512 dimensions**, and at `d_model` 256
  a 512-wide latent is **wider than the model**. Adopting V2's numbers verbatim would *expand* the cache and the
  parameter count; adopting a scaled version (rank 64, say) trades a ~2 MiB cache that is not a constraint for
  extra projections and a decoupled-RoPE path. ADR-0035's checkpoint format and the E2608/E2611 comparison both
  assume the current attention layout.
- **Verdict: REJECT at `d_model` 256.** Its documented purpose — cluster inference economics — is absent here,
  and its own primary quality evidence is a qualitative tier table. If a future Chakaso decoder reaches a context
  where the cache is measured to matter, MLA and GQA should be compared as *cache* experiments with a
  capacity-matched control, not as capability experiments.

### 6. Multi-token prediction (MTP)

- **Sources:** Gloeckle, Idrissi, Rozière, Lopez-Paz, Synnaeve, *Better & Faster Large Language Models via
  Multi-token Prediction*, [arXiv:2404.19737](https://arxiv.org/abs/2404.19737), 30 Apr 2024 (ICML 2024) —
  **Meta**; DeepSeek-V3 §2.2, §4.2, §4.5.1 **Table 4**,
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — **DeepSeek**. This is the one mechanism with two
  independent groups' measurements.
- **Mechanism:** at each position, predict the next *n* tokens through *n* independent output heads on a shared
  trunk (Gloeckle); or append a **1-depth MTP module** that sequentially predicts one further token while
  preserving causal order, sharing embeddings and the final projection (DeepSeek, D = 1). DeepSeek's framing is
  explicit: "Our MTP strategy mainly aims to improve the performance of the main model, so during inference, we
  can directly discard the MTP modules and the main model can function independently and normally" (§2.2). Loss
  weight "λ is set to 0.3 for the first 10T tokens, and to 0.1 for the remaining 4.8T tokens" (§4.2).
- **Reported evidence:**
  - **Meta, 0.3B–13B.** "The method is increasingly useful for larger model sizes"; the figure label states that
    the multi-token models "are worse than the baseline for small model sizes". At **1.3B**, four-token models
    score **7.4 MBPP pass@1 (+0.6)** and **4.8 HumanEval pass@1 (+0.2)**; at **3B**, 12.7 MBPP / 7.2 HumanEval;
    at **13B**, "solves 12 % more problems on HumanEval and 17 % more on MBPP". Repetition keeps the advantage
    ("+2.4% increase on pass@1 on MBPP and +3.2% increase on pass@100 on HumanEval"). Per-step slowdown
    **1.02×–1.22×** (their Table S5), and 4-token models are "up to 3 times faster at inference" with
    speculative-style decoding. Crucially, the fetched text reports that gains concentrate in **generative
    benchmarks** while "standard multiple-choice and likelihood metrics frequently plateau or dip".
  - **DeepSeek, arms at 15.7B total (trained on 1.33T tokens) and 228.7B total (540B tokens), Table 4.** Small
    MoE baseline → with MTP: **Pile-test BPB 0.729 → 0.729** (no change), BBH EM 39.0 → 41.4, MMLU 50.0 → 53.3,
    DROP 39.2 → 41.3, TriviaQA 56.9 → 57.7, NaturalQuestions 22.7 → 22.3 (**worse**), HumanEval 20.7 → 26.8,
    MBPP 35.8 → 36.8, GSM8K 25.4 → 31.4, MATH 10.7 → 12.6. Large MoE: BPB 0.658 → 0.657, MMLU 67.5 → 66.6
    (**worse**). Speculative use at inference reported 85–90% token acceptance giving **1.8× TPS** (§5.4.3). No
    average is computed by the paper.
- **Stated limitations:** Meta's own is the sharpest line in the whole reading — **the method is worse than the
  baseline at small model sizes, and its usefulness grows with scale**. DeepSeek's Table 4 caption identifies
  these baselines as *not* the final DeepSeek-V3 architecture, and the report's §6 limitations were truncated
  by the fetch (**could not verify**).
- **Reproducibility:** the two groups agree in direction (downstream generative/math/code gains) and agree in the
  negative (likelihood-style metrics flat or down). This is the strongest reproducibility profile on the list,
  and it is still self-reported at every scale above 0.3B with an explicit sign change below ~1B.
- **Relevance at Chakaso scale:** read the metric column, not the headline. Chakaso's declared measurement is
  **held-out bits per character** — exactly the axis on which both reports show **no movement or a decline**
  (DeepSeek's BPB: 0.729 → 0.729 and 0.658 → 0.657; Meta: likelihood metrics "plateau or dip"). The smallest arm
  where Meta reports a *positive* effect is 1.3B, ~380× this model, and the smallest DeepSeek arm is 15.7B
  total. E2610's finding — the released checkpoint loops because greedy decoding of a human-diverse distribution
  loops — is a decoding-distribution problem, and MTP does not address it; it adds a second target on the same
  data, which at 5.8 tokens per parameter is a **split of an already insufficient budget**, since the auxiliary
  head competes for the same 19.66M positions.
- **Verdict: REJECT as a capability lever at 3.4M and at 9.84M.** The two independent sources are the reason to
  reject it *specifically* rather than to test it: the documented sign of the effect below ~1B is negative, and
  the metric Chakaso reports is the metric the reports show as unmoved. If ever revisited, the honest version is
  a paired arm with the MTP loss weight declared before the run (DeepSeek's λ = 0.3/0.1 is a usable template)
  scored on bits per character *and* a generation-quality suite, because a BPB-only readout would miss whatever
  effect exists.

### 7. The Llama-3 tokenizer and why the vocabulary is 128K

- **Sources:** Llama 3 §3.2 and Table 3, [arXiv:2407.21783v3](https://arxiv.org/abs/2407.21783); tokenizer
  lineage in Llama 1 §2 ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) and Llama 2 Table 1
  ([arXiv:2307.09288](https://arxiv.org/abs/2307.09288)); Mistral 7B Table 1 (`vocab_size` 32,000).
- **Mechanism and the reported reasoning.** Llama 3: "We use a vocabulary with 128K tokens", composed as
  "100K tokens from the tiktoken tokenizer with 28K additional tokens to better support non-English languages".
  The stated *measured* justification is compression: the change "improves compression rates on a sample of
  English data from **3.17 to 3.94 characters per token**", and "adding 28K tokens from select non-English
  languages improved both compression ratios and downstream performance". The lineage it replaced: Llama 1 and
  Llama 2 used a 32,000-entry SentencePiece BPE ("split all numbers into individual digits, and fallback to
  bytes to decompose unknown UTF-8 characters" — Llama 1 §2), Mistral 7B kept 32,000, DeepSeek-V2 uses 100,000
  and Llama 3 8B/70B/405B all use 128K.
- **Reported evidence:** the compression ratio (3.17 → 3.94 chars/token) and a downstream improvement attributed
  to the 28K non-English additions. No vocabulary-size sweep is reported in the text fetched — 128K is presented
  as a decision with a measured consequence, not as the winner of a controlled comparison across 32K/64K/128K.
- **Stated limitations:** none in the portion read (the herd paper's Limitations section was beyond the fetch
  truncation — **could not verify**). Note what the mechanism is *for*: multilingual coverage and code
  compression at 15T tokens, in a family whose context went 8K → 128K.
- **Reproducibility:** self-reported; the compression ratio is, however, an ordinary property of a tokenizer and
  a corpus, so this is one of the few claims in this document Chakaso can check on its own data (and does check
  — see the next field).
- **Relevance at Chakaso scale — this is the one mechanism whose *procedure* transfers, and whose *value* does
  not.** Chakaso measures its own compression, which is exactly the quantity Llama 3 used to justify 128K:
  **2.0993 characters per token** at vocabulary 4,096 on the multi-register licensed corpus
  (`experiments/E2612/record.md`), and 2.0829 on repository text / 1.8782 on three external prose works at
  vocabulary 1,024 (E2605, E2607). Against Llama 3's **3.94 characters per token**, this project's tokenizer
  compresses roughly half as well per position, on text that is 19th-century rather than web-multilingual. Two
  consequences, both arithmetic. (a) **At a fixed position budget, characters-per-identifier is how much of the
  corpus a run can physically touch.** E2612's pinned corpus is 149,000,499 characters — about 71M identifiers at
  the measured 2.0993 — against the training pools the record declares, ~25M identifiers for the prose arm and
  ~48M for the mixture arm, whose 20,000,000-identifier packing cap covers 80% and 42% of them. If a larger
  vocabulary raised compression to, say, 2.6 characters per identifier, the same 19,660,800 positions would reach
  proportionally more distinct works before the cap binds. (b) **The price is the parameter budget**: at
  `d_model` 256 each 4,096 identifiers adds 1,048,576 parameters (30.7% of the current model), which is why the
  design document holds 4k and why Llama's 128K is not a portable number: Llama 3 8B's table gives `d_model`
  4,096, so its 128,256-entry table is ~0.5B of 8B (~6%), while at 3.4M the same vocabulary would be most of the
  model. A larger vocabulary also reshapes the output distribution's support in a
  way that interacts with E2610's repetition finding.
- **Verdict: ADAPT (the measurement, not the number).** E2612 held the tokenizer fixed at 4,096 as
  ADR-0041 required, and that was right: it is why its mixture result is attributable to data rather than to
  segmentation. The sweep belongs before any vocabulary or scale decision, not inside a data arm; run it as
  **characters per identifier and held-out bits per character at 4,096 / 8,192 / 16,384 on the same
  pinned corpus and the same document split**, with the embedding-table share reported alongside, since a fair
  comparison must state what the vocabulary cost in parameters. The fitting cost is a real constraint here —
  E2612 fitted 3,839 merges on a seeded 12,000,000-character sample because the whole training side was not
  affordable — so the sweep must be sized by what the fitter can actually fit, and the sampled-fit difference
  recorded as a property of every arm. **REJECT 128K** as a target: it is a 15T-token multilingual-and-code
  decision, and its own justification is a compression ratio this project can measure directly on its own text.

### 8. Long context and the RoPE base-frequency change

- **Source:** Llama 3 §3.2 ("We increase the RoPE base frequency hyperparameter to 500,000"), §3.4.2 (context
  extension), Table 3; RoPE itself is Su et al.,
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864), Apr 2021. Llama 2's context is 4096 (Table 1); Llama 1
  2048.
- **Mechanism:** a larger RoPE base stretches the rotary wavelengths so that positions far apart remain
  distinguishable at long sequence lengths; the context itself is bought by continued pretraining in stages, not
  by the constant alone.
- **Reported evidence:** Table 3 gives θ = 500,000 for all Llama 3 sizes at a 128K window; §3.4.2 reports the
  capacity was expanded "incrementally across six phases" to 128K and that this stage consumed "approximately
  800B training tokens", with the acceptance test that "the model perfectly solves 'needle in a haystack' tasks
  up to that length". YaRN and PoSE were **not** found in the fetched excerpt despite being asked for — the herd
  paper's account, as read, is staged continued pretraining plus the base change. Llama 3's scaling trials used
  8K contexts (§3.2.1).
- **Stated limitations:** the needle test is reported qualitatively in the portion read; no per-length score
  table was reached (**could not verify**).
- **Reproducibility:** self-reported; the 800B-token figure makes the cost of the mechanism the whole story.
- **Relevance at Chakaso scale:** Chakaso's context is 256 positions and its θ is 10,000. At 256 positions the
  lowest-frequency rotary components make less than one full turn of their wavelength across the window, so
  raising θ buys resolution at distances the model never sees; and the 2048→4096→8192→128K ladder shows what the
  reported benefit actually cost (~800B tokens, i.e. **40× this project's entire token history** at the E2612
  budget of 19.66M). E2610's positioning check (teacher-forced loss flat from ~position 16 to 255) says the
  current window is already under-exploited by length; the design document accordingly treats 512 as its own
  later comparison.
- **Verdict: REJECT the RoPE-base change; TEST the context length as its own paired experiment, after the data
  result.** Raising θ at 256 positions has no stated precondition in the sources. Context 256 → 512 is cheap
  (2× positions/step, so half the steps for the same tokens) and testable here, but only as a single-variable
  arm with its own record — and note that longer context raises tokens-per-step, which *reduces* the corpus
  fraction a fixed 2,400-step budget can touch.

### 9. Warmup-stable-decay versus cosine

- **Sources:** Llama 1 §2 and Llama 2 §2 (cosine with warmup, 10% floor); Llama 3 §3.2.1/§3.4.1 (cosine);
  **WSD primary:** Wen, Li, Wang, Hall, Liang, Ma, *Understanding Warmup-Stable-Decay Learning Rates: A River
  Valley Loss Landscape Perspective*, [arXiv:2410.05192](https://arxiv.org/abs/2410.05192), v1 7 Oct 2024,
  v2 2 Dec 2024. (See Observation 4: the "Llama 3 Handbook" that is usually cited for WSD was
  **not verifiable**.)
- **Mechanism:** constant peak learning rate for an open-ended main branch, then a rapid decay "branch" when a
  model is wanted, instead of a cosine whose shape is fixed by the total step count decided in advance. The
  authors' account: under a "river valley" landscape the stable phase makes progress along the river while
  oscillating across the valley, and "the rapidly dropping learning rate minimizes the iterate's oscillations" —
  "the sustained high learning rate phase and fast decaying phase are responsible for progress in the river and
  the mountain directions respectively, and are both critical". Their WSD-S variant "reuses previous checkpoints'
  decay phases and keeps only one main branch, where we resume from a decayed checkpoint".
- **Reported evidence:** **models of 0.1B, 0.3B, 0.6B and 1.2B parameters on the Pile**, at compute targets of
  **50B, 100B and 200B tokens**. WSD-S "outperforms Cyclic-Cosine with a significant performance gap of 4e-2"
  and "delivers competitive results compared to Cosine-Oracle in a single run" — i.e. budget-matched cosine still
  wins slightly when its endpoint is known, but WSD gets near it *without* separate runs per budget. The theory
  is also empirical at a smaller unit still: the landscape "can emerge from pretraining on a simple bi-gram
  dataset".
- **Stated limitations (their own):** "only consider a 1-dimensional river in the theoretical analysis above",
  a straight stochastic path is assumed, strict Hessian bounds are required, and tighter algorithmic constants
  are left open. Their experiments are 0.1B–1.2B — **30× to 350× this model** — on a general-web/Pile mixture,
  not on archaic prose, and the comparison is against *Cyclic*-Cosine, not the plain fixed-budget cosine Chakaso
  runs.
- **Reproducibility:** self-reported by one group (Stanford/Berkeley-adjacent theory authors), but the mechanism
  is scale-shallow and cheap to check, and the claim structure is verifiable: no architecture change, no extra
  parameters, no data change.
- **Relevance at Chakaso scale:** this is the **only mechanism on the list whose primary evidence sits within two
  orders of magnitude of Chakaso rather than three or four** — and the failure mode it addresses is the one this
  project has now measured twice. E2608's arm A reached its 2,400-step ceiling with validation still falling, and
  E2612 states it for both arms: `budget_ceiling` at 2,400 steps, `stopped_early: false`, "validation still
  falling in the last interval", "**Both arms are floors, not converged values**". A cosine schedule forces the
  decay endpoint to be chosen before the run and makes a longer run a different experiment; WSD lets the same
  branch train past 2,400 and decay on demand, producing **several budgeted checkpoints from one main branch**.
  On this machine that converts a wrong guess about the right budget into one run with N decays, and the measured
  cost is small: the record puts a full 2,400-step arm at ~73 minutes of gradient work and ~88–90 minutes wall
  including evaluation, so a branch plus a few decays is hours,
  while a mis-set cosine endpoint costs a whole re-run.
- **Verdict: TEST EXPERIMENTALLY — the single most transferable item in this document.** Implement as
  `wsd` alongside `cosine` and `constant`, then a paired arm at matched tokens on the E2612 corpus, same seed,
  same split, schedule as the only variable; the honest primary comparison is *final bits per character* versus
  cosine-at-same-total-budget, plus the free information of the intermediate decays. It is a schedule, not
  architecture, so it does not confound a data experiment if run as its own record — but it must not be
  smuggled into one: ADR-0041 holds the schedule fixed inside a data arm, and E2612 honoured that. Do **not**
  adopt it on the argument
  "Llama 3 uses it": the verifiable Meta primaries say cosine, and that claim is one this document could not
  source.

### 10. FP8 mixed-precision training and its loss scaling

- **Source:** DeepSeek-V3 §3.3 (training precision) and §3.1/Table 1,
  [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), v2 18 Feb 2025.
- **Mechanism:** core matrix multiplications (forward, activation-gradient, weight-gradient) and the MoE
  dispatch/combine traffic in **FP8**, with embeddings, attention, gating and normalisation kept in higher
  precision; outliers handled by **dynamic group scaling** over 1×128 activation tiles and 128×128 weight tiles
  (a micro-scaling format: a shared scale per tile).
- **Reported evidence:** "the relative loss error of our FP8-training model remains consistently below 0.25%"
  against a BF16 baseline; total cost 2.788M H800 GPU-hours, ~$5.576M under rental assumptions, 14.8T tokens;
  and the stability claim that the run "did not experience any irrecoverable loss spikes or perform any
  rollbacks" (abstract, §1) across training.
- **Stated limitations:** the §6 discussion was **truncated by the fetch and could not be verified**; the FP8
  validation figure is reported as a loss-error bound, not as a downstream-benchmark comparison, and the numbers
  are H800-specific — FP8 here depends on Hopper tensor cores and on a tile-scaled format the CPU cannot
  accelerate.
- **Reproducibility:** self-reported, one group, one cluster; the loss-spike-free claim is a claim about
  14.8T tokens and cannot be evaluated at 19.66M positions, where no spike has ever been observed here.
- **Relevance at Chakaso scale:** none, and the mismatch is total. There is no FP8 path on an x86 CPU at 18
  cores; the declared matrix is dependency-free-by-default with `numpy` in `[train]` and `torch` in `[nn]`
  ([ADR-0034](../../decisions/ADR-0034-optional-pytorch-model-extra.md)), and float32 is what the measured
  throughput and the checkpoint format
  ([ADR-0035](../../decisions/ADR-0035-transformer-checkpoint-format.md)) assume. Lower precision here would be
  bfloat16-in-software at best — a speed loss on a CPU, not a gain — and the numerical-provenance rule ("a
  measured result must stay re-trainable") argues against silently changing dtype mid-family, since bits per
  character across dtypes is not a comparable record. DeepSeek's *other* precision-adjacent ideas are worth
  naming for what they say about stability, not for adoption: their stability came from scale, batch discipline
  and a schedule, none of which is the thing keeping Chakaso's loss falling.
- **Verdict: REJECT.** Pure cluster cargo-cult at this scale, with no CPU implementation and no measured problem
  of that kind here. Chakaso's precision decision is already correct and should be left alone.

### 11. DeepSeek-R1's reinforcement-learning curriculum (and its distillation half)

- **Sources:** *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*,
  [arXiv:2501.12948](https://arxiv.org/abs/2501.12948), v1 22 Jan 2025, v2 4 Jan 2026; GRPO is from DeepSeekMath
  ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300), Feb 2024). **Distillation figures:** the official model
  card [huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
  (DeepSeek-AI org; read 2026-11-01) — the report's own Appendix F was **truncated by the fetch and could not be
  verified**, so the small-student numbers are taken from the release card rather than the paper.
- **Mechanism:** four stages — (i) cold-start supervised fine-tuning on "thousands of high-quality, diverse
  reasoning prompts", first-person chain-of-thought, trajectories generated by R1-Zero at temperature 1.0 and
  filtered "to retain only those with correct final answers and a readable format" (SymPy verification,
  repetition and cross-language checks); (ii) rejection sampling plus SFT on mixed reasoning and general
  instruction data; (iii) reasoning RL with **GRPO**, which "comput[es] the advantage … using a group of rewards"
  instead of a learned critic, over **173,000 curated prompts** (26k math, 25k coding, 22k STEM, 15k logic, 78k
  general/helpfulness-harmlessness); (iv) a second RL pass of 1,700 steps, the last 400 dedicated to
  helpfulness/harmlessness. Rewards are rule-based: binary accuracy (deterministic match or compiler execution)
  plus a format reward requiring `<think>`/`</think>`.
- **Reported evidence:** R1-Zero (RL without SFT) shows "a significant jump at the 8.2k step" with quadrupled
  mathematical pass rates and a spontaneous self-correction behaviour described as an "aha moment … characterized
  by a sudden increase in the use of the word 'wait' during reflections". Unconstrained RL produced
  "language mixing", which motivated a language-consistency penalty. **The distillation table is the part that
  speaks to Chakaso's scale:** `DeepSeek-R1-Distill-Qwen-1.5B` — a ~1.5B-parameter student, "fine-tuned … using
  samples generated by DeepSeek-R1", "finetuned with 800k samples curated with DeepSeek-R1" — reports **AIME 2024
  pass@1 28.9 / cons@64 52.7, MATH-500 83.9, GPQA Diamond 33.8, LiveCodeBench 16.9, CodeForces 954**, against
  GPT-4o-0513's 9.3 / 74.6 / 49.9 / 32.9. The 7B/14B/32B students are higher still (32B: 72.6 AIME, 94.3
  MATH-500, 62.1 GPQA), and all are MIT-licensed derivatives of Apache-2.0 Qwen bases.
- **Stated limitations (verbatim from the report as fetched):** "Currently, the structural output capabilities of
  DeepSeek-R1 remain suboptimal compared to existing models."; "instances of excessive reasoning—manifested as
  overthinking—are still observed in response to simpler questions."; it "might use English for reasoning and
  responses, even if the query is in a language other than English or Chinese."; "Few-shot prompting
  consistently degrades its performance."; "large-scale RL has not been applied extensively in software
  engineering tasks."; and "If the reward signal is assigned by a model instead of predefined rules, it becomes
  more susceptible to exploitation". The card adds an operational limit that matters here: it recommends
  temperature 0.5–0.7 "to prevent endless repetitions or incoherent outputs" and instructs users to avoid a
  system prompt.
- **Reproducibility:** self-reported; nobody outside the group has re-run RL at that scale, and the distilled
  students' scores are the authors' own evaluation. The card's own framing — students "fine-tuned based on
  open-source models" — is the honest description of what is being claimed.
- **Relevance at Chakaso scale:** split the two halves.
  - **RL curriculum: REJECT, and it is not close.** GRPO needs a base that already produces a correct final
    answer *sometimes* (the reward is binary and gated on a verifiable answer format), a rule-based verifier,
    ~173k prompt environments, and a base model whose generations are worth sampling. Chakaso's measured base
    writes repetitive text and hits its token bound on every sample (E2610); a binary accuracy reward over such
    generations is a constant-zero reward, which teaches nothing. ADR-0041 already defers post-training, and this
    reading confirms the reason rather than the timing.
  - **Distillation: the finding, not the recipe.** Every small model with *published capability* in this set
    got it from teacher-generated text, not from an architecture: the 1.5B student's 83.9 on MATH-500 is 800k
    traces from a 671B model. That pairs with TinyStories, whose abstract puts coherent multi-paragraph English
    in models "below 10 million total parameters" trained on a dataset "generated by GPT-3.5 and GPT-4"
    ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759), 12 May 2023). **Interpretation, not a reported fact:**
    at Chakaso's scale the literature's demonstrated capability lever is *what the model is shown*, and the two
    demonstrations of it both require a hosted teacher. Under
    [ADR-0033](../../decisions/ADR-0033-model-centred-product-contract.md) and the no-invented-capability rules,
    a teacher-derived mixture is admissible only as a labelled, provenance-carrying data source — and it would
    change what the released artifact is allowed to claim. That is a decision record, not an engineering tweak.
    **Verdict on this branch: REJECT for the base model Chakaso is trying to train from random initialization;
    RECORD it as the honest answer to "what would raise measured capability fastest at 1–10M parameters", because
    the evidence points at data and away from every architecture on this list.**
- **Verdict: REJECT the RL curriculum now; treat the distillation evidence as confirmation that the next unit is
  data, exactly as ADR-0041 decided.**

---

## What the ledger adds up to

**Result (of the reading, not of an experiment).** Of eleven mechanisms, the distribution of verdicts is:
**0 adopt as published, 1 adapt (tokenizer compression measurement), 2 test experimentally (WSD; context length),
8 reject at this scale.** And the rejects split into two different reasons, which matters:

- **Reject because the precondition is absent here** — MQA, GQA, SWA, MLA, FP8, MoE. Each is an economics
  mechanism for bandwidth, device dispatch, tile-scaled floating point or cache size at 7B–671B. Chakaso's cache
  is 2 MiB, its context is 256, its device has no FP8 unit and its run costs about an hour of CPU. Transplanting
  them is cargo-culting: they would change the parameter count and the code path without touching anything the
  sources measure as the constraint.
- **Reject because the sources themselves measure a sign change below their scale** — MTP is the clearest: Meta's
  own figure label says the multi-token models "are worse than the baseline for small model sizes", and DeepSeek's
  own Table 4 shows the LM-loss axis (BPB) unmoved even at 15.7B. This is the most valuable category, because the
  cargo-cult version of MTP would have looked like a cheap win on a downstream-metric reading and is a documented
  wash or loss on a bits-per-character reading, which is Chakaso's metric.

**Interpretation.** The three things in these reports that are *not* scale-conditional are: the learning-rate
schedule question (evidence at 0.1B–1.2B), the tokenizer compression measurement (a property of a tokenizer and a
corpus, and Chakaso already measures it: **2.0993 characters per token** at vocabulary 4,096 against Llama 3's
3.94 at 128K), and data (TinyStories and the R1 distillation table, both of which put capability in small models
by way of *teacher-generated* text). Everything else the frontier reports about its architectures is, on the
primaries' own numbers, about being a cluster artifact. That is a stronger version of what ADR-0041 decided than
the landscape document gave it: the architecture bundle is not merely "premature" here, it is documented as
*negative or null* at the metric and scale Chakaso has.

**And the internal measurement now agrees with the external reading.** E2612 changed one variable — the corpus —
at this configuration and moved held-out bits per character by more than any mechanism in this ledger claims at
this scale. Its numbers, its decision rule and the sequence it declares are in
[`../../../experiments/E2612/record.md`](../../../experiments/E2612/record.md); what matters here is the
direction of agreement: the literature's near-scale evidence points at data, schedule and tokenizer, and the one
data experiment this project has run confirmed the data half of that in the predicted direction.

**Limitation of this document.** It is a reading, not a measurement. No mechanism in the ledger was run here;
verdicts are inferences from other authors' numbers plus arithmetic against the declared configuration, and where
the arithmetic is load-bearing (parameter shares, cache bytes, corpus fraction reachable at a fixed budget) it is
labelled as derived rather than measured. Fetch truncation left the specific gaps listed below — including
DeepSeek-V3's §6 limitations, which means the V3 verdict rests on its architecture and precision sections rather
than on the authors' own caveats — and every benchmark figure in this document remains **self-reported by its
authors**, with the single exception of multi-token prediction, which two independent groups measured and both
report as unmoving on language-modelling loss.

---

## What this reading adds to the sequence the records already declare

**What has already been measured.** E2612 ran on 2026-10-30 at exactly this configuration, changing one variable
(the corpus), and its declared outcome is **decision rule 2 — H1 true, H2 false**: the technical mixture is
**0.27584 bits per character** better on the technical aggregate (13.8× the materiality band, better on all four
technical partitions, mathematics gaining 0.46584) while prose is **0.03026 bits per character** worse, against a
tolerance of 0.02. Rule 1's scale step is explicitly **not** licensed by it; the record's own next-unit list —
E2613's P½ exposure control, then a prose-share reweighting sweep, then a modern technical register, then more
held-out works per register — is the sequence, and this document does not restate or reorder it.

Both arms reached the declared 2,400-step ceiling with `stopped_early: false` and validation still falling in the
last interval; the record's words are "**Both arms are floors, not converged values**". Wall cost was 90.4 and
87.4 minutes per arm on CPU (~73 minutes of gradient work).

**Interpretation.** The data lever the literature points at is now measured here, and it is the largest effect
this project has recorded: a corpus change moved held-out bits per character by a quarter of a bit at a fixed
architecture, while all eleven mechanisms in the ledger are, on their own papers' evidence, inert or negative at
this scale. Two arms follow from that reading which the record's list does not contain, and both are
non-architecture, both are single-variable, and neither competes with E2613 for the next slot:

1. **Warmup-stable-decay versus cosine, at matched tokens.** Same model, corpus, split, seed and budget as the
   better arm of E2612, `wsd` implemented as a schedule with the decay length declared before the run. The reason
   is the ceiling itself: cosine's endpoint must be chosen in advance, so "was 2,400 steps right?" can only be
   answered by re-running, whereas a WSD main branch yields several decayed checkpoints from one branch at
   ~73 minutes of gradient work each. Score on held-out bits per character against cosine at the same total
   budget, and report every branch rather than the best one. It is a schedule, not architecture, so it does not
   confound a data arm — but it must not be smuggled into a data experiment, because ADR-0041 holds the schedule
   fixed within one. Do not land it on the argument "Llama 3 uses it": the verifiable Meta primaries say cosine,
   and that claim is one this document could not source.
2. **A vocabulary sweep, decided on measured compression.** Characters per identifier, identifier counts, and
   held-out bits per character at 4,096 / 8,192 / 16,384 merges on the same pinned corpus and the same document
   split, with the embedding-table share reported for each arm. This is the quantity Llama 3 used to justify its
   128K (3.17 → 3.94 characters per token), it is the one frontier argument on this list that is a property of a
   tokenizer and a corpus rather than of scale, and Chakaso already measures it (2.0993 at 4,096). The counterweight
   is arithmetic: 1,048,576 parameters per 4,096 identifiers at `d_model` 256, so the sweep must report what each
   vocabulary costs, not only what it gains. Needs a `Hypothesis` entry in
   [`../../research/hypotheses.md`](../../research/hypotheses.md) before it runs, and the sampled-fit constraint
   E2612 recorded (12,000,000-character fitting sample) stated as a property of every arm.
3. **Context 256 → 512, later and as its own paired run**, with the budget consequence stated: more positions per
   step means the fixed 2,400-step ceiling touches *less* of the corpus, which at a cap-limited exposure is the
   wrong direction.
4. **Not now, and not as an aside:** GQA/MQA (needs a bandwidth-bound decode to justify it, plus a
   capacity-matched control, because at 8 heads and `d_model` 256 it removes 13.4% of the parameters), SWA (a
   no-op at 256 — its reported window is 4096), MoE/MLA/FP8/MTP (preconditions absent, or the sign measured
   negative below ~1B), and any RL stage (no base that a verifiable reward can score).

---

## Could not verify

Recorded so that nobody re-reads these sources believing they were already read.

- **"Llama 3 Handbook" as the source of Llama 3's WSD schedule.** Searched twice by title and identifier;
  `arXiv:2404.07944` resolves to *"On the Durdevic approach to quantum principal bundles"* (Del Donno, Latini,
  Weber, 11 Apr 2024). No arXiv record for the Handbook was located. The herd paper's own text says cosine.
- **Mistral 7B and Mixtral 8x7B training budget.** No total tokens, batch size, learning rate, schedule,
  optimizer or GPU count in the fetched text (Mistral HTML v1, ar5iv rendering, and both abstracts). Reported as
  absent from what could be read, not as absent from the paper.
- **Mistral 7B limitations and any SWA quality ablation.** The fetch found no limitations section in the portion
  read; the "without impacting the model quality" claim was not backed by a table in that portion.
- **DeepSeek-V3 §4.5.1's prose around the MTP arms beyond the parameter counts, and §6 (Discussions and
  Limitations).** The arXiv HTML truncated at §4.4.2 on three attempts; §4.5.1's Table 4 and the D = 1 /
  λ = 0.3 → 0.1 details were recovered from the Hugging Face rendering of the same report
  (`huggingface.co/papers/2412.19437`) — same document, mirror access.
- **DeepSeek-R1 Appendix F (distillation) and section B.3.3 (SFT sample counts).** Both beyond the fetch
  truncation; the student-model benchmark table above comes from the official DeepSeek model card instead.
- **Llama 3's Limitations section** (§5 onward) and any needle-in-a-haystack score table: the HTML fetch cut off
  before §5.
- **Llama 1 context length, vocabulary size and GPU-hours.** Table 2 as returned gave layers/dimension/heads and
  the 1.0T/1.4T token counts and the SentencePiece-BPE description, but not the window, the 32,000 vocabulary or
  the A100-hours figure. Those are commonly quoted as 2048 and 32,000; this document does not assert them,
  because this read did not reach them.
- **MQA's quantitative speed-up and memory figures.** Abstract only; the technical note's numbers were not read.
- Any claim that these results have been **independently reproduced**: none was found in this reading, and every
  figure above is self-reported by its own authors.

---

## References

Primary sources, all accessed 2026-11-01, with the version read.

1. Touvron, H. et al. *LLaMA: Open and Efficient Foundation Language Models.*
   [arXiv:2302.13971](https://arxiv.org/abs/2302.13971), 27 Feb 2023.
2. Meta AI blog, *Introducing LLaMA: A foundational, 65-billion-parameter large language model*, 24 Feb 2023,
   and Meta's publication page [ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/).
3. Touvron, H. et al. *Llama 2: Open Foundation and Fine-Tuned Chat Models.*
   [arXiv:2307.09288](https://arxiv.org/abs/2307.09288), 18 Jul 2023 (v2 19 Jul 2023). Read the v2 HTML:
   Table 1 (2.0T tokens, 4k context, 4M-token global batch, GQA ✗ at 7B/13B and ✓ at 34B/70B), §2 optimizer
   language, §5 limitations.
4. Grattafiori, A. et al. *The Llama 3 Herd of Models.*
   [arXiv:2407.21783](https://arxiv.org/abs/2407.21783), v1 31 Jul 2024, v3 23 Nov 2024. Read §3.2/Table 3
   (128K vocabulary, 8 KV heads, θ = 500,000), §3.2.1 and §3.4.1 (cosine schedules), §3.4.2 (six-phase context
   extension, ~800B tokens), §1 (15T tokens against "1.8T tokens for Llama 2").
5. Jiang, A. Q. et al. *Mistral 7B.* [arXiv:2310.06825](https://arxiv.org/abs/2310.06825), 10 Oct 2023. Read the
   abstract and §2 (SWA with W = 4096, rolling buffer, "8x" cache at 32k, "2x" at 16K) and Table 1.
6. Jiang, A. Q. et al. *Mixtral of Experts.* [arXiv:2401.04088](https://arxiv.org/abs/2401.04088), 8 Jan 2024.
   Abstract: 47B total / 13B active, 8 experts, top-2 routing, 32k context.
7. Liu, A. et al. (DeepSeek-AI). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language
   Model.* [arXiv:2405.04434](https://arxiv.org/abs/2405.04434), v1 7 May 2024. Abstract (236B/21B, 8.1T tokens,
   128K context, 42.5% training-cost saving, 93.3% KV-cache reduction, 5.76× throughput), the MLA section
   (latent 512, query compression 1,536, decoupled RoPE key), Table 1, 172,800 GPU-hours per trillion tokens.
8. DeepSeek-AI. *DeepSeek-V3 Technical Report.*
   [arXiv:2412.19437](https://arxiv.org/abs/2412.19437), v1 27 Dec 2024, v2 18 Feb 2025; §4.5.1 Table 4 and §6
   recovered from [huggingface.co/papers/2412.19437](https://huggingface.co/papers/2412.19437) (mirror of the same
   report). 671B/37B, 14.8T tokens, 2.788M H800 hours on 2048 GPUs, §2.1.2 balancing, §2.2 and §4.2 MTP, §3.3 FP8.
9. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.*
   [arXiv:2501.12948](https://arxiv.org/abs/2501.12948), v1 22 Jan 2025, v2 4 Jan 2026. Four-stage pipeline, GRPO,
   173k RL prompts, the 8.2k-step jump, the "aha moment", language mixing, and the limitations quoted above.
10. DeepSeek-AI. Model card, *DeepSeek-R1-Distill-Qwen-1.5B.*
    [huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),
    read 2026-11-01. Distillation table, 800k-sample provenance, MIT/Apache derivation chain, sampling
    instructions.
11. Ainslie, J. et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.*
    [arXiv:2305.13245](https://arxiv.org/abs/2305.13245), v1 22 May 2023, v3 23 Dec 2023. Table 1 (47.2 / 47.1 /
    46.6; 1.51 s / 0.28 s / 0.24 s), 8 groups, 5% uptraining compute, T5.1.1 Large/XXL only.
12. Shazeer, N. *Fast Transformer Decoding: One Write-Head is All You Need.*
    [arXiv:1911.02150](https://arxiv.org/abs/1911.02150), 6 Nov 2019. Abstract only.
13. Gloeckle, F. et al. *Better & Faster Large Language Models via Multi-token Prediction.*
    [arXiv:2404.19737](https://arxiv.org/abs/2404.19737), 30 Apr 2024. 0.3B–13B, the small-model regression, the
    1.3B/3B/13B deltas, 1.02×–1.22× overhead, 3× inference speed-up, and the flat-or-down likelihood metrics.
14. Wen, K. et al. *Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective.*
    [arXiv:2410.05192](https://arxiv.org/abs/2410.05192), v1 7 Oct 2024, v2 2 Dec 2024. The Pile, 0.1B–1.2B,
    50B/100B/200B-token budgets, WSD-S versus Cyclic-Cosine (4e-2) and Cosine-Oracle, and their theory limitations.
15. Su, J. et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.*
    [arXiv:2104.09864](https://arxiv.org/abs/2104.09864), Apr 2021. RoPE itself; the Llama-3 base change is read
    from source 4, not from this paper.
16. Eldan, R. and Li, Y. *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*
    [arXiv:2305.07759](https://arxiv.org/abs/2305.07759), v1 12 May 2023, v2 24 May 2023. Coherent multi-paragraph
    English "below 10 million total parameters", dataset "generated by GPT-3.5 and GPT-4".
17. Shao, Z. et al. *DeepSeekMath.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300), Feb 2024. Origin of GRPO,
    cited by source 9.

Repository-side sources for the configuration arithmetic:
`experiments/E2608/record.md`, `experiments/E2610/record.md`, `experiments/E2611/record.md`,
`experiments/E2612/record.md` (2.0993 characters per token at vocabulary 4,096; the declared pool sizes and the
20M packing cap), `experiments/E2607/record.md` (1.8782 characters per identifier at vocabulary 1,024),
`experiments/E2605/record.md`, `MODEL_CARD.md`, `docs/decisions/ADR-0034`, `ADR-0035`, `ADR-0037`, `ADR-0040`,
`ADR-0041`, [`docs/research/open-model-landscape.md`](../../research/open-model-landscape.md),
[`docs/research/chakaso-next-training-design.md`](../../research/chakaso-next-training-design.md).

Secondary-by-access but primary-by-content: source 8's Table 4 and source 10's benchmark table were read from
Hugging Face-hosted renderings of the reports and the official release card, not from the arXiv PDF, because the
arXiv HTML truncated. This is noted per claim above rather than buried.
