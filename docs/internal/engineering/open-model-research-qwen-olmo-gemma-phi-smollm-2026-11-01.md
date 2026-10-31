# Open-model research for a sub-10M CPU decoder: Qwen, OLMo, Gemma, Phi, SmolLM, TinyLlama

**Status:** Research — a sourced comparison read against primary technical reports and
official project materials. It is **not** a measurement of any model in this repository, and
nothing here is independently reproduced by Chakaso.
**Date:** 2026-11-01. **Access date for every source:** 2026-11-02 (all arXiv PDFs/HTML and
official blogs were read that day; the report revisions cited are the latest I could open).
**Scope:** the mechanisms that bear on Chakaso's current artifact — a 3,413,760-parameter rotary
decoder (d_model 256, 4 layers, 8 heads, FFN 640, tied embeddings, RoPE θ=10,000, RMSNorm, GELU,
context 256, **4,096-identifier BPE**, AdamW lr 6e-4 / 200-step warmup / cosine to 0.1 peak, 2,400
steps = 19.66M tokens, float32 CPU, ~3,900 positions/s, no GPU), trained on archaic public-domain
Project Gutenberg English, producing **no usable text** and measured only in held-out
bits-per-character. See E2608/E2611 records and
[`../../research/open-model-landscape.md`](../../research/open-model-landscape.md), which is the
broad nine-family table; this document is the **mechanism-level, small-scale read** of six
families, with ADOPT / ADAPT / TEST EXPERIMENTALLY / REJECT verdicts for Chakaso's scale.

**Reading protocol.** arXiv IDs, formulas and every number below were transcribed from the source
itself (HTML where it rendered, otherwise the downloaded PDF text, otherwise the official project
blog/card). Where a commonly-repeated claim was **not** in the primary text I could open, that is
stated as *could not verify* — including two claims this project might otherwise have cargo-culted
(see "Corrections to the brief"). Every figure is **as reported by the cited authors**; "Reproduced
here?" is "no" for all rows unless a Chakaso record says otherwise.

**Labels.** *Primary* = the family's own technical report or the official team's blog/card.
*Secondary* = a third-party study that aggregates others' tokenizers/models; flagged as such inline.

---

## Summary table (one line per mechanism)

| # | Mechanism | Source (family) | Verdict for Chakaso | One-line reason |
|---|---|---|---|---|
| 1 | Byte-level BPE, 151,669 vocab | Qwen2/Qwen3 (primary) | **REJECT at this scale** | vocab sized for multilingual/frontier; embedding table would dominate a 3.4M model |
| 2 | vocab-size ∝ model-size scaling law | 2407.13623 (primary) | **ADAPT / TEST** | optimum *rises* with size (γ≈0.83); a small vocab is the *correct direction*; sweep V locally |
| 3 | data-quality-over-quantity classifier filtering | SmolLM2, Phi (primary) | **TEST EXPERIMENTALLY** | the one finding plausibly scale-free; register/classifier mix is Chakaso's live E2612 unit |
| 4 | overtraining a small model (tokens ≫ Chinchilla) | TinyLlama, SmolLM2 (primary) | **ADOPT (already doing it)** | Chakaso's 2,400-step ceiling with falling val is the same regime; extend the budget |
| 5 | knowledge distillation into small models | Gemma 2/3, Qwen3 (primary) | **TEST EXPERIMENTALLY** | Gemma's own ablation shows gains down to 200M; needs a teacher Chakaso does not yet have |
| 6 | full-run reproducibility artifacts (data+logs+all ckpts) | OLMo / OLMo 2 (primary) | **ADAPT (cheap subset)** | Chakaso already pins snapshots; add per-step logs + intermediate checkpoints + dataloader seed |
| 7 | logit soft-capping (tanh, C=50/30) | Gemma 2 (primary) | **REJECT** | stated as a *stability* device; Gemma 3 **removed** it and replaced it with QK-norm |
| 8 | alternating local/global attention | Gemma 2/3 (primary) | **REJECT / low priority** | benefit documented at 9B/27B; at context 256 there is nothing to window |
| 9 | PerToken-Fn activation | Gemma (task premise) | **REJECT (premise unverified)** | Gemma 2's Table 1 lists plain **GeGLU**; no small-model PerToken-Fn ablation found |
| 10 | synthetic textbook data + curriculum | Phi-1→4 (primary) | **REJECT wholesale / keep the idea** | the recipe needs a frontier teacher + 20–300B-token scale; archaic-English domain is the only transferable part |

---

## Priority 1 — Small-model-specific evidence (SmolLM, TinyLlama, Phi)

### 1a. SmolLM2 data-centric training
- **Source:** *SmolLM2: When Smol Goes Big — Data-Centric Training of Small Language Models*, arXiv
  2502.02737v1, 2025-02-04 (primary, HuggingFace / Corex). Blog companion (primary team):
  `huggingface.co/blog/smollm`, July 2024.
- **Mechanism:** violate Chinchilla deliberately — train sub-billion models on far more tokens than
  compute-optimal, and win capability by *classifier-filtered* high-signal data instead of raw
  volume. Tokenizer is a fixed byte-level BPE, **vocabulary 49,152** ("trained on the SmolLM corpus
  with a vocab size of 49152" per the blog).
- **Reported evidence (authors):** model/token grid **135M → 2T**, **360M → 4T**, **1.7B → 11T**
  tokens (≈14,815 / 11,111 / 6,471 tokens-per-parameter). English web mix is **60% FineWeb-Edu + 40%
  DCLM** (Table 1: MMLU 37.0, ARC 56.0). Code: applying an educational classifier to StarCoder data,
  **40B Python tokens → 4B kept (top ~10%)**, "converges more than **3× faster**", reaching **16%
  pass@1 after only 12B tokens**; MultiPL-E Python **20.7 → 25.6**, C++ **16.7 → 24.8** (Table 2).
  Math: FineMath 3-scale/5-scale Llama classifiers; "10B high-reasoning tokens double GSM8K versus
  larger, lower-signal sets."
- **Stated limitations (authors):** pretraining cost ≈ "**$250,000 USD** worth of GPU compute"; a
  "**loss spike**" at phase three forced a checkpoint rollback; small architectures "necessitate
  trimmed instruction corpora" (they drop function-calling to avoid capacity overload).
- **Reproducible here?** No. Their smallest model (135M) is ~40× Chakaso; their data pipeline needs
  a 1-trillion-token web corpus and a Llama annotator that Chakaso does not have.
- **Relevance at Chakaso scale:** the *rank-by-classifier-then-train-on-the-top-10%* mechanism is the
  most scale-portable idea in the whole review — it is a statement about *marginal data value*, not
  about model size. But Chakaso's only corpus is ~9M chars of one register, already seen ~4.2×;
  there is no large pool to rank down.
- **Verdict: TEST EXPERIMENTALLY (data-selection axis only).** Reframe the SmolLM result for Chakaso
  as: with a fixed tiny budget, *which subset of the register mixture earns the lowest held-out
  bits/char?* This is adjacent to the live E2612 equal-token data experiment (prose vs technical
  mixture). The 49,152 vocab and 2–11T token budgets are **not** transferable and must not be cited
  as if they were.

### 1b. TinyLlama overtraining baseline
- **Source:** *TinyLlama: An Open-Source Small Language Model*, arXiv 2401.02385 (primary).
- **Mechanism:** a 1.1B decoder that "follow[s] the same architecture and tokenizer as Llama 2"
  (**vocabulary 32,000**), deliberately trained past the scaling-law optimum. hidden 2048, 22 layers,
  32 heads.
- **Reported evidence (authors):** aimed at "cumulatively processing **3 trillion tokens**"; the
  released v1.1 checkpoint ≈ **2T** (≈**2,727 tokens/parameter**). Trained on **16 × A100-40G**; the
  authors report "implementation issues" in the LR scheduler and data loading during the live run.
  Base results: **HellaSwag 61.47, MMLU 26.58, HumanEval 6.71**; "surpasses both OPT-1.3B and
  Pythia-1.4B." Rationale quoted: "existing scaling laws may not predict accurately in situations
  where smaller models are trained for longer periods."
- **Stated limitations (authors):** positioned as a "lightweight platform for testing … innovative
  ideas," i.e. a **baseline, not a competitive model**; the **base** model's HumanEval **6.71** is
  near-floor (they note code needs domain-specific continued training, and regional/math variants
  use different splits like "75% Slimpajama / 15% Starcoder / 10% Proof pile").
- **Reproducible here?** The *conclusion* (a small model at a huge multiple of its optimal token
  count beats a comparably-sized undertrained model) is consistent with Chakaso's own
  E2608/E2611 finding that both arms were **still improving at the 2,400-step ceiling**. The *numbers*
  are not.
- **Relevance at Chakaso scale:** strongest match to Chakaso's actual situation — an under-trained
  small decoder. Two directly usable lessons: (i) Chakaso's 19.66M tokens is ~5.8 tokens/param for a
  3.4M model — *nowhere near* overtrained, and the falling validation loss says the ceiling is the
  binding constraint, not the architecture; (ii) TinyLlama **inherits a proven tokenizer** rather than
  training its own — a caution for Chakaso's home-grown 4,096 vocab (see 3a).
- **Verdict: ADOPT the regime (train past compute-optimal), already Chakaso's implicit stance.** The
  concrete action is to raise the step budget and confirm the loss floor, not to change architecture.

### 1c. Phi-1 / Phi-1.5 "textbook-quality data" (and the tokenizer misattribution)
- **Source:** *Textbooks Are All You Need* (phi-1), arXiv 2306.11644v2 (primary); *Textbooks Are All
  You Need II: phi-1.5 technical report*, arXiv 2309.05463 (primary).
- **Mechanism:** 1.3B decoder (24 layers, hidden 2048, MLP 8192, 32 heads of dim 64, RoPE dim 32)
  trained on "textbook-quality" web data (6B tokens) + GPT-3.5-generated synthetic textbooks
  (1B tokens); phi-1.5 adds ≈20B synthetic tokens.
- **Reported evidence (authors):** phi-1 **50.6% pass@1 HumanEval**, **55.5% MBPP**; the 350M
  phi-1-small **45% HumanEval**, beating several same-scale open code models.
- **Correction that matters (verified against the PDF text of both papers):** phi-1/phi-1.5 use "**the
  same tokenizer as codegen-350M-mono**" and contain **no BPE-vocabulary-size ablation** — the widely
  repeated "phi-1 found 512 vocab best for small models" claim is **not in the primary text I read**.
  I could not verify it in 2306.11644v2 or 2309.05463. **Do not cite phi for a vocab-size result.**
- **Stated limitations (authors, verbatim):** "phi-1 is specialized in Python coding, which restricts
  its versatility"; "phi-1 lacks the domain-specific knowledge of larger models"; "phi-1 is less
  robust to stylistic variations or errors in the prompt."
- **Reproducible here?** No — the synthetic recipe depends on GPT-3.5/4 as a generator.
- **Verdict: REJECT the recipe at this scale; keep the *direction*.** "Curate/label data for the
  register you actually want" transfers; "buy a frontier model to write the corpus" does not, and Chakaso
  is a local-first project that must not require a commercial data provider (AGENTS non-negotiable).

---

## Priority 2 — OLMo's transparency model

### 2a. OLMo (original) — what "fully open" means
- **Source:** *OLMo: Accelerating the Science of Language Models*, arXiv 2402.00838v4 (primary, AI2).
- **Mechanism / reported release list (verbatim):** "we release OLMo alongside open training data,
  training and evaluation code, intermediate model checkpoints, and training logs"; "we also release
  **hundreds of intermediate checkpoints** available as revisions on HuggingFace"; the data is
  **Dolma** "including code that produces the training data, and tools for analyzing pretraining
  data," under "a permissive license." Models **1B and 7B**, "all trained on at least 2T tokens."
- **Stated limitations (authors):** quality below closed SOTA at the same size (the paper frames
  OLMo as *competitive*, not leading); per-batch data-ordering traces are not named as a separate
  artifact in the paper text.

### 2b. OLMo 2 — the strengthened claim, and mid-training
- **Source:** *2 OLMo 2 Furious*, arXiv 2501.00656v3 (COLM) (primary). Companion provenance tool:
  *OLMoTrace*, arXiv 2504.07096, 2025-04-09 (primary). Blog: `allenai.org/blog/olmo2` (primary team).
- **Mechanism / reported release list (verbatim, abstract):** OLMo 2 is "a family of dense
  autoregressive language models at **7B, 13B and 32B** scales with fully released artifacts —
  **model weights, full training data, training code and recipes, training logs and thousands of
  intermediate checkpoints**." Sizes/tokens: "7B … trained on **4.05 trillion** tokens (3.90T
  pretraining stage)"; "13B … **5.6 trillion** (5T pretraining)." New mid-training data mix
  **Dolmino Mix 1124**, "Learning-rate annealing" and "**Checkpoint Soups**" are separate named
  contributions (Sections 4.1–4.5). "fully open science requires more than just open weights" (blog).
  Result framing: OLMo 2 base models "sit at the Pareto frontier … matching or outperforming …
  Llama 3.1, Qwen 2.5, and Gemma 2 while using fewer FLOPs."
- **On "traces":** OLMoTrace attributes a model output by "match[ing] against the **entirety** of an
  LM's training data" — i.e. the reproducibility comes from *publishing the whole corpus + code*, not
  from a proprietary per-batch log. I did **not** find the word "traces" (as a per-step dataloader
  file) in 2501.00656v3 or OLMoTrace's read text, and the `allenai/OLMo-core` README as fetched does
  not describe releasing one. **Could not verify a distinct "dataloader trace" artifact from the
  primary papers**; treat the community "traces" framing as *not confirmed against primary text*.
- **Stated limitations (authors):** "**Due to cost concerns we did not explore the full range of
  learning rates** … This is the main limitation of this line of experimentation"; "we have no
  guarantee that tasks we consider unseen … are not part of the development set of other models."
- **What OLMo does that Chakaso does not:** Chakaso already (per ADR-0036) pins checksum-verified
  corpus **snapshots** and refuses drift — that is the *data-identity* half of OLMo's promise, and it
  is genuinely good practice. Chakaso currently does **not** (i) publish full W&B-style per-step
  training logs, (ii) keep **every** intermediate checkpoint (E2608/E2611 keep the selected one), or
  (iii) release the exact dataloader order/seed.
- **Reproducible here?** Chakaso is already single-run reproducible on its own machine; OLMo's bar is
  *third-party re-execution on different hardware*, which needs the logs + checkpoints.
- **Verdict: ADAPT a cheap subset.** At Chakaso's 3.4M / 19.66M-token scale the OLMo artifacts are
  *inexpensive*: (1) commit a per-step metrics trace (loss/lr/grad-norm vs step) alongside
  `results.json` — the records already store aggregates, so this is a small extension that makes a
  reproduction auditable *mid-curve* (directly supports E2608/E2611's "both selected the ceiling");
  (2) retain a few named intermediate checkpoints at pre-declared steps instead of only the winner;
  (3) record the dataloader seed + shuffle order hash. **Do NOT** adopt OLMo's *full-corpus re-download
  release* posture — Chakaso's corpora are pinned snapshots with fingerprints and, for licensed text,
  cannot always be redistributed; snapshot+checksum is the right local equivalent.

---

## Priority 3 — Tokenizer: Qwen's vocab and the vocab-size trade-off

### 3a. Qwen's byte-level BPE, 151,669 vocab
- **Source:** *Qwen2 Technical Report*, arXiv 2407.10671v1 (primary); *Qwen3 Technical Report*,
  arXiv 2505.09388 (primary).
- **Mechanism (verbatim):** Qwen2 — "Following Qwen (Bai et al., 2023), we employ the **identical
  tokenizer based on byte-level byte-pair encoding** … exhibits high encoding efficiency, as
  evidenced by its **better compression rate relative to alternatives, facilitating the multilingual
  capabilities** of Qwen2. Models of all sizes employ a common vocabulary consisting of **151,643
  regular tokens and 3 control tokens**" (config table lists **151,646**; context 32,768 → 131,072).
  Qwen3 — "byte-level BPE (BBPE) … with a **vocabulary size of 151,669**"; smallest dense model
  **0.6B** (tied embeddings, 32K context).
- **Correction to the brief:** the switch was **not** word-piece → byte-level between Qwen1 and Qwen2;
  Qwen2 states the tokenizer is *identical* to Qwen1 (Bai et al. 2023), which is already BBPE. The
  large **~152k** vocab is a *multilingual* decision ("facilitating multilingual capabilities"),
  uniform across 0.5B–72B.
- **Reported evidence (authors):** only a qualitative claim ("better compression rate relative to
  alternatives"); the reports give **no cross-tokenizer characters-per-token table** in the text I
  read. A cross-tokenizer efficiency comparison is therefore a **secondary** claim unless measured.
- **Stated limitations (authors):** not stated for the tokenizer; the same-vocab-for-all-sizes design
  is noted "owing to considerations in distributed training, the effective size for the embeddings is
  larger."
- **Verdict: REJECT the 151k vocab for Chakaso.** The benefit is documented for multilingual/frontier
  models. Gemma 2's Table 2 (embedding params: **2B model → 590,118,912 embedding** of a 2B model)
  makes the cost concrete: a 152k vocab × d_model 256 ≈ **39M embedding parameters**, ~11× Chakaso's
  *entire* 3.41M model. Adopting it is impossible under Chakaso's parameter budget and its tied-
  embedding design.

### 3b. The real vocab-size-vs-model-size evidence (this is the paper Chakaso should use)
- **Source:** *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*, arXiv
  2407.13623v3 (NeurIPS 2024, primary). Complementary: *Large Vocabulary Size Improves Large Language
  Models*, arXiv 2406.16508v1 (ACL 2024, primary).
- **Mechanism / reported evidence (2407.13623, authors):** they train **33M → 3B** parameters while
  sweeping **V from 4K to 96K** and find **optimal vocabulary grows with model size** — "larger
  models requiring larger vocabularies" — as a power law "N_vopt ∝ N^γ with **γ ≈ 0.83**." For a
  **33M-parameter model the predicted optimal vocab is ~37K–43K.** Mechanism for why big vocab hurts
  small models: "Larger vocabularies lead to more clustering of embeddings, especially for infrequent
  words" (sparse signal, diluted softmax). **2406.16508** reports the *opposite direction at larger
  scale* ("larger vocabulary sizes lead to better performance in LLMs") but explicitly flags the
  confound that a smaller-vocab model has fewer embedding+output parameters, and that standard
  vocabularies are "30k–60k for monolingual … ~250k for multilingual."
- **Stated limitations (authors):** 2407.13623's own limitations are in appendices (B.1 method limits,
  B.4 multilingual/multimodal extensions); neither paper tests **below 33M**.
- **Reproducibility:** neither was reproduced by Chakaso. The **direction** (smaller model ⇒ smaller
  optimal vocab) is the transferable part; the **absolute** 37–43K optimum is their measured scale,
  ~10× Chakaso's parameter count.
- **Relevance at Chakaso scale — the honest read:** Chakaso's **4,096 vocab is well below** the
  published optimal (~37K+) even for the 33M model, so on the *quality* axis a larger vocab would
  likely reduce bits/char. But the *budget* is the hard constraint: tied embeddings put
  **4,096 × 256 = 1,048,576 params ≈ 30.7%** of Chakaso's 3,413,760 total into the embedding, and
  raising V toward the published optimum would make the embedding dominate or exceed the whole model
  unless d_model grows with it. **The published scaling law and Chakaso's CPU parameter budget pull
  in opposite directions, and no source resolves the collision at 3.4M.**
- **Verdict: TEST EXPERIMENTALLY.** Chakaso should run its own vocab sweep — e.g. BPE trained at
  {2k, 4k, 8k, 16k, 32k} on the pinned snapshot, **holding d_model and total parameter count fixed**
  (reduce FFN width to pay for a larger table), measuring held-out bits-per-**character** so the
  tokenizer change is not confounded by token-vs-character units. Report characters-per-token for each.
  This is cheap on CPU (the vocab is small) and answers the question the literature genuinely does
  not: what is optimal **below 10M**, on archaic English?

### 3c. Cross-tokenizer efficiency on multilingual / code / math — what is actually primary
- **Sources:** Qwen2/3 claim only "better compression rate" (no table, above). Phi-3 (2404.14219) is
  the closest thing to a *design* signal in the family: **phi-3-mini (3.8B)** uses the **Llama-2
  tokenizer, vocab 32,064**, while **phi-3-small (7B)** switched to **tiktoken, vocab 100,352, "for
  better multilingual tokenization"** — i.e. a *larger* vocab adopted *as the model got bigger* and
  *when multilingual coverage mattered*. SmolLM2 (49,152) and Gemma (256,128) sit in between / far
  out. A rigorous cl100k-vs-llama3-vs-Qwen characters-per-token comparison exists in **secondary**
  sources (e.g. arXiv 2410.12989, a third-party tokenizer-clustering study; and Frontiers 2025
  "Tokenization efficiency of current foundational LLMs") — I read these only at headline level and
  label them **secondary; not adopted as evidence**.
- **Verdict: TEST EXPERIMENTALLY / do not import.** Chakaso trains on monolingual archaic English with
  no code and no multilingual requirement, so the *only* efficiency number that matters is
  characters-per-token on its own corpus, which Chakaso can measure directly. Importing cl100k/llama3
  multilingual comparisons would be cargo-culting an axis Chakaso does not exercise.

---

## Priority 4 — Phi synthetic-data and curriculum, claimed vs replicated

### 4a. The Phi thesis across four papers
- **Sources:** phi-1/1.5 (above); *Phi-3 Technical Report*, arXiv 2404.14219 (primary); *Phi-4
  Technical Report*, arXiv 2412.08905v1, 2024-12 (primary, Microsoft Research).
- **Mechanism (verbatim):** Phi-3 — "a combination of **LLM-based filtering of publicly available web
  data, and LLM-created synthetic data**, enable performance in smaller language models that were
  typically seen only in much larger models"; phi-3-mini (3.8B) trained for **3.3T tokens**, bf16,
  context 4K/128K (LongRoPE). Phi-4 (14B) — "**synthetic data constitutes the bulk of the training
  data** … generated using … **multi-agent prompting, self-revision workflows, and instruction
  reversal**," plus an optimized "training curriculum and data mixture."
- **Reported evidence (authors, and flag the distillation):** Phi-3 claims phi-2 (2.7B) "matched the
  performance of models **25 times larger** trained on regular data" and phi-3-mini "69% on MMLU …
  rivals … Mixtral 8x7B and GPT-3.5." Phi-4 reports "substantially **surpasses its teacher model
  (GPT-4)** on STEM-focused QA," with MMLU **84.8**, GPQA **56.1**, MATH **80.4**. **Crucially, the
  Phi family "largely distill[s] the capabilities of a teacher model (specifically GPT-4)"** (Phi-4's
  own words) — the quality claim is inseparable from distillation from a frontier teacher.
- **Stated limitations (authors):** Phi-3/Phi-4 dedicate a section ("Benchmarking Considerations") to
  **data contamination** — "Many benchmarks rely on datasets that overlap with pretraining corpora"
  — which they claim to mitigate but cannot eliminate; Phi-3.5-Vision notes ungrounded outputs in
  sensitive domains. The "matched models 25× larger" figure is a **claimed author result on academic
  benchmarks the authors themselves flag as contamination-prone**.
- **Independently replicated?** The *existence* of small, strong Phi-class models is replicated (they
  are public and widely re-benchmarked). The *isolation of data-quality as the cause* (vs distillation
  from GPT-4) is **not cleanly separated even by the authors**, and I found no primary ablation here
  proving quality alone (without distillation) at ≤1B produces the headline results. Treat "quality
  dominates quantity" for Phi as **claimed, entangled with teacher distillation.**
- **Relevance at Chakaso scale:** Chakaso has **no frontier teacher** and must not import a commercial
  one (AGENTS: runtime must not require a commercial API). The one transferable, teacher-free idea is
  **register/curriculum structure in the mixture** — ordering and balancing data categories — which is
  exactly the E2612 equal-token prose-vs-technical experiment already preregistered.
- **Verdict: REJECT the synthetic-data recipe wholesale (teacher-dependent); ADAPT the curriculum
  idea (mixture ordering + annealing), which OLMo 2 and Phi-4 both name as independent contributions.**

---

## Priority 5 — Gemma 2 mechanisms below 1B (soft-capping / local-global / PerToken-Fn)

### 5a. What Gemma 2 actually did, and what it *proved*
- **Source:** *Gemma 2: Improving Open Language Models at a Practical Size*, arXiv 2408.00118
  (primary), from PDF text. Sizes **2B / 9B / 27B** trained on **2T / 8T / 13T** tokens; vocab
  **256,128**; Table 1 lists Non-linearity **GeGLU**, GQA, pre+post RMSNorm, sliding window 4096 /
  global span 8192.
- **Logit soft-capping (verbatim formula):** "logits ← soft_cap ∗ tanh(logits / soft_cap) … soft_cap
  **50.0 for the self-attention layers and 30.0 for the final layer**." Framed alongside RMSNorm
  "To **stabilize** training"; the report provides **no quality ablation** attributing benchmark gains
  to soft-capping.
- **Alternating local/global attention (verbatim):** "We alternate between a local sliding window
  attention … and global attention … **in every other layer**." Their inference study (Table 10) shows
  changing the sliding window at inference moves validation perplexity only **1.63 → 1.63 → 1.64**
  (windows 4096/2048/1024) — "moderate impact … a leverage for slight inference speed gain." So the
  *quality* benefit of the pattern is not isolated in the report.
- **The small-model result that *is* isolated — distillation:** "We also train the **2B and 9B models
  with knowledge distillation** instead of next token prediction … release a **27B trained from
  scratch**"; students are trained "on a quantity of tokens that is more than **50×** the
  compute-optimal." **Table 6:** a 2B over 500B tokens scores **60.3 from-scratch → 67.7 distilled**.
  **Table 7 (the below-1B evidence):** perplexity at 200M / 400M / 1B is **23 / 19 / 17 from scratch**
  vs **21 / 17 / 15 distilled** — the gain "**remains as the model size is scaled** [down]."
- **Could not verify:** no per-mechanism ablation table for soft-capping or PerToken-Fn was present in
  the PDF text; "full attention for training speed" and "**PerToken-Fn**" appear in the task brief but
  **not** in Gemma 2's report (Table 1 says plain **GeGLU**). Gemma 1 (2403.08295) as read here
  likewise shows only GeGLU/RoPE/MQA/RMSNorm; I could not locate a PerToken-Fn quality ablation in
  either PDF. **Flag: the "PerToken-Fn matters for small models" premise is unverified against the
  primary reports.**
- **Verdict:** soft-capping **REJECT** (stability device, no quality evidence, superseded — see 5b);
  alternating local/global **REJECT / low priority** at context **256** (a 4096/8192 window is
  meaningless when the whole sequence fits in one window; there is nothing to save); PerToken-Fn
  **REJECT (premise unverified)**; **distillation → TEST EXPERIMENTALLY**, because it is the *only*
  Gemma 2 mechanism with a primary small-model ablation (down to 200M), and it aligns with Qwen3's
  "strong-to-weak distillation" for 0.6B–1.7B. Chakaso lacks a teacher today, but a larger self-trained
  checkpoint could become one — the mechanism, not the frontier-teacher version, is what's portable.

### 5b. Gemma 3 removed one of these mechanisms (the cargo-cult warning)
- **Source:** *Gemma 3 Technical Report*, arXiv 2503.19786v1, 2025-03-08 (primary).
- **Reported (verbatim):** "**we replace the soft-capping of Gemma 2 with QK-norm**"; attention uses
  "a pattern of **5 local layers for every global layer**" with a "**1024**" token local window;
  tokenizer is "**a SentencePiece tokenizer … with byte-level encodings**" (262k in text / 256k in
  Table); released dense models **1B / 4B / 12B / 27B**, and "**All Gemma 3 models are trained with
  knowledge distillation**."
- **Why this matters:** the **same team dropped soft-capping one generation later** and kept the
  small models' gains by **distillation** (now including the 1B) — direct evidence that soft-capping
  was not the source of quality, and confirmation that **distillation is the mechanism that survives
  down toward 1B.** (PerToken-Fn is still not mentioned.)
- **Verdict:** reinforces 5a. For Chakaso the two facts to keep are "**distillation helps the smallest
  models the most**" and "**the stability/activation tricks do not.**"

---

## Cargo-cult flags (mechanisms whose benefit is documented only at billions of parameters)

- **151k / 256k vocabularies** (Qwen, Gemma, Llama-3): a *multilingual + frontier-scale* decision. At
  3.4M params it would consume the entire budget; the only published small-scale evidence (2407.13623)
  actually argues for a **smaller** vocab than 151k, though it still suggests Chakaso's 4,096 is on the
  undersized side. Do not add a big tokenizer because Qwen has one.
- **Logit soft-capping** (Gemma 2 → removed in Gemma 3): a large-model training-stability device with
  no isolated quality ablation and explicitly superseded. Chakaso's CPU runs are small and stable
  enough to not need it.
- **Alternating local/global / sliding-window attention** (Gemma, Mistral, Qwen long-context): its value
  is *long* context (4096/8192/128K). With Chakaso's **context 256** there is no windowing to do.
- **MoE, GRPO/RL post-training, thinking modes** (Qwen3, DeepSeek, OLMo-2 RLVR): cluster-bound and
  post-training; irrelevant while the model is a toy and answers nothing.
- **"Small model = large-model recipe minus the GPUs"** in general: TinyLlama's HumanEval **6.71** at
  1.1B is the cautionary number — a small model faithfully copying a big architecture and *merely*
  overtraining does not become useful. Chakaso is at 3.4M; the gap is one of **data and tokens**, not
  missing architectural tricks.

## Scale-dependent extrapolations (findings whose *test range* does not reach Chakaso)
- SmolLM2 data-classifier result: validated at **135M–1.7B**; below 10M untested.
- 2407.13623 vocab scaling law: measured **33M–3B**; Chakaso is ~10× below its smallest.
- Gemma 2 distillation: ablated **200M–2B**; the "gain persists as size shrinks" claim stops at 200M.
- TinyLlama / Phi data-quality claims: **1.1B–14B**, and Phi's are entangled with GPT-4 distillation.
- OLMo / Gemma / Qwen token budgets (**2–15T**): three to six orders of magnitude above Chakaso's
  19.66M; only the *shape* of the recipe (overtrain, then anneal/curate) transfers.

## Corrections to the brief (so the record does not repeat a false citation)
1. **Qwen did not "switch from word-piece to byte-level BPE" in Qwen2.** Qwen2 states the tokenizer is
   *identical* to Qwen1 (already BBPE); it is Qwen3 that reports vocab **151,669** (Qwen2: 151,643 + 3).
2. **Phi-1/1.5 contain no tokenizer-vocabulary ablation.** phi-1 uses the codegen-mono tokenizer; the
   "512-vocab-best-for-small-models" story is not in the primary text — use **2407.13623** instead.
3. **Gemma 2's released non-linearity is GeGLU, not PerToken-Fn** (Table 1), and Gemma 3 removed the
   soft-capping. The "Gemma 2 PerToken-Fn evidence below 1B" question has no affirmative primary answer.

## What Chakaso has actually validated vs this review
None of the mechanisms above is a Chakaso measurement. This document is **Research**, not evidence of
competence. The only locally-validated facts remain E2608/E2611: a 3.4M rotary decoder at
**1.6865** and a ~1.0M control at **1.8077** bits-per-character against unigram ~3.23 and uniform ~5.57,
both **still improving at the 2,400-step ceiling** and **neither producing usable text**. That the loss
was still falling at the ceiling is the one point this review sharpens into a next step: the binding
constraint is **tokens and budget** (TinyLlama/SmolLM regime), not the missing architectural tricks —
which is consistent with ADR-0041 choosing a **data** experiment next.

---

### Sources (all primary unless marked; every arXiv PDF/HTML opened 2026-11-02)
- SmolLM2 2502.02737v1 (2025-02-04) — https://arxiv.org/html/2502.02737v1 ; SmolLM blog (team, July 2024) — https://huggingface.co/blog/smollm
- TinyLlama 2401.02385 — https://arxiv.org/html/2401.02385
- Phi-1 2306.11644v2 — https://arxiv.org/pdf/2306.11644 ; Phi-1.5 2309.05463 — https://arxiv.org/abs/2309.05463
- Phi-3 2404.14219 — https://arxiv.org/pdf/2404.14219 ; Phi-4 2412.08905v1 (2024-12) — https://arxiv.org/abs/2412.08905
- OLMo 2402.00838v4 — https://arxiv.org/pdf/2402.00838 ; OLMo 2 2501.00656v3 — https://arxiv.org/pdf/2501.00656 ; OLMoTrace 2504.07096 (2025-04-09) — https://arxiv.org/html/2504.07096v1 ; OLMo2 blog (team) — https://allenai.org/blog/olmo2 ; allenai/OLMo-core (official, README read 2026-11-02)
- Qwen2 2407.10671v1 — https://arxiv.org/html/2407.10671v1 ; Qwen3 2505.09388 — https://arxiv.org/pdf/2505.09388
- Gemma 1 2403.08295 — https://arxiv.org/pdf/2403.08295 ; Gemma 2 2408.00118 — https://arxiv.org/pdf/2408.00118 ; Gemma 3 2503.19786v1 (2025-03-08) — https://arxiv.org/html/2503.19786v1
- Vocab scaling: 2407.13623v3 (NeurIPS 2024) — https://arxiv.org/html/2407.13623v3 ; 2406.16508v1 (ACL 2024) — https://arxiv.org/html/2406.16508v1
- **Secondary (labelled, not used as evidence):** tokenizer-clustering study 2410.12989 — https://arxiv.org/html/2410.12989v1 ; Frontiers in AI (2025) "Tokenization efficiency of current foundational LLMs" — https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1538165/full
