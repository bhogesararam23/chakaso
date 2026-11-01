# Citation audit, 2026-11-01

**Status:** Research record — a measurement of this repository's own documentation, not of any model. Produced by
[`scripts/audit_citations.py`](../../../scripts/audit_citations.py), whose parsing and verdict rules are covered by
nine offline tests (`tests/test_audit_citations_script.py`).

## What was checked, and what that can and cannot show

The research package cites papers by arXiv identifier. A citation makes two claims: that the identifier exists,
and that it is the paper the sentence says it is. The audit extracts every `arXiv:` and `arxiv.org/abs/`
identifier from the **tracked Markdown** (git-ignored scratch files are not the project's), queries arXiv's API
for each distinct identifier, and applies the strongest check the sentence supports:

| Check | Applies when the line... | Passes when |
| --- | --- | --- |
| `author-ok` | names a first author (`Shukor et al.`, or `Lin, S. et al.` in a reference list) | arXiv's first author has that surname |
| `title-ok` | quotes an italic title and names no author | at least 60 % of the title's non-trivial words occur in arXiv's title |
| `date-ok` | states a date and names neither | a stated date equals the first version's month (and day, if given) |
| `date-differs` | states only dates, none equal to the first version's | — reported for reading, never failing: a note may rightly cite a later version or the day it was read |
| `unchecked` | gives none of the above | — |

**This does not verify what any sentence says the paper found.** That is a reading task; the earlier
[verification ledger](research-verification-ledger-2026-11-01.md) did it for 33 claims. The audit's job is to
remove from that task every citation whose identifier, author, title or date is already wrong, and to say honestly
how many it could not look at.

## Result

* **261 citations of 106 distinct identifiers. All 106 resolve.**
* **131 of the 261 citations were corroborated by arXiv's metadata** (47 by first author, 6 by title, 78 by date).
  **122 were unchecked** — a bare identifier with no author, title or date on its line — and they remain a reading
  task, not a verified set.
* **No named first author was wrong.** Of 47 sentences that name one, all 47 match arXiv's first author.
* **Three defects in two entries were found and fixed in this commit**, both in
  [`open-model-landscape.md`](../../research/open-model-landscape.md):
  1. *OLMo 2* (`2501.00656`) was cited under Ai2's announcement title, "OLMo 2: The Best Fully Open Language
     Model to Date". arXiv titles the report **"2 OLMo 2 Furious"**; both titles were read from their live pages
     on 2026-11-01. The same entry dated the report "Jan 2025", the identifier's month; arXiv's first version is
     **2024-12-31**. The entry now gives arXiv's title and date and names the announcement's title.
  2. *Gemma 2* (`2408.00118`) was dated "Aug 2024", the identifier's month. arXiv's first version is
     **2024-07-31**.
* **Six further `date-differs` lines were read and are correct as written**: five cite a later version or a read-on date
  ("v3 read 28 October 2024", "v3 14 April 2025", "v5 10 June 2016", "v1 read 14 May 2026", "v2 read 2 May 2024"),
  and one is the DeepSeek-V3 "v2 18 Feb 2025" against a 2024-12-27 first version. None claims the first-version
  date, so none is an error.

## Limits

* The audit cannot see a citation that is *wrong in substance but well-formed*: right paper, wrong claim.
* A 60 % word-overlap bar tolerates a shortened title and would also tolerate a similar one; it catches the
  wrong paper, not the nearly-right one. Common function words are ignored so that "all you need" cannot make two
  papers look alike.
* Only arXiv identifiers are audited. The project also cites model cards, repositories and blog posts by URL;
  those are not covered here.
* arXiv's API was reachable and answered for every identifier on 2026-11-01. The metadata is cached under
  `.scratch/` (git-ignored); `--refresh` re-queries it.

## Reproduce

```bash
python scripts/audit_citations.py --refresh --markdown citation-audit-tables.md
```

The exit status is non-zero if any identifier does not resolve or any named author or quoted title fails its check.

## Every identifier the documentation cites

| Identifier | Published | First author | Title (from arXiv) |
| --- | --- | --- | --- |
| [1508.07909](https://arxiv.org/abs/1508.07909) | 2015-08-31 | Rico Sennrich | Neural Machine Translation of Rare Words with Subword Units |
| [1606.06031](https://arxiv.org/abs/1606.06031) | 2016-06-20 | Denis Paperno | The LAMBADA dataset: Word prediction requiring a broad discourse context |
| [1706.03762](https://arxiv.org/abs/1706.03762) | 2017-06-12 | Ashish Vaswani | Attention Is All You Need |
| [1711.05101](https://arxiv.org/abs/1711.05101) | 2017-11-14 | Ilya Loshchilov | Decoupled Weight Decay Regularization |
| [1804.10959](https://arxiv.org/abs/1804.10959) | 2018-04-29 | Taku Kudo | Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates |
| [1808.06226](https://arxiv.org/abs/1808.06226) | 2018-08-19 | Taku Kudo | SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing |
| [1812.06162](https://arxiv.org/abs/1812.06162) | 2018-12-14 | Sam McCandlish | An Empirical Model of Large-Batch Training |
| [1904.01557](https://arxiv.org/abs/1904.01557) | 2019-04-02 | David Saxton | Analysing Mathematical Reasoning Abilities of Neural Models |
| [1909.10351](https://arxiv.org/abs/1909.10351) | 2019-09-23 | Xiaoqi Jiao | TinyBERT: Distilling BERT for Natural Language Understanding |
| [1910.01108](https://arxiv.org/abs/1910.01108) | 2019-10-02 | Victor Sanh | DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter |
| [1910.07467](https://arxiv.org/abs/1910.07467) | 2019-10-16 | Biao Zhang | Root Mean Square Layer Normalization |
| [1910.10683](https://arxiv.org/abs/1910.10683) | 2019-10-23 | Colin Raffel | Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer |
| [1911.02150](https://arxiv.org/abs/1911.02150) | 2019-11-06 | Noam Shazeer | Fast Transformer Decoding: One Write-Head is All You Need |
| [2001.08361](https://arxiv.org/abs/2001.08361) | 2020-01-23 | Jared Kaplan | Scaling Laws for Neural Language Models |
| [2002.05202](https://arxiv.org/abs/2002.05202) | 2020-02-12 | Noam Shazeer | GLU Variants Improve Transformer |
| [2002.10957](https://arxiv.org/abs/2002.10957) | 2020-02-25 | Wenhui Wang | MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers |
| [2004.02334](https://arxiv.org/abs/2004.02334) | 2020-04-05 | Thamme Gowda | Finding the Optimal Vocabulary Size for Neural Machine Translation |
| [2005.11401](https://arxiv.org/abs/2005.11401) | 2020-05-22 | Patrick Lewis | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| [2009.03300](https://arxiv.org/abs/2009.03300) | 2020-09-07 | Dan Hendrycks | Measuring Massive Multitask Language Understanding |
| [2103.03874](https://arxiv.org/abs/2103.03874) | 2021-03-05 | Dan Hendrycks | Measuring Mathematical Problem Solving With the MATH Dataset |
| [2104.09864](https://arxiv.org/abs/2104.09864) | 2021-04-20 | Jianlin Su | RoFormer: Enhanced Transformer with Rotary Position Embedding |
| [2105.13626](https://arxiv.org/abs/2105.13626) | 2021-05-28 | Linting Xue | ByT5: Towards a token-free future with pre-trained byte-to-byte models |
| [2107.03374](https://arxiv.org/abs/2107.03374) | 2021-07-07 | Mark Chen | Evaluating Large Language Models Trained on Code |
| [2107.06499](https://arxiv.org/abs/2107.06499) | 2021-07-14 | Katherine Lee | Deduplicating Training Data Makes Language Models Better |
| [2108.02170](https://arxiv.org/abs/2108.02170) | 2021-08-04 | Daniel Campos | Curriculum learning for language modeling |
| [2109.07958](https://arxiv.org/abs/2109.07958) | 2021-09-08 | Stephanie Lin | TruthfulQA: Measuring How Models Mimic Human Falsehoods |
| [2110.14168](https://arxiv.org/abs/2110.14168) | 2021-10-27 | Karl Cobbe | Training Verifiers to Solve Math Word Problems |
| [2203.02155](https://arxiv.org/abs/2203.02155) | 2022-03-04 | Long Ouyang | Training language models to follow instructions with human feedback |
| [2203.03466](https://arxiv.org/abs/2203.03466) | 2022-03-07 | Greg Yang | Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer |
| [2203.15556](https://arxiv.org/abs/2203.15556) | 2022-03-29 | Jordan Hoffmann | Training Compute-Optimal Large Language Models |
| [2205.14135](https://arxiv.org/abs/2205.14135) | 2022-05-27 | Tri Dao | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness |
| [2206.07682](https://arxiv.org/abs/2206.07682) | 2022-06-15 | Jason Wei | Emergent Abilities of Large Language Models |
| [2206.14486](https://arxiv.org/abs/2206.14486) | 2022-06-29 | Ben Sorscher | Beyond neural scaling laws: beating power law scaling via data pruning |
| [2211.15533](https://arxiv.org/abs/2211.15533) | 2022-11-20 | Denis Kocetkov | The Stack: 3 TB of permissively licensed source code |
| [2302.02971](https://arxiv.org/abs/2302.02971) | 2023-02-06 | Bryn Elesedy | U-Clip: On-Average Unbiased Stochastic Gradient Clipping |
| [2302.13971](https://arxiv.org/abs/2302.13971) | 2023-02-27 | Hugo Touvron | LLaMA: Open and Efficient Foundation Language Models |
| [2305.07759](https://arxiv.org/abs/2305.07759) | 2023-05-12 | Ronen Eldan | TinyStories: How Small Can Language Models Be and Still Speak Coherent English? |
| [2305.10429](https://arxiv.org/abs/2305.10429) | 2023-05-17 | Sang Michael Xie | DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining |
| [2305.13230](https://arxiv.org/abs/2305.13230) | 2023-05-22 | Fuzhao Xue | To Repeat or Not To Repeat: Insights from Scaling LLM under Token-Crisis |
| [2305.13245](https://arxiv.org/abs/2305.13245) | 2023-05-22 | Joshua Ainslie | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| [2305.16264](https://arxiv.org/abs/2305.16264) | 2023-05-25 | Niklas Muennighoff | Scaling Data-Constrained Language Models |
| [2305.18290](https://arxiv.org/abs/2305.18290) | 2023-05-29 | Rafael Rafailov | Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| [2305.20050](https://arxiv.org/abs/2305.20050) | 2023-05-31 | Hunter Lightman | Let's Verify Step by Step |
| [2306.01116](https://arxiv.org/abs/2306.01116) | 2023-06-01 | Guilherme Penedo | The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only |
| [2306.11644](https://arxiv.org/abs/2306.11644) | 2023-06-20 | Suriya Gunasekar | Textbooks Are All You Need |
| [2306.15595](https://arxiv.org/abs/2306.15595) | 2023-06-27 | Shouyuan Chen | Extending Context Window of Large Language Models via Positional Interpolation |
| [2307.09288](https://arxiv.org/abs/2307.09288) | 2023-07-18 | Hugo Touvron | Llama 2: Open Foundation and Fine-Tuned Chat Models |
| [2309.00071](https://arxiv.org/abs/2309.00071) | 2023-08-31 | Bowen Peng | YaRN: Efficient Context Window Extension of Large Language Models |
| [2309.05463](https://arxiv.org/abs/2309.05463) | 2023-09-11 | Yuanzhi Li | Textbooks Are All You Need II: phi-1.5 technical report |
| [2310.06825](https://arxiv.org/abs/2310.06825) | 2023-10-10 | Albert Q. Jiang | Mistral 7B |
| [2310.15961](https://arxiv.org/abs/2310.15961) | 2023-10-24 | Szymon Antoniak | Mixture of Tokens: Continuous MoE through Cross-Example Aggregation |
| [2310.16787](https://arxiv.org/abs/2310.16787) | 2023-10-25 | Shayne Longpre | The Data Provenance Initiative: A Large Scale Audit of Dataset Licensing & Attribution in AI |
| [2311.05232](https://arxiv.org/abs/2311.05232) | 2023-11-09 | Lei Huang | A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions |
| [2311.07911](https://arxiv.org/abs/2311.07911) | 2023-11-14 | Jeffrey Zhou | Instruction-Following Evaluation for Large Language Models |
| [2311.12022](https://arxiv.org/abs/2311.12022) | 2023-11-20 | David Rein | GPQA: A Graduate-Level Google-Proof Q&A Benchmark |
| [2401.00448](https://arxiv.org/abs/2401.00448) | 2023-12-31 | Nikhil Sardana | Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws |
| [2401.02385](https://arxiv.org/abs/2401.02385) | 2024-01-04 | Peiyuan Zhang | TinyLlama: An Open-Source Small Language Model |
| [2401.04088](https://arxiv.org/abs/2401.04088) | 2024-01-08 | Albert Q. Jiang | Mixtral of Experts |
| [2402.00159](https://arxiv.org/abs/2402.00159) | 2024-01-31 | Luca Soldaini | Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research |
| [2402.00838](https://arxiv.org/abs/2402.00838) | 2024-02-01 | Dirk Groeneveld | OLMo: Accelerating the Science of Language Models |
| [2402.03300](https://arxiv.org/abs/2402.03300) | 2024-02-05 | Zhihong Shao | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| [2403.08295](https://arxiv.org/abs/2403.08295) | 2024-03-13 | Gemma Team | Gemma: Open Models Based on Gemini Research and Technology |
| [2403.16952](https://arxiv.org/abs/2403.16952) | 2024-03-25 | Jiasheng Ye | Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance |
| [2404.06395](https://arxiv.org/abs/2404.06395) | 2024-04-09 | Shengding Hu | MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies |
| [2404.07944](https://arxiv.org/abs/2404.07944) | 2024-04-11 | Antonio Del Donno | On the Durdevic approach to quantum principal bundles |
| [2404.10102](https://arxiv.org/abs/2404.10102) | 2024-04-15 | Tamay Besiroglu | Chinchilla Scaling: A replication attempt |
| [2404.10830](https://arxiv.org/abs/2404.10830) | 2024-04-16 | Hantian Ding | Fewer Truncations Improve Language Modeling |
| [2404.14219](https://arxiv.org/abs/2404.14219) | 2024-04-22 | Marah Abdin | Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone |
| [2404.19737](https://arxiv.org/abs/2404.19737) | 2024-04-30 | Fabian Gloeckle | Better & Faster Large Language Models via Multi-token Prediction |
| [2405.04434](https://arxiv.org/abs/2405.04434) | 2024-05-07 | DeepSeek-AI | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model |
| [2405.14578](https://arxiv.org/abs/2405.14578) | 2024-05-23 | Shuaipeng Li | Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling |
| [2406.16508](https://arxiv.org/abs/2406.16508) | 2024-06-24 | Sho Takase | Large Vocabulary Size Improves Large Language Models |
| [2406.17557](https://arxiv.org/abs/2406.17557) | 2024-06-25 | Guilherme Penedo | The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale |
| [2406.18326](https://arxiv.org/abs/2406.18326) | 2024-06-26 | Huixuan Zhang | PaCoST: Paired Confidence Significance Testing for Benchmark Contamination Detection in Large Language Models |
| [2406.19146](https://arxiv.org/abs/2406.19146) | 2024-06-27 | Tomer Porian | Resolving Discrepancies in Compute-Optimal Scaling of Language Models |
| [2407.01492](https://arxiv.org/abs/2407.01492) | 2024-07-01 | Qian Liu | RegMix: Data Mixture as Regression for Language Model Pre-training |
| [2407.13623](https://arxiv.org/abs/2407.13623) | 2024-07-18 | Chaofan Tao | Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies |
| [2407.21783](https://arxiv.org/abs/2407.21783) | 2024-07-31 | Aaron Grattafiori | The Llama 3 Herd of Models |
| [2408.00118](https://arxiv.org/abs/2408.00118) | 2024-07-31 | Gemma Team | Gemma 2: Improving Open Language Models at a Practical Size |
| [2408.03617](https://arxiv.org/abs/2408.03617) | 2024-08-07 | Steven Y. Feng | Is Child-Directed Speech Effective Training Data for Language Models? |
| [2408.13359](https://arxiv.org/abs/2408.13359) | 2024-08-23 | Yikang Shen | Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler |
| [2410.05192](https://arxiv.org/abs/2410.05192) | 2024-10-07 | Kaiyue Wen | Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective |
| [2410.05229](https://arxiv.org/abs/2410.05229) | 2024-10-07 | Iman Mirzadeh | GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models |
| [2412.04403](https://arxiv.org/abs/2412.04403) | 2024-12-05 | Akshita Bhagia | Establishing Task Scaling Laws via Compute-Efficient Model Ladders |
| [2412.08905](https://arxiv.org/abs/2412.08905) | 2024-12-12 | Marah Abdin | Phi-4 Technical Report |
| [2412.15115](https://arxiv.org/abs/2412.15115) | 2024-12-19 | Qwen | Qwen2.5 Technical Report |
| [2412.19437](https://arxiv.org/abs/2412.19437) | 2024-12-27 | DeepSeek-AI | DeepSeek-V3 Technical Report |
| [2501.00656](https://arxiv.org/abs/2501.00656) | 2024-12-31 | Team OLMo | 2 OLMo 2 Furious |
| [2501.12948](https://arxiv.org/abs/2501.12948) | 2025-01-22 | DeepSeek-AI | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning |
| [2502.01481](https://arxiv.org/abs/2502.01481) | 2025-02-03 | Jingzhe Shi | Intrinsic Entropy of Context Length Scaling in LLMs |
| [2502.02737](https://arxiv.org/abs/2502.02737) | 2025-02-04 | Loubna Ben Allal | SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model |
| [2502.17259](https://arxiv.org/abs/2502.17259) | 2025-02-24 | Tom Sander | Detecting Benchmark Contamination Through Watermarking |
| [2503.19786](https://arxiv.org/abs/2503.19786) | 2025-03-25 | Gemma Team | Gemma 3 Technical Report |
| [2505.09388](https://arxiv.org/abs/2505.09388) | 2025-05-14 | An Yang | Qwen3 Technical Report |
| [2506.01913](https://arxiv.org/abs/2506.01913) | 2025-06-02 | Thomas Pethick | Generalized Gradient Norm Clipping & Non-Euclidean $(L_0,L_1)$-Smoothness |
| [2506.15025](https://arxiv.org/abs/2506.15025) | 2025-06-17 | Soufiane Hayou | Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size |
| [2507.09404](https://arxiv.org/abs/2507.09404) | 2025-07-12 | Mustafa Shukor | Scaling Laws for Optimal Data Mixtures |
| [2509.00072](https://arxiv.org/abs/2509.00072) | 2025-08-26 | Terry Jingchen Zhang | Test of Time: Rethinking Temporal Signal of Benchmark Contamination |
| [2509.23963](https://arxiv.org/abs/2509.23963) | 2025-09-28 | Rylan Schaeffer | Evaluating the Robustness of Chinchilla Compute-Optimal Scaling |
| [2510.01135](https://arxiv.org/abs/2510.01135) | 2025-10-01 | Zhaolin Gao | Prompt Curriculum Learning for Efficient LLM Post-Training |
| [2510.02386](https://arxiv.org/abs/2510.02386) | 2025-09-30 | Han Wang | On The Fragility of Benchmark Contamination Detection in Reasoning Models |
| [2511.08199](https://arxiv.org/abs/2511.08199) | 2025-11-11 | Arzu Burcu Güven | Do Syntactic Categories Help in Developmentally Motivated Curriculum Learning for Language Models? |
| [2512.13705](https://arxiv.org/abs/2512.13705) | 2025-12-05 | Siqi Wang | Scaling and Transferability of Annealing Strategies in Large Language Model Training |
| [2512.13961](https://arxiv.org/abs/2512.13961) | 2025-12-15 | Team Olmo | Olmo 3 |
| [2604.27263](https://arxiv.org/abs/2604.27263) | 2026-04-29 | Théo Gigant | Decoupling the Benefits of Subword Tokenization for Language Model Training via Byte-level Simulation |
| [2609.01431](https://arxiv.org/abs/2609.01431) | 2026-09-01 | Zhiliang Chen | Efficiently Estimating Optimal Hyperparameter Scaling Laws through Power-Law Entropy Search |

### Citations that failed a check

*(As produced before the two corrections above; a re-run after them reports none.)*

- `docs/research/open-model-landscape.md:284` arXiv:2501.00656 — **title-mismatch**: cited as 'OLMo 2: The Best Fully Open Language Model to Date' but arXiv titles it '2 OLMo 2 Furious' (overlap 0.14)

### Citations to read before trusting (a stated date is not the first version's)

- `docs/internal/engineering/foundations-research-2026-11-01.md:157` arXiv:2406.19146 — **date-differs**: arXiv's first version is 2024-06-27; the line states 28/10/2024
- `docs/internal/engineering/foundations-research-2026-11-01.md:169` arXiv:2401.00448 — **date-differs**: arXiv's first version is 2023-12-31; the line states 14/4/2025
- `docs/internal/engineering/foundations-research-2026-11-01.md:356` arXiv:1508.07909 — **date-differs**: arXiv's first version is 2015-08-31; the line states 10/6/2016
- `docs/internal/engineering/foundations-research-2026-11-01.md:412` arXiv:2604.27263 — **date-differs**: arXiv's first version is 2026-04-29; the line states 14/5/2026
- `docs/internal/engineering/foundations-research-2026-11-01.md:763` arXiv:2404.10830 — **date-differs**: arXiv's first version is 2024-04-16; the line states 2/5/2024
- `docs/internal/engineering/open-model-research-llama-mistral-deepseek-2026-11-01.md:454` arXiv:2412.19437 — **date-differs**: arXiv's first version is 2024-12-27; the line states 18/2/2025
- `docs/research/open-model-landscape.md:353` arXiv:2408.00118 — **date-differs**: arXiv's first version is 2024-07-31; the line states ?/8/2024
