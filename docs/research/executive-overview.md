# Executive research overview

**Status:** Research. A reader's map, not a result. Every row below is one conclusion, labelled with **what kind of
evidence stands behind it**, because the repository's rule is that a hunch must not acquire a measurement's authority.
The detail, the caveats and the numbers live in the documents linked from each row; this page deliberately repeats
none of them beyond a headline.
**As of:** 2026-11-01. E2613 (a valid arm) and E2614 are running or preregistered and have no result; nothing here
depends on them.

## What Chakaso is, in three sentences

Chakaso is a language-model research project whose product is a model it trains itself from random initialization, plus
the Python interface that loads and measures it ([ADR-0033](../decisions/ADR-0033-model-centred-product-contract.md)).
The largest model trained so far has 3,413,760 parameters, beats every baseline on held-out text, and produces **no
usable text**. Nothing in the repository is evidence of language-model competence, and the direction below is a plan for
earning that evidence in order, not a claim that it has been earned.

## Evidence classes used

| Label | Meaning |
| --- | --- |
| **Published** | A source outside this repository states it; the source was read (abstract or full text, as each document says) |
| **Measured here** | Computed in this repository from committed code and a pinned input; the record names both |
| **Experimental** | A preregistered run with a result in this repository; one seed unless stated |
| **Computed** | Arithmetic on numbers given, shown in the document |
| **Hypothesis** | Falsifiable, with the run that would settle it; not believed |
| **Decision** | A choice made, with its reasons, that a later measurement may overturn |

## The conclusions that matter, and what each rests on

| # | Conclusion | Class | Where |
| --- | --- | --- | --- |
| 1 | A 3.4M-parameter rotary decoder trained from scratch reaches 1.6865 bits per character held out, below unigram and uniform baselines — **a floor** (training stopped at its step ceiling, validation still falling), **within-work** (chapter-level split, K-023) | Experimental | [E2608](../../experiments/E2608/record.md), [K-023](../agent/KNOWN_ISSUES.md) |
| 2 | At matched capacity and context (~1M parameters) the decoder family beats the transformer by 0.0966 bits per character; about 44 % of E2608's gap survives the match | Experimental, one seed, **band unmeasured** | [E2611](../../experiments/E2611/record.md) |
| 3 | The repetition in the released checkpoint's text is greedy decoding of a human-diverse distribution, not a pipeline defect, and no decoding setting writes coherent text | Measured here | [E2610](../../experiments/E2610/record.md) |
| 4 | Training on five registers instead of prose lowers held-out technical loss by 0.27584 and raises prose loss by 0.03026 — a trade-off at one seed per arm | Experimental | [E2612](../../experiments/E2612/record.md) |
| 5 | The prose cost is prose *dose*, not the presence of technical text | **Hypothesis (H8)**; E2613 preregistered, no valid result | [hypotheses](hypotheses.md), [E2613](../../experiments/E2613/record.md) |
| 6 | Frontier mechanisms (GQA, MLA, MoE, sliding-window attention, multi-token prediction, FP8, logit soft-capping) have no supporting evidence at 3.4M and several published sources place their benefit at 70B or above: **0 adopt, 1 adapt, 2 test, 8 reject** | Published evidence + Decision | [landscape](open-model-landscape.md), [campaign](chakaso-research-campaign-2026-11.md) |
| 7 | Small models are trained on far more tokens than Chinchilla's ~20 per parameter (TinyLlama 1.1B on 3T; SmolLM2 1.7B on ~11T); Chakaso trains at ~5.8 tokens per parameter on 19.66M tokens | Published + Measured here | [landscape](open-model-landscape.md), [compute plan](chakaso-compute-and-scaling-plan.md) |
| 8 | A web-scale corpus licenses its *packaging*, not its text; the only upgrade whose licence reaches the content is public-domain or CC0 text | Published (licences read at source) + Decision | [data review](../internal/engineering/data-and-licensing-research-2026-11-01.md) |
| 9 | The 0.02 bits-per-character band every decoder verdict uses is an inherited convention, and a standard deviation from three seeds is known only to within 0.52×–6.29× at 95 % | Computed; Pythia's ten-seed suite is Published | [Pythia note](../internal/engineering/open-model-research-pythia-2026-11-01.md), [E2614](../../experiments/E2614/record.md) |
| 10 | The committed English vocabulary encodes Cyrillic, Arabic, Devanagari and Han as raw bytes (1.00 bytes per token); a vocabulary serving them at d_model 256 would be larger than the whole current model | Measured here (one sentence per language) + Computed | [multilingual note](../internal/engineering/multilingual-tokenization-2026-11-01.md) |
| 11 | On E2612's split, 0.217 % of held-out 8-word windows also occur in training; no held-out work above 3 % | Measured here | [split-leakage audit](../internal/engineering/split-leakage-audit-2026-11-01.md) |
| 12 | The documentation's arXiv citations resolve (106 of 106) and every named first author is right (47 of 47); 122 of 261 citations could not be checked from their line | Measured here | [citation audit](../internal/engineering/citation-audit-2026-11-01.md) |
| 13 | Base pretraining, mid-training and post-training are separated by the one thing each changes, and the gates into the latter two are closed | Decision | [ADR-0043](../decisions/ADR-0043-training-stages-are-separated-by-what-changes-and-entered-by-gates.md) |
| 14 | **No instrument in the repository is known to detect a real improvement in what the model has learned**: in E2612's own data, held-out loss and repetition moved in opposite directions | Measured here | [campaign §7](chakaso-research-campaign-2026-11.md), [K-020](../agent/KNOWN_ISSUES.md) |

## Where the evidence is thin, in order of how much it matters

1. **Everything is one seed per arm** until E2614 reports. Conclusions 2, 4 and 5 are single draws against a band nobody
   has measured.
2. **Every decoder run is truncated at its step ceiling.** Every absolute number is a floor.
3. **The corpus is archaic public-domain English.** No partition is modern; the corpus's age has never been measured, only
   disclaimed.
4. **There is no coherence or usefulness instrument.** Until one exists, a better loss cannot be called a better model
   (conclusion 14), and post-training stays closed (ADR-0043).
5. **The staged-training recipes read are for models of 1.7B parameters and above, and the multilingual and
   mixture-of-experts sources are for models in the hundreds of millions and up.** None was read below 10M.

## The direction, as a decision and not a promise

Data before architecture ([ADR-0041](../decisions/ADR-0041-next-training-target-general-base-with-technical-mixture.md)):
establish what the measurement can see (seed spread, then an era partition), spend the data lever that has evidence
(register weighting, then a modern licence-clean slice), and only then consider capacity. The ordered plan, its
milestones and what is explicitly out of scope are in [the campaign roadmap](chakaso-research-campaign-2026-11.md) §5–6
and the live ledger in [ACTIVE_TASK](../agent/ACTIVE_TASK.md). A frontier architecture, code data, teacher-derived data in
the released line, and any post-training stage are **not** on it.
