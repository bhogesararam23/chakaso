# Research verification ledger, 2026-11-01

**Status:** Research. This file records what was **independently re-checked** during the 2026-11-01 research
campaign, and what each re-check found. It exists because the campaign produced a large volume of claims from
both artifacts and external sources, and a claim that was only read once from a summary is not evidence.

Every row states: the claim, how it was checked, what the check produced, and the outcome. An outcome of
**corrected** means the claim as first written was wrong and has been fixed where it was recorded. An outcome of
**not verified** means nobody should rely on it.

Reading order for a newcomer: this ledger is the trust layer. [The measurement harness
audit](measurement-harness-audit-2026-11-01.md), [the model path audit](model-path-audit-2026-11-01.md) and the
four research reviews are the findings; this file says which findings were reproduced rather than accepted.

## Method used for each class of claim

| Claim class | Verification method |
| --- | --- |
| "an experiment ran and produced X" | read the JSON artifact on disk field by field; recompute the derived quantity from its inputs rather than quoting the summary |
| "the harness does/does not leak" | read the function that fits or samples, at the line, and check which side it is handed |
| "a source reports N" | fetch the primary source; where the number is not in the abstract, fetch the body and quote the sentence |
| "a dataset is sized/licensed X" | range-read the object over anonymous HTTPS and count rows and character lengths from the bytes |

## Artifacts: E2612

| # | Claim | Check performed | Result | Outcome |
| --- | --- | --- | --- | --- |
| V-01 | Both E2612 arms actually ran and produced metrics | listed `data/experiments/E2612/{prose,mixture}/` and read both `report.json` in full | 12 files: two reports, two checkpoints (17,905,511 B each), two resumable states, two manifests, one shared tokenizer cache, two logs. Both reports carry `transformer.curve` with 7 records, `test`, `validation` and five `test_partitions` | **verified** |
| V-02 | The arms are comparable because everything but the data was held fixed | compared the fields directly, not the record's prose | corpus `312571f9fa8351a8`, split `adf650435c3a9068`, tokenizer `434debe82f949347`, test positions 254,976, test characters 535,282, parameters 3,413,760, seed 7, `max_steps` 2,400, and all five partition token counts — **equal in both reports**; `training_group_prefixes` `["prose-"]` vs `null` is the only difference | **verified**, and written into the record as a ten-row matched table |
| V-03 | H1 (mixture helps technical text) | recomputed the declared aggregate: Σ(bits/char × scored characters) over the four non-prose partitions | P 2.39222, T 2.11638, Δ **−0.27584**, 13.8× the declared 0.02 band | **verified** |
| V-04 | H2 (mixture does not damage prose) | read the prose partition from both reports | P 1.80926, T 1.83952, Δ **+0.03026** — outside the 0.02 tolerance by 1.5× | **verified**; the preregistered decision rule therefore resolves to outcome 2, a trade-off |
| V-05 | The aggregate number depends on which field you weight by — found by the harness audit | recomputed all three weightings from the same artifacts | scored characters −0.27584 (13.8× band); tokens scored −0.27884 (13.9×); **population** characters −0.18650 (9.3×). The two fields differ 23× for prose (12,414,225 population vs 539,534 scored) | **verified**, and the record now shows all three, because the verdict survives them but the effect size does not |
| V-06 | The arms' prose exposure is unequal, as the record declared in advance | walked the pinned corpus manifest and grouped its 9,388 documents by work and then by side | training-side pool: prose 52,513,592 characters (50.51 %), science 23,690,657 (22.79 %), technology 10,433,868 (10.04 %), mathematics 9,805,768 (9.43 %), medicine 7,520,945 (7.23 %) over 103,964,830 characters | **measured**. Applied to each arm's cap, arm T saw ≈10.1 M prose identifiers against arm P's 20.0 M. Recorded as a proportional estimate from character shares, **not** as a counted token total |
| V-07 | A partition's `characters` is a population, not a denominator | compared `test_partitions[k].characters` with `test_partitions[k].metrics.characters_scored` | differ for every register (prose 12,414,225 vs 539,534; medicine 101,923 vs 101,008) | **verified** — a real trap in the report shape; filed as an issue |
| V-08 | `samples_refused: 1` means a model refused | enumerated every sample's `label` and `status` in both reports | all five continuations have status `generated`; the single refusal is the designed `empty-prompt` edge probe recorded under `generation.edge_samples` | **verified, and the first reading was wrong** — the run's own summary conflates the two counts. The record now says so |
| V-09 | Both arms trained to convergence | read `budget.completion`, `stopped_early` and the final two curve points | both `budget_ceiling` at step 2,400, `stopped_early: false`, validation still falling (P 3.1508→3.1406, T 2.9235→2.9032) | **verified**: every E2612 number is a floor, not a converged value |
| V-10 | The gap is an early-training artefact | differenced the two validation curves step by step | −0.1332, −0.1454, −0.1866, −0.1996, −0.2273, −0.2374 across steps 400→2,400 | **verified**: monotone widening, not yet plateaued when the budget ended |
| V-11 | The generation block describes these models' own domain | read `plan.sample_prompts` and each sample's `prompt_source` | `"repository"` / `chakaso-repo-text-v1` — the probes quote **this repository's section headings** to models trained on Gutenberg text | **verified defect.** The alternative already exists (`held_out`, `subword_experiment.py:979`) and its own docstring at `:225` warns about exactly this. Filed as an issue |
| V-12 | Reported digests let a stranger confirm the artifact | compared the log line "report digest …" with the SHA-256 of the report file | identical (`3764ba3f…`, `295fbc7a…`) — the report digest **is** the file hash, and the body embeds `wall_clock_s`, `positions_per_second`, `peak_resident_bytes`, `torch_threads` | **verified limitation**: a faithful re-run reproduces every metric and cannot reproduce the digest. Reproduction means the numbers agree, not the hashes |

## Harness: leakage and provenance

| # | Claim | Check performed | Result | Outcome |
| --- | --- | --- | --- | --- |
| V-13 | The unigram baseline is fitted on the arm's own training targets, not on test | read `baseline_metrics` at `src/chakaso/training/subword_experiment.py:952-973` | `UnigramBaseline.fit(train_targets, …)`, then scored against `test_targets`. Uniform is `log(vocabulary.size)`. No test text reaches either fit | **verified clean** |
| V-14 | The tokenizer sample cannot touch the held-out side | read `_tokenizer_sample` at `subword_experiment.py:636-654` | draws only from `split.train`, shuffled by `effective_split_seed`, stopping at the character cap; its own docstring explains that catalogue order would otherwise fit one register | **verified clean** |
| V-15 | Under whole-work splits the run reports its deduplication | read `_split_whole_works` (`data.py:274-336`) and its return | it constructs `UnitSplit` **without** `dropped_duplicates`, which therefore defaults to `()`, while the document-level path passes it (`data.py:249`). Dedup happens earlier and correctly at `data.py:186-207` | **verified defect**: `documents_per_side.dropped_duplicates: 0` in both E2612 reports while three documents were excluded. Measurement unaffected; provenance impaired. Filed as an issue |
| V-16 | Which three documents were excluded | matched the `duplicate_groups` pairs against `identifiers_per_side.train` (identifiers carry a `@digest` suffix, so a bare-path test gives a false negative — the first attempt did) | the `mathematics-pg633` members are present; all three `mathematics-pg634` members are absent; 6,515 manifest documents on the training side against 6,512 in the split | **verified**: one member per pair dropped, both members of every pair were on the same side, so no leakage existed to remove |
| V-17 | Whole-work splitting prevented within-work leakage | read `corpus.works_on_multiple_sides` and `groups_per_side` in both reports | 312 train / 56 validation / 62 test works, `works_on_multiple_sides: 0` | **verified** — K-023's leak is absent by construction in E2612, though the field is *reported*, not *enforced* |

## Tokenizer correctness — the campaign's principal finding

This is the one finding that changed the repository rather than only the documentation, so its whole evidence
chain is recorded here. Each row was run on this machine on 2026-11-01, not quoted from the audit.

| # | Claim | Check performed | Result | Outcome |
| --- | --- | --- | --- | --- |
| V-24 | The indexed merge loop still picks the most frequent pair | compared `train_subword_vocabulary` against a brute-force reference that rescans every shape on every merge, on a 43-piece fixture | **disagrees at merge rank 8 and on 25 of 43 merges.** The loop no longer implements the rule in its own docstring | **verified defect.** Mechanism read at `subword.py:473-483`: a stale entry is popped and discarded, and a pair whose count *fell* has no other live entry, so it leaves the candidate set permanently |
| V-25 | The performance commit's own equivalence claim | extracted `6f3274f~1` and compared old, new and reference on identical input | target 300 → identical (`8994ed1d6ecb3e97` both). Target 500 → **34 of 111 merges differ, first at rank 76, and the new loop yields one merge fewer (111 vs 112) with a different fingerprint** (`7aa3693f6652633e` vs `56e90bd13d682d05`) | **verified false.** "The merge sequence on a reference text is byte-identical to the previous implementation, so no vocabulary moves" holds only below the depth at which a count ever falls — the equivalence check ran in the regime where the bug cannot appear |
| V-26 | The measured E2608 tokenizer can be re-fitted from current code | read the pinned snapshot (whose own digest chain verified), selected its 643 training documents, fitted 4,096 identifiers at `min_frequency` 3 in 12.3 s | fingerprint **`a1ef04f64739cd4b`** against the record's **`b597eb34f13d9058`** — **no match**. Divergence begins at merge rank 5; 3,692 of 3,839 merges differ. 20 training documents encode as 152,939 identifiers against the cached vocabulary's 152,053, a **+0.583 %** token cost | **verified breach of ADR-0036**: E2608, E2611 and the released `chakaso-base@0.1.0` were not re-trainable from HEAD. Their tokenizers survive only in git-ignored cache files |
| V-27 | The fix restores reproduction | re-armed the heap (a popped stale entry is re-pushed at its current count instead of discarded) and re-ran V-26 verbatim | fingerprint **`b597eb34f13d9058` — matches the measured record**; 20-document token count returns to **152,053**, identical to the cache; 3,839 merges | **verified fix**, committed as `cfdcd1a` with two regression tests that fail against the pre-fix loop |
| V-28 | The fix is affordable | timed the exact fit on the multi-register corpus, where nearly every pre-tokenized piece is unique, and on the E2612 sample itself | 1,002,483 characters → **6.6 s**; 4,014,828 → **494 s**; the 12,663,402-character E2612 sample → **2,116.4 s (35.3 min)** for 3,839 merges. Against the 29.2 s the optimising commit measured at 4.0M, and against E2612's own 145.55 s of *total* preparation | **verified cost, and it is severe**: exactness cannot be free because the heap may no longer discard work. Preparation for a 4,096-identifier vocabulary on a 12.7M-character sample is now ~35 minutes instead of ~2, so a tokenizer sample is a budgeted parameter of an experiment and not a default |
| V-29 | E2612's tokenizer can no longer be re-fitted by current code | refit the 12,663,402-character sample at 4,096 identifiers and `min_frequency` 3 under the fixed loop | fingerprint **`3abe4df4b123d4c5`** against the measured **`434debe82f949347`** — **no match** | **verified, and it was predicted as a derivation in the first draft of this row.** E2605–E2611's vocabularies (fitted before `6f3274f`) are reproducible again under the fix; E2612's is not, because it was fitted *by the defect*. Its report remains internally valid — both arms scored against the one shared vocabulary, so H1 and H2 stand — but the record is now reproducible only through its git-ignored cache, and a re-run under fixed code is a new tokenizer and therefore a new record. This is the residue filed as K-024(b) |

## External sources

| # | Claim | Check performed | Result | Outcome |
| --- | --- | --- | --- | --- |
| V-18 | *Scaling Laws with Vocabulary* reports N_vopt ∝ N^γ with γ ≈ 0.83, and ~37 K–43 K for a 33 M model | abstract fetch first, then the paper body | the abstract states only that "larger models require larger vocabularies" for 33 M–3 B and cites a 32 K→43 K move for Llama2-70B; the body gives the sentence "Nvopt ∝ Nnv^γ where γ ≈ 0.83 < 1". The 37–43 K figure is the authors' **projection** from a low-cost empirical optimum, not a measured 33 M result | **verified with a caveat now attached**: the exponent is the paper's; the absolute optimum is extrapolated, and neither paper tests below 33 M |
| V-19 | Gemma 3 replaced Gemma 2's soft-capping with QK-norm | abstract fetch (silent) then the report body | body states "we replace the soft-capping of Gemma 2 with QK-norm", "a pattern of 5 local layers for every global layer", local window "span of only 1024 tokens", "a SentencePiece tokenizer… vocabulary has 262k entries", "All Gemma 3 models are trained with knowledge distillation" | **verified** — this is the strongest published evidence in the campaign that a stability trick does not survive one generation while distillation does |
| V-20 | `HPLT/HPLT2.0_cleaned` `eng_Latn_1` holds ≈0.9 billion characters | range-read the parquet object and measured 1,000 documents | 104,293 rows confirmed; **mean 4,122.9 characters** over the first row group → **≈430 million characters**, not 0.9 billion. The earlier figure projected a single 8,853-character document | **corrected** (÷2.1). The recommendation survives — one shard is still ~2.9× the target slice — but the number published was wrong |
| V-21 | What the HPLT columns mean (the data review could not name four of them) | read the schema and sampled values from row group 0 | `f` WARC path, `o` offset, `s`/`rs` sizes, **`u` full URL**, `c` MIME type, `ts` crawl timestamp, `collection`, `lang`/`prob` per-language lists, `robotstxt`, `filter`, `pii`, **`doc_scores` = nine 0–10 per-document scores** | **recovered** — this matters: a `url` and a `doc_scores` column make group-by-domain splitting and declarative quality selection possible rather than invented |
| V-22 | `eng_Latn_1` is representative of "modern technical English" | read the first documents and their content types | `collection: archivebot`, 2020-08-12 crawl, `c` values including `application/rss+xml`, hosts like `steinarlene.wordpress.com`, documents opening on a bare record number | **not supported**: this shard's register is blog and RSS-feed English. The licence and cost case stands; the register case must be measured on a slice before any arm trains against it |
| V-23 | The Gutenberg selection sampled a pinned population | read `data/corpora/gutenberg-multiregister-v1/selection.json` keys against the script's own docstring ("The catalogue snapshot's digest is recorded beside the selection", `scripts/acquire_gutenberg_corpus.py:8`) | the report records `selection_seed`, quotas, bounds and per-work `url`+`sha256`, but **no digest, URL or retrieval date for the input `pg_catalog.csv`**, and E2612's reproduction block names only a local file | **verified gap, and the docstring is false**: the corpus bytes are pinned, the population the rule sampled is not. Filed as an issue |

| V-30 | Meta's own multi-token-prediction result reverses sign below its scale | abstract first (which carries only "increasingly useful for larger model sizes"), then the paper body | body states verbatim: **"Multi-token prediction models are worse than the baseline for small model sizes, but outperform the baseline at scale."** Smallest arm evaluated 300 M | **verified** — this is the single strongest published argument against importing MTP at 3.4 M, and it is the authors' own limitation, not an inference |
| V-31 | Llama 2 used grouped-query attention | fetched the report's Table 1 from the body | `7B 4k ✗ 2.0T`, `13B 4k ✗ 2.0T`, `70B 4k ✓ 2.0T` | **verified** — GQA enters the family at 70 B. "Llama does it, so we should" is factually wrong at 7 B, let alone at 3.4 M |
| V-32 | `arXiv:2404.07944` supports "Llama 3 uses warmup-stable-decay" | resolved the identifier | it resolves to an unrelated quantum-bundles paper | **could not verify** — the WSD case must rest on `2410.05192` alone, whose own evidence is 0.1 B–1.2 B, i.e. nearer Chakaso's scale than any frontier report |
| V-33 | Five arXiv identifiers circulating in this project's own notes were wrong | re-resolved each | BPE is **1508.07909** (07921 is a graph-drawing paper); "Beyond neural scaling laws" is **Sorscher et al. 2206.14486** (vision pruning), not Besiroglu; TinyBERT is **1909.10351**; 2310.12485 is not Mixture-of-Tokens; 2209.01649 is not μP | **corrected** in the foundations review. A mis-numbered citation is the failure mode this project's rules exist to prevent, and it occurred inside our own drafts |

## Open, and load-bearing on purpose

| # | Question | Why it is not settled |
| --- | --- | --- |
| O-01 | Whether E2612's H2 failure is exposure or mixture | the arms differ in prose tokens by ~2× as well as in registers; no arm exists that sees ~10 M prose identifiers and nothing else. Only a third arm answers it |
| O-02 | Whether the 0.02-bit materiality band is the right band at all | it is inherited from E2611's declared threshold, not calibrated against a seed spread. E2612's H2 misses by 1.5× a band nobody has measured the noise for |
| O-03 | Whether any of this transfers off archaic English | no partition in any run so far is modern. E2612's own limits say so, and V-22 says the cheapest modern candidate is not obviously the right register either |
| O-04 | The model-path audit's findings | recorded in [model-path-audit-2026-11-01.md](model-path-audit-2026-11-01.md); each is pending its own re-check before it enters a decision |

## What this ledger changes about the campaign

1. **Every headline number in the E2612 record was recomputed from the artifacts**, not copied from a summary,
   and one of the three readings that could have been published (−0.18650, weighting by the population field)
   is 0.089 bits away from the one that was — 4.5× the materiality band. A record that quotes a single number
   without naming the weighting field is not reproducible in the sense this project means.
2. **Three defects were found by reading code against artifacts, not by reading documentation**: the
   `dropped_duplicates` reporting gap (V-15), the off-domain generation probe (V-11), and the false docstring
   about the catalogue digest (V-23). None distorts a published measurement; all three degrade provenance, and
   two of them make a claim in tracked text untrue.
3. **One external number was corrected by measuring it** (V-20). The dataset remains a reasonable choice and
   the correction cuts its stated yield by more than half, which is the kind of difference that changes a
   slicing plan.
4. **Two clean results are worth stating positively** because a research phase otherwise only records faults:
   the baselines and the tokenizer sample are train-side only (V-13, V-14), and the matched-condition equality
   between E2612's arms holds on ten fields checked individually (V-02).
