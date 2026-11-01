# E2612 — equal-token data experiment: prose control against a technical mixture

**Status:** completed 2026-10-30 — preregistered in `4f45574` before either arm ran; both arms reached the
declared 2,400-step ceiling that day and the results are recorded below. **Outcome: decision rule 2** — H1
true, H2 false.
**Date ran:** 2026-10-30 (arm P 90.4 min, arm T 87.4 min, both on CPU)
**Type:** Research experiment, **preregistered**

**Question.** At a fixed model, tokenizer, split, seed and token budget, does training on a mix of prose and
technical registers lower held-out loss on technical text without raising it on prose, compared with
training on prose alone?

This is the unit [ADR-0041](../../docs/decisions/ADR-0041-next-training-target-general-base-with-technical-mixture.md)
names as next: data before architecture, at the measured 3.4M decoder configuration. It is a data-mixture
question, not a capability question. No claim about usefulness, answering, instruction following or Level 3
follows from it (K-001 stands).

## Corpus

`gutenberg-multiregister-v1` 1.0.0, pinned 2026-10-30.

| Property | Value |
| --- | --- |
| Fingerprint | `312571f9fa8351a8` |
| Snapshot checksum | `236c8d5a8e4eb6f2820a5feea1fd8edcf2d0311c9ff9b7281771015a975df61a` |
| Works | 430 |
| Documents | 9,388 |
| Characters | 149,000,499 |
| Catalogue | [`configs/corpora/gutenberg-multiregister-v1.toml`](../../configs/corpora/gutenberg-multiregister-v1.toml) |
| Data card | [`docs/data/gutenberg-multiregister-v1.md`](../../docs/data/gutenberg-multiregister-v1.md) |
| Acquisition rule | [`scripts/acquire_gutenberg_corpus.py`](../../scripts/acquire_gutenberg_corpus.py) |

Register by characters, from the catalogue's per-source records:

| Register | Works | Characters | Share |
| --- | ---: | ---: | ---: |
| prose (`P`) | 220 | 77,287,446 | 51.7 % |
| science (`Q` excluding `QA`) | 90 | 34,551,716 | 23.1 % |
| technology (`T`) | 60 | 16,711,383 | 11.2 % |
| medicine (`R`) | 25 | 8,084,482 | 5.4 % |
| mathematics (`QA`) | 35 | 12,923,330 | 8.6 % |

Register character counts are measured on each source's ingested text before document chunking; the
pinned corpus's 149,000,499 characters are measured over the chunked documents, which join paragraphs
with one blank line and therefore drop 0.4 % of the raw whitespace. The two figures are consistent, not
contradictory, and the pinned figure is the one a run consumes.

The selection walked a seeded shuffle (seed 20261030) of each register's English catalogue entries and
accepted works between 20,000 and 2,000,000 characters of ingested text; every rejection is recorded with
its reason in the work directory's `selection.json` (54 download failures, 29 ingest failures, 20 too short,
1 too long, 1 duplicate work). The corpus is one language and an archaic one: its "technical" register is
19th- and early-20th-century science, engineering and mathematics, not modern papers or code.

## Split

Whole-work, [ADR-0042](../../docs/decisions/ADR-0042-held-out-means-held-out-works.md): every chapter of a
work lands on one side. Split seed 7, 15 % validation and 15 % test. Measured on the smoke run below:
**312 works train, 56 validation, 62 test, zero works on more than one side** — the two-level leak that
produced K-023 is absent by construction.

*Added 2026-11-01 (after the results; nothing above changes).* The split puts no work on two sides; it cannot see two
works that are the same text. Measured on this exact split, the share of held-out 8-word windows that also occur
anywhere in the 312-work training side is **0.217 %** on the test side (0.156 % at 16 words) and **0.142 %** on the
validation side, with no held-out work above 3.0 % and none near 10 %
([`split-leakage-audit-2026-11-01.md`](../../docs/internal/engineering/split-leakage-audit-2026-11-01.md)). Exact word
repetition only; the effect on bits per character was not measured.
## Model and optimisation

Held identical to E2608 arm A, the reference configuration:

| Field | Value |
| --- | --- |
| Family | rotary decoder (`chakaso-decoder-checkpoint/1`) |
| `d_model` / layers / heads / feedforward | 256 / 4 / 8 / 640 |
| Context | 256 |
| Vocabulary | 4,096 byte-pair identifiers, `min_frequency` 3 (the plan default; see the note below) |
| Positions | tied input and output embeddings, RoPE θ = 10,000, RMSNorm, GELU |
| Parameters | 3,413,760 |
| Optimizer | AdamW, learning rate 6e-4, weight decay 0.01 |
| Schedule | 200 warmup steps, cosine to 0.1 of the peak |
| Batch | 32 micro-batches × 1 accumulation = 8,192 positions per step |
| Precision / device | float32, CPU |
| Seed | 7 |
| Steps | **2,400**, evaluation every 400, `patience` 99 so no arm early-stops |

## Tokenizer sampling

Fitting 4,096 merges on the whole training side of a 149M-character corpus is not affordable here: the fit
was killed twice by memory and, before the loop was indexed, 300,000 characters took 306 seconds. The
vocabulary is therefore fitted on a seeded sample of **12,000,000 characters** drawn by shuffling the
training documents (`tokenizer_sample_characters`), which the smoke measured at 747 documents and 70.4 s of
preparation including packing. The sample is part of the tokenizer's cache identity, so a vocabulary fitted
on one sample cannot be reused under another. Both arms share the sample, because both share the corpus and
the split.

**Correction before the mixture arm's result was read (2026-10-30).** This section first said
`min_frequency` 2. The runs used the plan's default of 3, which is what both reports record under
`tokenizer.min_frequency`; the protocol was fixed before either arm started and only this paragraph was
wrong. It is corrected here rather than left as a declaration the reports contradict, and no parameter
was changed after a result was read.

## Arms and what differs

Both arms use the same corpus, snapshot, split, tokenizer cache, model, seed, batch shape, step count and
evaluation packs. The only declared difference is which training works are packed:

| Arm | `training_group_prefixes` | Training pool, before the cap |
| --- | --- | --- |
| **P** (prose control) | `("prose-",)` | the 220 prose works on the training side |
| **T** (technical mixture) | none (all five registers) | all 430 works on the training side |

Packing is capped at **20,000,000 training identifiers** and shuffled with the split seed, so each arm
trains on a seeded sample of its pool rather than the alphabetically first works. 2,400 steps × 8,192
positions = **19,660,800 positions per arm** — about one pass over each arm's capped pool. Both arms see the
same number of tokens; what differs is their composition.

**Declared consequence of the cap.** The control's prose pool is ~25M tokens, so the cap covers about 80 %
of it; the mixture arm's pool is ~48M tokens, so the cap covers about 42 % of it — roughly 10M prose and
10M technical tokens. The two arms therefore see prose works of different *sizes* (a larger share in P),
though not of different kind. That is a property of training on a fixed budget from pools of different
size, and it is stated here rather than discovered later.

## Metrics, baselines and success bands

Held-out **bits per character** on the shared test pack (250,000 identifiers, sampled over a seeded shuffle
of held-out documents) and on each register partition scored separately (250,000 identifiers each), each
with the uniform and unigram baselines refitted on that arm's own training targets. The test works are
identical for both arms; only the model differs.

Declared before the runs:

* **H1 — the mixture helps technical text.** On the technical aggregate (science + technology + medicine +
  mathematics, weighted by each partition's scored characters), **T is at least 0.02 bits per character
  lower than P**.
* **H2 — the mixture does not damage prose.** T's prose partition is **no more than 0.02 bits per character
  worse** than P's.
* **H3 — the null.** Neither holds.

The 0.02-bit band is the materiality threshold used by E2611; a difference inside it is reported as
indistinguishable rather than as a direction.

## Decision rule, declared before the runs

1. **H1 and H2** — the data lever works at this scale. The next unit is the ~9.84M-parameter scale step in
   [`chakaso-next-training-design.md`](../../docs/research/chakaso-next-training-design.md), with the
   mixture recipe held.
2. **H1 only** — a trade-off. The mixture is reported as buying technical loss with prose loss, and the
   next unit reweights before scaling.
3. **H2 only, or H3** — the data lever is not shown at 3.4M parameters and 19.66M tokens. The next unit is
   capacity or optimization, and ADR-0041's ordering is amended on the evidence.

No band, metric or budget is changed after a result is read. A re-run with a different budget is a new
record.

## Known limits, declared in advance

* One seed per arm. A difference outside the band is still a single-draw estimate; E2609's decoder spread is
  not yet measured.
* 19.66M tokens at 3.4M parameters is ~5.8 tokens per parameter, not Chinchilla's ~20. This experiment tests
  the mixture at an affordable budget, not the scaling law.
* No code register. The design's 50/25/10/10/5 mixture included 10 % code; no code corpus, licence review
  or code tokenizer decision exists, so the technical arm is prose + science + technology + medicine +
  mathematics only.
* Held-out estimates come from sampled documents of held-out works (250,000 identifiers per side), not from
  every held-out document.
* The corpus is public-domain English from Project Gutenberg; a result here does not transfer to modern web
  text without a new corpus and a new record.

## The measured basis for the budget

From the smoke run `SMOKE-V3` (2026-10-30, 60 steps, vocabulary 1,024 because the smoke omitted
`--vocab-size`; the arms below declare 4,096):

| Measurement | Value |
| --- | --- |
| Training throughput | **4,493.7 positions/s** |
| 60-step wall clock | 169.05 s |
| Preparation (tokenizer fit + packing) | **70.42 s** |
| Peak resident set | **1,710 MB** |
| Whole-work split | 312 / 56 / 62 works, 0 on multiple sides |

At the measured rate one 2,400-step arm is ~73 minutes of training; the two arms and their evaluation are
expected to take about three hours. The host is a 16.6 GB, 18-logical-CPU Windows machine with no CUDA
device; no timing here is presented as intrinsic.

## Reproduction

```bash
python scripts/acquire_gutenberg_corpus.py --catalog pg_catalog.csv \
  --work data/corpora/gutenberg-multiregister-v1 \
  --quota prose=220 --quota science=90 --quota technology=60 --quota medicine=25 --quota mathematics=35
python scripts/acquire_gutenberg_corpus.py --work data/corpora/gutenberg-multiregister-v1 \
  --catalogue configs/corpora/gutenberg-multiregister-v1.toml --version 1.0.0 --retrieved 2026-10-30
chakaso corpus prepare --catalog configs/corpora/gutenberg-multiregister-v1.toml \
  --raw-dir data/corpora/gutenberg-multiregister-v1/raw \
  --cache-dir data/corpora/gutenberg-multiregister-v1/cache \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --report data/corpora/gutenberg-multiregister-v1/dataset-report.json \
  --data-card docs/data/gutenberg-multiregister-v1.md

# Arm P
chakaso experiment model --work-dir data/experiments/E2612/prose --experiment E2612-P \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --model-family decoder --vocab-size 4096 --context-length 256 --d-model 256 --num-layers 4 \
  --num-heads 8 --feedforward-dim 640 --batch-size 32 --grad-accum-steps 1 \
  --learning-rate 0.0006 --weight-decay 0.01 --warmup-steps 200 --schedule cosine --decay-to 0.1 \
  --max-steps 2400 --eval-every 400 --patience 99 --seed 7 \
  --whole-work-splits --shuffle-training-units --training-group-prefix prose- \
  --max-training-tokens 20000000 --max-validation-tokens 250000 --max-test-tokens 250000 \
  --tokenizer-sample-characters 12000000 --tokenizer-cache-dir data/experiments/E2612/tokenizer \
  --test-partition prose- --test-partition science- --test-partition technology- \
  --test-partition medicine- --test-partition mathematics- \
  --no-character-baseline --write-report data/experiments/E2612/prose/report.json

# Arm T: identical, with --experiment E2612-T, work-dir data/experiments/E2612/mixture,
# and no --training-group-prefix.
```

**Tokenizer provenance, added 2026-11-01 (after the results; no number above changes).** The vocabulary
`434debe82f949347` was fitted by the byte-pair merge loop as it stood in `6f3274f`..`0a42cc9`, which had the defect
recorded as K-024: it chose less frequent pairs than its documented rule. Current code fitting the same sample
yields `3abe4df4b123d4c5`, so **`--vocab-size 4096` followed by a fresh fit no longer reproduces this record**. The
vocabulary the arms actually used is committed as [`vocabulary-434debe82f949347.json`](vocabulary-434debe82f949347.json)
(file SHA-256 `d1061205f91289cc00cb873dbaff5f338dc97effe66b32b7bcfb48469ce2aed3`, identical to the cache file listed
under Artifacts) under [ADR-0044](../../docs/decisions/ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md).
A reproduction passes `--tokenizer-file experiments/E2612/vocabulary-434debe82f949347.json
--expected-tokenizer-fingerprint 434debe82f949347` in place of `--tokenizer-cache-dir`. The comparison between the
two arms was never affected: both scored the one shared vocabulary.

## Results

**Both arms ran, on 2026-10-30, to the declared 2,400-step ceiling.** This section was added after both
reports existed and reports nothing but what their fields say; the declarations above are unchanged.
Environment for both: Windows-11-10.0.26200-SP0, Python 3.13.5, `torch 2.13.0+cpu`, numpy 2.4.0, 14 torch
threads, device `cpu`. The machine was shared with the author's session, so wall clock is an observed
condition, not an intrinsic rate.

### Artifacts

| Artifact | Path | SHA-256 of the file |
| --- | --- | --- |
| Arm P report | `data/experiments/E2612/prose/report.json` | `3764ba3f9ea6ba3cdce9b84874338b482e365f68704e916baec899f6c5ed8215` |
| Arm T report | `data/experiments/E2612/mixture/report.json` | `295fbc7aef181cc37bb91b55a57f5e821b430ad1665109daeca0037c6005e6d5` |
| Arm P checkpoint | `data/experiments/E2612/prose/E2612-P-decoder.pt` | `80c4c1a2714ddbdd62022f96a1c460078cd7684abef1134a1049cc282162b362` |
| Arm T checkpoint | `data/experiments/E2612/mixture/E2612-T-decoder.pt` | `1db322fd1be4a4baddc924c1036979a5dd56ca13516b3fc3fe865496ac71c726` |
| Shared tokenizer cache | `data/experiments/E2612/tokenizer/gutenberg-multiregister-v1-tokenizer-4a15763d77123157.json` | `d1061205f91289cc00cb873dbaff5f338dc97effe66b32b7bcfb48469ce2aed3` |
| Corpus snapshot | `data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json` | `79c3ff88dad631f323c90d494cd1c74717a0bf4a955c1de2acd62ed0d170b159` |
| Aggregate, committed | [`results.json`](results.json) | `3875ae8c1a8e20861d762f4cff41b880f408fd4c6e154519a9b014f314495b8e` |

Two different digest conventions appear above and in the declarations, and a reader should not confuse them.
The file column is SHA-256 over the file bytes. The `artifact.checksum` and `Snapshot checksum` fields the
reports and this record's declarations carry are SHA-256 over **canonical content** (`DataManifest.checksum`,
`src/chakaso/training/manifest.py:344`), which is content identity and is stable across re-serialization; the
snapshot's content digest is `236c8d5a8e4eb6f2…` while its bytes hash to `79c3ff88dad631f3…`. Both are correct
and they are not the same number.

**The report-file digests identify these artifacts on this machine only.** Each report's `digest` field is the
SHA-256 of the report file itself (`3764ba3f…` for P and `295fbc7a…` for T match the file column exactly), and
the report body embeds `wall_clock_s`, `positions_per_second`, `peak_resident_bytes` and `torch_threads`. A
re-run that reproduces every metric to the last digit still cannot reproduce these digests, because it cannot
reproduce the timings. The `results.json` digest above has the same property. Reproduction here means the
numbers agree, not the hashes; where a record implies otherwise, the numbers are the claim.

`results.json` holds both arms' test, validation, partition and curve blocks, the matched-condition checks
below, the generation and resource blocks, and each source report's SHA-256, so the comparison can be
re-derived without the git-ignored tree.

### What was verified identical before the numbers were read

Each row is a field-by-field equality over the two reports, not an assumption:

| Condition | Value | Equal |
| --- | --- | --- |
| Corpus fingerprint | `312571f9fa8351a8` | yes |
| Split fingerprint / seed | `adf650435c3a9068` / 7 | yes |
| Tokenizer fingerprint | `434debe82f949347` (3,839 merges, size 4,096, `min_frequency` 3) | yes |
| Works per side (train/validation/test) | 312 / 56 / 62 | yes |
| Works on more than one side | 0 | yes — the K-023 leak is absent |
| Test identifiers | 256,125 | yes |
| Test positions / characters scored | 254,976 / 535,282 | yes |
| Parameters | 3,413,760 | yes |
| Seed / max steps / schedule / batch | 7 / 2,400 / cosine to 0.1 / 32 × 1 | yes |
| Per-partition tokens scored | equal for all five registers | yes |
| Declared difference | `training_group_prefixes`: `["prose-"]` against `null` | — the only one |

Arm P fitted the vocabulary (`reused_from_cache: false`, 145.55 s of preparation); arm T read the same cache
(`reused_from_cache: true`, 57.02 s). Characters per token on the test side is **2.0993** for both.

### Held-out result — the hypothesis test

Bits per character, lower is better. Δ is T minus P, so negative means the mixture arm did better.

| Register partition | works held out | docs | tokens scored | Arm P | Arm T | Δ (T − P) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose | 35 | 34 | 252,672 | **1.80926** | **1.83952** | **+0.03026** |
| science | 12 | 35 | 251,392 | 2.05867 | 1.94862 | −0.11004 |
| technology | 9 | 36 | 253,184 | 2.18647 | 1.90273 | −0.28374 |
| medicine | 1 | 7 | 46,592 | 1.99522 | 1.83814 | −0.15708 |
| mathematics | 5 | 34 | 250,624 | 3.04361 | 2.57777 | −0.46584 |
| **Technical aggregate** (the four non-prose partitions, weighted by scored characters, as declared) | | | | **2.39222** | **2.11638** | **−0.27584** |
| Whole test pack (all registers, one sampled pack) | 62 | 1,443 | 254,976 positions | 2.16369 | 2.05545 | −0.10825 |

Supporting numbers: token accuracy on the whole pack 0.43410 (P) against 0.44235 (T); perplexity 23.30 against
19.91; validation bits per character 2.12765 against 1.96680.

**Verdict against the declared bands.**

- **H1 — true, by 13.8× the band.** The mixture is **0.27584** bits per character lower on the technical
  aggregate against a declared threshold of 0.02. It is lower on every one of the four technical partitions,
  and the gain is largest on the partition the control arm had the most difficulty with: mathematics, whose
  control loss is 3.04361 against 1.99522–2.18647 for the other three, improves by **−0.46584** (15.3 %).
- **H2 — false, by 1.5× the band.** The mixture's prose partition is **0.03026** bits per character worse,
  against a declared tolerance of 0.02. It misses, and it misses by a small margin rather than a large one.
- **H3 — false**, since H1 holds.

**H1 does not depend on how the aggregate is weighted.** The aggregation field is a genuine trap in the report's
shape: a partition block's top-level `characters` is the register's **whole held-out population** (prose 12,414,225)
while `metrics.characters_scored` is what was actually **scored** (prose 539,534) — a 23× difference. The record
declared "weighted by each partition's scored characters", which is what the headline number uses. Both other
readings were computed from the same artifacts:

| Aggregate weighting | Arm P | Arm T | Δ (T − P) | H1 margin |
| --- | ---: | ---: | ---: | ---: |
| Scored characters (declared, used above) | 2.39222 | 2.11638 | −0.27584 | 13.8× band |
| Tokens scored | 2.40321 | 2.12437 | −0.27884 | 13.9× band |
| Population characters (the top-level field) | 2.19461 | 2.00811 | −0.18650 | 9.3× band |

The three differ by up to 0.089 bits per character — 4.5× the materiality band — which is enough that a reader who
picked the wrong field would report a different effect size. It is not enough to change the verdict: **H1 holds
under every available weighting.** H2 is a single partition and so has no weighting choice at all.

**Decision rule 2 applies as declared**: this is a trade-off. The mixture buys technical held-out loss with
prose held-out loss, and the next unit reweights before scaling. Rule 1's scale step is not licensed by this
result on its own, and rule 3's pivot to capacity is not required either — the data lever demonstrably works,
just not for free.

### The confound the record declared in advance, now measured

The 20,000,000-identifier packing cap was declared to make the two arms' prose exposure unequal. It did, and
the inequality is the most likely explanation for H2's narrow failure rather than an intrinsic cost of mixing.
Measured from the pinned manifest over the 312 works on the training side:

| Register | Training-side documents | Characters | Share of the pool |
| --- | ---: | ---: | ---: |
| prose | 3,354 | 52,513,592 | 50.51 % |
| science | 1,527 | 23,690,657 | 22.79 % |
| technology | 676 | 10,433,868 | 10.04 % |
| mathematics | 471 | 9,805,768 | 9.43 % |
| medicine | 487 | 7,520,945 | 7.23 % |
| **Total** | **6,515** | **103,964,830** | 100 % |

6,515 manifest documents map to the training side and 6,512 reach the split; the 3-document difference is
exactly the three duplicate groups the corpus reports, each of which had its second member dropped
(`mathematics-pg634/part-0011.md`, `part-0009.md`, `part-0003.md`; the `mathematics-pg633` halves are in the
training identifiers, the `pg634` halves are not). The deduplication is real and correct — and both members of
every pair were on the training side, so it removed no leakage. See the deviations section for why the report
still prints `dropped_duplicates: 0`.

Arm P packed 2,737 documents / 20,000,402 identifiers, all of them prose. Arm T packed 2,756 documents /
20,000,103 identifiers drawn by seeded shuffle from the whole 6,512-document pool. **This paragraph is a
proportional estimate, not a counted token total:** at the measured 50.51 % prose share of pool characters,
arm T's prose exposure is roughly **10.1 M identifiers against arm P's 20.0 M** — about half. So the arms do
not differ only in composition; they differ in how much prose each saw. The experiment as preregistered
therefore cannot separate "the mixture costs prose 0.030 bits" from "half as much prose costs 0.030 bits".
The gap between 0.03026 and the band is small enough that the difference between those two readings is not
something this record can settle.

What H1 does **not** suffer from is that asymmetry in the other direction: arm T saw strictly more registers
and the same total tokens, and it won on all four technical partitions, including on registers where P had
zero exposure. The mixture's technical gain is the robust half of this result.

### The curves: both arms truncated, the gap widening

Validation loss is on one shared pack, so this column is comparable across arms; train loss is not, and is
shown only to make that point.

| Step | P validation nats/token | T validation nats/token | gap | P train | T train |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8.3718 | 8.3718 | 0.0000 | 8.3694 | 8.3733 |
| 400 | 3.6231 | 3.4900 | −0.1332 | 3.1449 | 3.3764 |
| 800 | 3.3359 | 3.1905 | −0.1454 | 2.8145 | 2.9616 |
| 1,200 | 3.2495 | 3.0629 | −0.1866 | 2.7324 | 2.9052 |
| 1,600 | 3.1779 | 2.9783 | −0.1996 | 2.5248 | 2.7528 |
| 2,000 | 3.1508 | 2.9235 | −0.2273 | 2.5847 | 2.6891 |
| 2,400 | **3.1406** | **2.9032** | **−0.2374** | 2.5736 | 2.7765 |

Three readings, in decreasing order of confidence:

1. **Both arms are floors, not converged values.** Both completed `budget_ceiling` at step 2,400 with
   `stopped_early: false` and validation still falling in the last interval (P 3.1508 → 3.1406, T 2.9235 →
   2.9032). Every number in this record is what each arm had reached, not what each arm reaches.
2. **The advantage widens monotonically with training** (−0.133 → −0.237 over the observed evaluations). It
   is not an early-training artefact of one arm converging faster, and it had not plateaued when the budget
   ended, which is a reason to expect the mature gap to be larger than 0.27584 rather than smaller. This is an
   extrapolation from seven points on a still-descending curve, not a measurement.
3. **Arm P has the lower training loss at every evaluation and the higher held-out loss.** A prose-only model
   fits its narrower objective better; that is what a narrower distribution means. It is the clearest available
   demonstration that this project's cross-arm comparisons must be made on the shared held-out pack, and never
   on training loss.

### Generation: the loss gain did not reach the text

Both arms sampled through the recorded generation block. Neither writes usable text.

| | Arm P | Arm T |
| --- | --- | --- |
| Continuation samples generated | 5 of 5 | 5 of 5 |
| All bound-reached (truncated at the length limit) | yes, all five | yes, all five |
| Tail-looping | 2 | 3 |
| Cycling | 3 | 3 |
| `looping_fraction` | 0.4 | **0.6** |
| Mean bigram repetition | 0.675527 | **0.712437** |
| Edge-probe refusals (designed: the empty prompt) | 1 | 1 |
| Greedy characters/s (warm, 3 samples) | 59.61 | 78.34 |

The `samples_refused: 1` in both reports is the **designed edge probe** — the empty-prompt case, which
`generation.edge_samples` records as a named `ModelError` refusal — not a continuation the model failed to
produce; every one of the five continuation samples has status `generated`. Reading it as a failure would be a
misreading of the field, and the run's own summary conflates the two counts.

The same prompt, from each arm, verbatim:

```text
prompt: The tokenizer is trained by the project because

arm P:  of the world
        is the most of the world. The other is the most of the world who have
        been in the world of the world and the world of the world. The same
        thing is the most of the world and the world of the world

arm T:  the piston is
        the same as the same as the body is the same as the same as the body is
        the same as the same as the body is the same as the body is the same
        as the body is the same as the body
```

**Interpretation.** The arm with 0.108 better held-out bits per character produced the more repetitive
samples. This is consistent with E2610's diagnosis and extends it: at this scale, held-out loss and decoding
behaviour are not coupled tightly enough for a loss improvement to be read as a quality improvement, and it
was 0.030 bits worse on prose while being 0.276 better on the technical aggregate, so which model a reader
would prefer is not decidable from the loss table at all. K-001 stands — nothing here is evidence of
capability.

**A methodological defect in this block, found while reading the reports.** The plan field is
`sample_prompts: "repository"`, and the prompt source recorded on every sample is `chakaso-repo-text-v1`. The
generation probes therefore quote **this repository's own section headings** ("# Chakaso\n\nChakaso is a",
"## Status\n\nThe retrieval layer") to models trained on 19th-century Gutenberg text. That is the condition
the code itself warns about — `SubwordExperimentPlan.sample_prompts` at
`src/chakaso/training/subword_experiment.py:225` says a prompt quoting Chakaso's section titles "measures
memorization of a corpus the model never saw, not behaviour on the corpus it trained on" — and the alternative
is already implemented (`held_out`, at `subword_experiment.py:979`, taking the opening of test-side documents
in document order). The default was carried into E2612 unchanged. This does not touch H1 or H2, which are
computed on the shared held-out pack, and it does not change any loss number in this record; it does mean the
qualitative samples above are not a description of these models' behaviour on their own domain, and the
generation comparison between the arms should not be used for anything until it is re-run with
`--sample-prompts held_out`.

## Deviations from the preregistration

1. **`min_frequency` was mis-declared, not mis-run.** Corrected in the Tokenizer sampling section above before
   either result was read: the runs used the plan default of 3, the paragraph said 2, and no parameter changed
   after a result existed. Both reports record 3.
2. **The medicine partition fell far short of its declared size.** The record declared 250,000 identifiers per
   register partition; medicine scored **46,592**, because the whole-work split placed exactly **one** held-out
   medicine work (7 documents, 101,923 characters) and that is the entire pool. The number is real, and it is a
   single-work estimate; it carries 6 % of the weighted technical aggregate. Mathematics rests on 5 works and
   technology on 9 — the registers where the effect is largest are the registers with the least held-out text.
3. **The test pack exceeded its declared cap.** `max_test_tokens` was 250,000; both reports record 256,125 test
   identifiers (2.45 % over), because documents are appended whole until the cap is passed. Identical in both
   arms, so it cannot have favoured either.
4. **Wall clock overran the smoke's projection.** The record projected ~73 min per arm from the smoke's 4,493.7
   positions/s. The arms reported 3,851.8 (P) and 3,976.1 (T) positions/s — 11–14 % slower than the smoke — and
   finished in 5,426.02 s (90.4 min) and 5,246.90 s (87.4 min) including evaluation, for ~3.0 hours of both arms
   plus preparation. Peak resident memory was 2,040.1 MB (P) and 2,560.8 MB (T); the 520 MB difference between
   the arms is not explained by anything this record measured, and both arms pack about the same 20 M
   identifiers, so it should not be read as the cost of holding five registers.
5. **`documents_per_side.dropped_duplicates` reported 0 when 3 documents were dropped.** The exclusion is real
   and the report's own identifier lists prove it. The counter is wrong because `_split_whole_works` builds its
   `UnitSplit` without passing `dropped_duplicates` (`src/chakaso/training/data.py:332`), while the document-level
   path does pass it (`data.py:249`); the deduplication itself happens earlier and correctly at `data.py:186-207`.
   Consequence for measurement: none here. Consequence for the record format: no whole-work run can state its
   deduplication from its own report, which is a provenance gap, filed as an issue.

## Limitations

1. **One seed per arm.** The 0.02 band is a declared materiality threshold, not a confidence interval. H1
   clearing it 13.8× is unlikely to be noise; H2 missing it by 1.5× is well within what a single draw can do.
2. **Half the prose explanation is not separable here.** As measured above, T saw roughly half the prose
   identifiers P saw. "The mixture damages prose" and "less prose damages prose" are the same observation from
   inside this design.
3. **Both arms truncated**, so this is a comparison of two unfinished models at equal tokens.
4. **~5.8 tokens per parameter**, not a scaling regime. Nothing here tests a scaling law.
5. **Archaic English, no code.** The registers are 19th- and early-20th-century prose, science, technology,
   medicine and mathematics. The gain is a gain on that distribution.
6. **Unequal statistical support across partitions** — 35 held-out prose works against 1 medicine work — so the
   aggregate weight and the per-register confidence run in opposite directions.
7. **The generation block is off-domain** for the reason given above, and is reported as a defect, not as a
   result about these models' text.

## Conclusion

**The data mixture is a real lever at 3.4M parameters, and it is not free.** At matched tokens, model,
tokenizer, split, seed and step budget, training across five registers instead of prose alone lowered held-out
bits per character on technical text by **0.27584 (11.5 %)**, won on all four technical partitions individually,
and won the whole mixed test pack by **0.10825** — while raising prose by **0.03026 (1.7 %)**, which is outside
the declared tolerance by a factor of 1.5. Under this record's own rule that is outcome 2: reweight before
scaling.

Two things follow that are not in the declared rule. First, the size of the technical gain against the size of
the prose loss — 9× apart — says the interesting question is no longer *whether* to mix but *how much* of each
register, at a budget where prose exposure is itself a variable. Second, the confound is now measurable and not
just declared: a third arm that sees the same ~10M prose identifiers as T and nothing else would separate the
two explanations of H2, and costs one 90-minute run.

No claim about usefulness follows. The loss improvement did not reach the generated text, the arms are
truncated, the corpus is archaic, and K-001 is undisturbed.

## Follow-up

The next unit this result points at, in order of what it buys:

1. **E2613 — the exposure control.** Three arms at the same 20,000,000-token cap: T unchanged; P unchanged;
   and a new **P½** packing 10,000,000 prose identifiers only. If T's prose partition lands at P½'s prose loss,
   H2's failure is exposure, not mixture, and the mixture is close to free. If T is worse than P½, the mixture
   genuinely costs prose and the reweighting is a real trade-off with a price. One 90-minute arm, no new
   machinery. Run it with `--sample-prompts held_out` so its generation block describes its own domain.
2. **Reweight rather than choose.** The register shares above are the pool's, not a design. With P½ as the
   exposure control, sweep the prose share (for example 25 / 50 / 75 %) at fixed tokens and look for the point
   where the technical gain and the prose loss are both inside the band. This is the mixture-weight question the
   literature treats as a first-class hyperparameter, and this project has never measured it.
3. **A modern technical register.** The largest gains are in mathematics (−0.466) and technology (−0.284), and
   both are measured on archaic text with 5 and 9 held-out works. A corpus that carries modern technical English
   — and code, which the design wanted and this corpus cannot supply — is the change most likely to alter what
   the model learns, and it needs a licence review and a new record before a run.
4. **More held-out works per register**, especially medicine, whose 46,592 scored tokens and single held-out work
   make it the weakest number in this table and the one carrying 6 % of the aggregate weight.
