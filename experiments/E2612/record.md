# E2612 — equal-token data experiment: prose control against a technical mixture

**Status:** preregistered 2026-10-30, before either arm ran. Results follow in the same file after the runs,
in a section that is added, never substituted for the declarations below.

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

## Model and optimisation

Held identical to E2608 arm A, the reference configuration:

| Field | Value |
| --- | --- |
| Family | rotary decoder (`chakaso-decoder-checkpoint/1`) |
| `d_model` / layers / heads / feedforward | 256 / 4 / 8 / 640 |
| Context | 256 |
| Vocabulary | 4,096 byte-pair identifiers, `min_frequency` 2 |
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
  --expected-corpus-fingerprint 312571f9fa8351a8

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
