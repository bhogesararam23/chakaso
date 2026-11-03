# E2625 — H17 with updates and documents both held fixed: is the space-prefixed tokenizer better per character of the same text, at the same compute?

**Status:** measured — **X-a: with updates and training documents both equal to the control's, the space-prefixed tokenizer still lowers held-out loss beyond seed noise (O1 −0.0229, whole pack −0.0286).** Preregistered and committed (`439392b`) before the run; the Results section was added after the report was on disk.
**Type:** Research experiment, **preregistered**. It is the cell [E2624](../E2624/record.md) named as missing. E2620 held updates and positions fixed and gave the treatment 40 % more text; E2624 held the text fixed and gave the treatment 30 % fewer updates. Each moved a second thing.
**Protocol file:** [`configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml`](../../configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml), read by `chakaso experiment study --protocol … --seeds 7`. A test holds it to E2620's on exactly three fields.

**Question.** With the model, the **2,400 updates**, the 8,192 positions per update and **the control's own 2,756 training documents** all unchanged, does the space-prefixed vocabulary score lower held-out bits per character than the default? The updates are the control's, so compute is equal; the documents are the control's, so the
distinct text is equal. What differs is the tokenizer and what follows from it: the treatment makes 1.40 passes over the pool (19,660,800 positions ÷ 14,033,424 identifiers) where the control made 0.98, which [E2619](../E2619/record.md) found costs nothing measurable between one and two passes.

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed — E2617 (the control): corpus `312571f9fa8351a8`, split seed 7 (`adf650435c3a9068`), the 3,413,760-parameter rotary decoder, **2,400 updates × 8,192 positions**, warmup 200, cosine to 0.1, AdamW 6e-4, seed 7, this host, the held-out documents (chosen by counting with `434debe82f949347`, as in E2620 and E2624), and the **training documents**
(2,756 units, 43,459,447 characters).

Moves — the vocabulary, E2620's `4a47f0281c81a9e5` read as a committed input, and with it the cap that selects the control's documents when counted with it (`max_training_tokens` = 14,033,424, E2624's). Everything else is E2620's. The treatment's distinct text equals the control's; the **text read** is 1.40× the control's (60.9M characters against 42.7M),
because the same positions cover more characters; this is the one thing the design lets differ and it is declared, not removed.

## The three arms, together

| Arm | updates | training documents | passes over them | characters read | tokenizer |
| --- | ---: | --- | ---: | ---: | --- |
| control (E2617) | 2,400 | 2,756 (43.5M chars) | 0.98 | 42.7M | default |
| E2620 | 2,400 | 3,802 (pool 20,000,297 identifiers, ~58M chars) | 0.98 | ~57.7M | space-prefixed |
| E2624 | 1,684 | the control's 2,756 | 0.98 | 42.7M | space-prefixed |
| **E2625** | **2,400** | **the control's 2,756** | **1.40** | **~60.9M** | space-prefixed |

(E2620's document count and pool are from its report; the character figures are the identifier counts times the held-out text's characters per identifier and are approximate.)

## Metrics, margins and rules, declared before the run

Primary: O1 and the whole pack; margins E2614's (O1 tie 0.0062, separable 0.02; whole pack tie 0.00933). Δ = control (E2617) − treatment.

* **X-a — better per character of the same text, at equal compute.** Δ(O1) > 0.02 **and** Δ(whole pack) > 0.00933. The tokenizer lowers loss when only the tokenizer and the passes differ; the space-prefixed rule is then a strong candidate on both readings (compute and text), and the replacement experiment (seeds) is the next record.
* **X-b — worse.** Δ(O1) < −0.02 and Δ(whole pack) < −0.00933. At equal compute over the same distinct text the tokenizer hurts; E2620's gain was the extra distinct text (3,802 documents, about 58M characters, against 2,756 and 43M), and the rule is not a candidate on a compute reading either.
* **X-c — no measurable difference.** |Δ(O1)| ≤ 0.0062 and |Δ(whole pack)| ≤ 0.00933. Over the same text and updates the tokenizers tie: E2620's gain was the extra distinct text it reached, and the space-prefixed rule buys reach (more text per position), not quality.
* **X-d — indeterminate.** Anything else, reported with the numbers and where it falls between E2624's −0.035 and E2620's +0.026.

The reading is made by `scripts/read_tokenizer_arm.py --same-documents`, which refuses an arm whose updates or training documents differ from the control's.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** Δ(O1) lies between E2624's −0.0351 and E2620's +0.0264 (E2625 differs from E2620 only in having less distinct text, and from E2624 only in having more updates).
* **P2.** E2625's O1 is worse than E2620's: the same updates over 25 % less distinct text cannot do better by the amount E2619 allows only for repetition up to two passes (1.40 passes is inside the free range, so the two should differ by a small amount, not a large one).
* **P3.** Every partition moves in the same direction as O1.

**Stopping, failure and what may not change.** The arm runs its 2,400 updates with no early stopping. A crashed run is rerun once; a second failure is reported. The margins, outcomes and predictions are not changed after the result is read.

## Known limits, declared in advance

* One seed per arm; margins are E2614's default-tokenizer spread. The three treatment arms (E2620, E2624, E2625) share one vocabulary and one fitted-sample, so they cannot show what refitting would change.
* The treatment re-reads: 1.40 passes over the pool. E2619 found one and two passes indistinguishable for prose at the default tokenizer; the same for a different tokenizer and for the other registers is assumed.
* The pool cut is at a unit boundary (the control's own total under the new vocabulary); the batch plan differs, since it is a function of the pool's size.
* Truncated budget, archaic English, handful-of-works partitions.

## Cost

One run of 2,400 updates: about **65 minutes** (the vocabulary is read, not fitted).

## Reproduction

```bash
chakaso experiment study --work-dir data/experiments --experiment E2625 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml --write-report data/experiments/E2625/study.json
python scripts/read_tokenizer_arm.py --same-documents --treatment data/experiments/E2625/seed-7/report.json \
  --control experiments/E2617/E2617-report.json --out experiments/E2625/results.json
```

## Results (added 2026-11-03)

Produced by `scripts/read_tokenizer_arm.py --same-documents` from committed files: [`E2625-report.json`](E2625-report.json), [`../E2617/E2617-report.json`](../E2617/E2617-report.json) and [`results.json`](results.json) (mode `equal-updates-and-documents`). The guards passed: the control's 2,400 updates and the control's 2,756 training documents
(14,033,424 identifiers under the committed vocabulary `4a47f0281c81a9e5`), the same corpus, split and parameter count, the control's held-out documents. 2,400 updates to the ceiling on this host (2,605 s, none early-stopped, none diverged).

| Partition | control | E2625 treatment | **Δ here** | Δ in E2620 (equal updates, +40 % text) | Δ in E2624 (equal text, 30 % fewer updates) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mathematics | 2.5753 | 2.5570 | +0.0183 | +0.0223 | -0.0300 |
| medicine | 1.8379 | 1.8151 | +0.0228 | +0.0277 | -0.0335 |
| prose | 1.8392 | 1.8229 | +0.0163 | +0.0217 | -0.0382 |
| science | 1.9488 | 1.9228 | +0.0260 | +0.0307 | -0.0362 |
| technology | 1.9022 | 1.8711 | +0.0311 | +0.0297 | -0.0374 |
| **O1** | **2.0207** | **1.9978** | **+0.0229** | +0.0264 | -0.0351 |
| **whole pack** | **2.0555** | **2.0269** | **+0.0286** | +0.0320 | -0.0319 |

### The preregistered reading

* **Outcome: X-a.** Δ(O1) = **+0.0229** is over the 0.02 separable margin and Δ(whole pack) = **+0.0286** over the 0.00933 tie margin. Every partition improves (0.016 to 0.031). The tokenizer lowers loss when only the tokenizer, and the 1.40 passes it implies, differ from the control — at the same compute and over the same distinct text.
* **Predictions.** P1 held (+0.0229 lies between E2624's −0.0351 and E2620's +0.0264). P2 held: E2625's O1 is worse than E2620's, by **0.0035** (whole pack 0.0035), a small amount. P3 held: all five partitions move the same way.

### What the three records show together

The three treatment arms differ from the control as follows and no other way: E2620 = more distinct text, same updates; E2624 = same text, 30 % fewer updates; E2625 = same text, same updates. Read as a set:

* **Most of E2620's gain is the tokenizer, not the extra text.** E2625 keeps 87 % of E2620's O1 gain (0.0229 of 0.0264) and 89 % of its whole-pack gain, with 25 % less distinct text. The remaining 0.0035 is the value, at this budget, of 15M more distinct characters (58M against 43M), consistent with E2619's finding that repetition up to two passes is nearly free.
* **E2624's loss was the updates.** E2625 and E2624 differ in the number of updates (2,400 against 1,684) and the passes (1.40 against 0.98), over the same text and the same tokenizer: O1 +0.0229 against −0.0351, a difference of **0.058 bits per character for 30 % fewer updates** at this scale. The W-b of E2624 is therefore not a statement about the tokenizer, as that record said it could not be.
* **The tokenizer's gain is robust to which of text or updates is held fixed** (it is +0.023 or +0.026 whenever the updates are the control's) and **is bought by compute, not by text:** at the control's compute the arm with fewer identifiers per character wins by about 1.1 % of O1.

### What it does and does not establish

It establishes, for one seed, one vocabulary and one budget, that the space-prefixed pre-tokenizer lowers held-out bits per character by about 0.023 on the equal-weight mean at the control's compute and documents. **It does not change a default:** replacement needs the seeds the E2620 consequence names (the margins are E2614's, measured with the default tokenizer;
the treatment's seed spread is unmeasured), a refit sensitivity check, and a decision with its own record. The vocabulary is also fitted on the corpus the arms are scored against; whether the gain holds on another corpus is untested. The probe instruments (E2622, E2623) have not been run on these three checkpoints, and the tokenizer is the part of the model they hold fixed in their own design.
Nothing here is capability (K-001).
