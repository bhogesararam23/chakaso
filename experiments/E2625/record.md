# E2625 — H17 with updates and documents both held fixed: is the space-prefixed tokenizer better per character of the same text, at the same compute?

**Status:** preregistered, not yet run. Committed before the run starts; the Results section is added after the report is on disk.
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
