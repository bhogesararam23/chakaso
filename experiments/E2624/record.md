# E2624 — H17 at equal text: does the space-prefixed tokenizer still lower held-out loss when it reads the same text for the same fraction of the pool?

**Status:** measured — **W-b by the declared rule: at equal text and equal passes the space-prefixed arm is worse than the control, but it also made 30 % fewer updates, so this does not show the tokenizer is worse per character; with E2620 it shows the gain is in text per update, not in text.** Preregistered and committed (`da72cb0`) before the run; the Results section was added after the report was on disk. The design that separates the two confounds is [E2625](../E2625/record.md).
**Type:** Research experiment, **preregistered**. It is the follow-up [E2620](../E2620/record.md) declared for a V-a result: E2620 held updates and positions fixed, and its treatment therefore read about 40 % more *characters* than its control. This record holds the
**text read** fixed instead.
**Protocol file:** [`configs/experiments/E2624-space-prefixed-equal-text.toml`](../../configs/experiments/E2624-space-prefixed-equal-text.toml), read by `chakaso experiment study --protocol … --seeds 7`. A test holds it to E2620's on every field except six.

**Question.** E2620: with the model, parameters, updates and positions per update fixed, the space-prefixed tokenizer lowered the equal-weight mean from 2.0207 to 1.9943 (−0.0264) and the whole pack from 2.0555 to 2.0235 (−0.0321) on identical held-out documents. Is that because the tokenizer is better, or because 19.7M positions of the
space-prefixed vocabulary cover 40 % more text than 19.7M of the default one? Give the treatment exactly the control's training documents and the control's fraction of them, and ask again.

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed — E2617 (the control) and E2620: corpus `312571f9fa8351a8`, split seed 7 (`adf650435c3a9068`), the 3,413,760-parameter rotary decoder, batch 32 × context 256, AdamW 6e-4, cosine to 0.1, seed 7, this host, the held-out documents (chosen by counting with the control's vocabulary
`434debe82f949347`, as in E2620), and **the training documents**: the control's 2,756 training documents (20,000,103 identifiers under the default vocabulary, 43,459,447 characters), because the pool is capped so that the treatment selects the same units (below).

Moves — the tokenizer, which is E2620's fitted vocabulary `4a47f0281c81a9e5` read as a committed input (ADR-0044), and the schedule length, which follows from it:

| Field | E2620 (equal positions) | This arm (equal text) | Why |
| --- | ---: | ---: | --- |
| `max_training_tokens` | 20,000,000 | **14,033,424** | the cap that selects the control's own 2,756 units when counted with the space-prefixed vocabulary: the control's units total 14,033,424 of its identifiers, and 14,028,223 before the last unit, so the cap stops after the same unit (computed from the pinned snapshot before this record; the control's report states 2,756 documents) |
| `max_steps` | 2,400 | **1,684** | 2,400 × 14,033,424 ÷ 20,000,103: the same 0.983 of the pool read (13,795,328 positions ÷ 14,033,424) |
| `warmup_steps` | 200 | 140 | the same fraction (0.083) of the schedule |
| `eval_every` | 400 | 281 | six evaluations, as before |
| `tokenizer_file`, `expected_tokenizer_fingerprint` | — (fitted in the run) | the committed file and `4a47f0281c81a9e5` | the vocabulary is an input, not refitted |

On the training text the control reads 19,660,800 positions × 2.173 characters per identifier = 42.7M characters; this arm reads 13,795,328 × 3.097 = 42.7M. **Equal text, equal passes (0.983), equal pool, equal characters read** — with 29.8 % fewer updates and positions, and a schedule of the same shape.

## Metrics, margins and rules, declared before the run

Primary: O1 (equal-weight mean of the five partitions) and the whole pack, as E2620; margins E2614's (O1 tie 0.0062, separable 0.02; whole pack tie 0.00933). Let Δ = control (E2617) − treatment.

* **W-a — better per character of text read.** Δ(O1) > 0.02 **and** Δ(whole pack) > 0.00933. The tokenizer's gain survives holding the text fixed; the space-prefixed rule is then a strong candidate (it also costs 30 % fewer updates for the same loss).
* **W-b — worse.** Δ(O1) < −0.02 and Δ(whole pack) < −0.00933. The E2620 gain was more text, not a better tokenizer, and the tokenizer is harmful at equal text.
* **W-c — no measurable difference.** |Δ(O1)| ≤ 0.0062 and |Δ(whole pack)| ≤ 0.00933. At equal text the tokenizers tie, so E2620's gain is the extra text: 40 % more text per update, with 30 % fewer updates needed for the same loss.
* **W-d — indeterminate.** Anything else, including a gain that remains but is smaller than the E2620 margins: reported with the numbers, with the fraction of E2620's gain that survives, and seeds as the next step.

The reading is made by `scripts/read_tokenizer_arm.py --equal-text`, which refuses an arm that did not train on the control's documents (the training-documents count is the guard) and otherwise applies the same classification as E2620.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** The gain shrinks: Δ(O1) here is smaller than E2620's +0.0264 (a model that reads fewer updates and fewer positions has less opportunity to profit from the tokenizer).
* **P2.** The held-out characters per identifier match E2620's (2.93), since it is the same vocabulary on the same text.
* **P3.** The ordering of the five partitions' improvements is as in E2620 (mathematics smallest in relative terms).

**Stopping, failure and what may not change.** The arm runs its 1,684 updates with no early stopping. A crashed run is rerun once; a second failure is reported. A diverged run is recorded as diverged. The margins, outcomes and predictions are not changed after the result is read.

## Known limits, declared in advance

* One seed per arm; the control is E2617 (the E2614 configuration on this host). The spread of a treatment at fewer updates is unmeasured and assumed to be E2614's.
* The pool is cut at a unit boundary: the cap is the control's own total under the new vocabulary, so the selected units are the same, but a shuffle of the example windows within the pool is a function of the pool's size (the batch plan differs from the control's); two arms never see the same batches.
* "Equal text" is equal characters of training text read, to within the 0.983 convention and the tails each packing drops (under one block).
* Fewer updates and a different learning-rate trajectory in absolute steps accompany the shorter schedule; the shape is held, not the step count.
* Truncated budget, archaic English, handful-of-works partitions, as every record in the series.

## Cost

One run of 1,684 updates: about **45–50 minutes** (the vocabulary is not refitted; preparation took 462 s in E2620 including the fit).

## Reproduction

```bash
chakaso experiment study --work-dir data/experiments --experiment E2624 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2624-space-prefixed-equal-text.toml --write-report data/experiments/E2624/study.json
python scripts/read_tokenizer_arm.py --equal-text --treatment data/experiments/E2624/seed-7/report.json \
  --control experiments/E2617/E2617-report.json --out experiments/E2624/results.json
```

## Results (added 2026-11-02)

Produced by `scripts/read_tokenizer_arm.py --equal-text` from committed files: [`E2624-report.json`](E2624-report.json), [`../E2617/E2617-report.json`](../E2617/E2617-report.json) and [`results.json`](results.json) (mode `equal-text`). The guards passed: the arm trained on the control's 2,756 documents (14,033,424 identifiers under the committed
vocabulary `4a47f0281c81a9e5`, loaded as an input), the same corpus, split and parameter count, the control's held-out documents. 1,684 updates to the ceiling on this host (1,903 s, 7,617 positions per second), none early-stopped, none diverged.

| Partition | control (2,400 updates, default rule) | treatment (1,684 updates, space-prefixed) | Δ = control − treatment | relative | E2620's Δ (equal positions) |
| --- | ---: | ---: | ---: | ---: | ---: |
| mathematics | 2.5753 | 2.6054 | -0.0300 | -1.17 % | +0.0223 |
| medicine | 1.8379 | 1.8714 | -0.0335 | -1.82 % | +0.0277 |
| prose | 1.8392 | 1.8774 | -0.0382 | -2.08 % | +0.0217 |
| science | 1.9488 | 1.9850 | -0.0362 | -1.86 % | +0.0307 |
| technology | 1.9022 | 1.9397 | -0.0374 | -1.97 % | +0.0297 |
| **O1** | **2.0207** | **2.0558** | **-0.0351** | -1.74 % | +0.0264 |
| **whole pack** | **2.0555** | **2.0874** | **-0.0319** | -1.55 % | +0.0320 |

### The preregistered reading

* **Outcome: W-b (worse).** Δ(O1) = −0.0351 is under −0.02 and Δ(whole pack) = −0.0319 is under −0.00933, with every partition worse by 0.030–0.038. Characters per identifier on the held-out text are 2.933, as in E2620 (P2 held).
* **P1 held, and then some:** the gain did not merely shrink, it reversed (−0.0351 against +0.0264). **P3 did not hold as worded:** mathematics improves least in relative terms at equal positions (E2620) and is also the *least damaged* here (−1.17 % against −1.8 to −2.1 %), the same ordering seen from the other side.

### What it does and does not show — read this before the headline

* **The declared meaning of W-b is "the E2620 gain was more text, not a better tokenizer."** That sentence over-reaches, and the record's own limits say why: this arm made **1,684 updates against the control's 2,400** (13.8M positions against 19.7M). Reading the same text in 30 % fewer optimiser updates is a shorter schedule, and a shorter schedule is expensive at this scale: E2616's halved schedule cost prose
  0.134 bits per character and E2619 showed that repetition is not the cause. So the 0.035 here **cannot be split** between "the space-prefixed tokenizer is worse per character of text" and "30 % fewer updates cost this much". Nothing in this experiment can.
* **What the two records show together, and no more.** Each differs from the control in a different way: E2620 made the control's updates over 40 % more text and scored 0.026 *better*; this arm read the control's text in 30 % fewer updates and scored 0.035 *worse*. Neither isolates the tokenizer. A reading that the space-prefixed rule's value lies in text per unit of compute
  (positions), not in text, fits both, but it is a hypothesis here, not a finding: the two effects are not additive in any way this experiment tests.
* **The design that separates the confounds** is equal updates *and* equal text: the treatment makes the control's 2,400 updates over the control's own documents (so about 1.4 passes over the pool, inside the range E2619 found free). That is [E2625](../E2625/record.md); until it is run the tokenizer remains a candidate, with a result that depends on which of text or updates is held fixed.
* **Limits as declared:** one seed, margins from E2614's default-tokenizer spread, truncated budget, archaic English.
