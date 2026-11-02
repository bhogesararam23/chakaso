# E2620 — H17: does folding the single space into the next word lower held-out bits per character at equal positions?

**Status:** measured — **V-a: at equal updates and positions, the space-prefixed tokenizer lowers held-out bits per character beyond seed noise on both decision-bearing measures.** Preregistered and committed (`bc3b4a0`) before the run; the Results section was added after the report was on disk. The result confounds a better tokenizer with 40 % more text per update (declared); the equal-characters follow-up is [E2624](../E2624/record.md).
**Type:** Research experiment, **preregistered**. It is not one of tasks T11–T15; it is the first experiment on a tokenizer hypothesis (H17) filed on
2026-11-02, queued **after** E2617–E2619 and not displacing them.
**Protocol file:** [`configs/experiments/E2620-space-prefixed.toml`](../../configs/experiments/E2620-space-prefixed.toml), read by
`chakaso experiment study --protocol … --seeds 7`. A test holds it to E2614's plan on every field except four, all about the tokenizer.

**Question.** The committed vocabulary spends 34–39 % of its identifiers on whitespace and 30–36 % on a single space
([H17](../../docs/research/hypotheses.md), research log 2026-11-02, `scripts/measure_token_classes.py`). A pre-tokenizer in which a word carries its
single preceding space ([ADR-0048](../../docs/decisions/ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md)) removes most of
them. With the model, parameters, updates and positions per update unchanged, the same 19,660,800 positions then cover more text. Does that lower
bits per character on the same held-out text?

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed — E2614 seed 7 / [E2617](../E2617/record.md), unchanged: corpus `gutenberg-multiregister-v1` 1.0.0 (`312571f9fa8351a8`), whole-work split with split seed
7 (`adf650435c3a9068`), the 3,413,760-parameter rotary decoder, **2,400 updates × 8,192 positions**, cosine to 0.1, AdamW 6e-4, warmup 200, 20,000,000 training
identifiers, seed 7, this host (4 threads, torch 2.15.0+cpu), the vocabulary size (4,096), the frequency floor (3) and the fitting sample (12,000,000 characters, the
seeded shuffle of the training documents). The control is **E2617's run**: same host, same code, committed report extract.

Moves — **the pre-tokenizer, and with it the vocabulary the run fits.** The treatment fits its vocabulary in the run under the space-prefixed rule instead of reading
the committed one. The fit costs about 35 minutes on this host for the default rule; the space-prefixed rule's cost is unmeasured.

**The held-out text is held fixed by construction, not by luck.** The validation and test packs are capped in identifiers, counted with the vocabulary being
packed, so a more economical tokenizer would reach the cap on more characters and be scored on different documents. The treatment's protocol therefore names
`evaluation_selection_tokenizer_file`: the documents are chosen by counting with the committed (default-rule) vocabulary — which is what the control did by
default — and packed with the treatment's own. The report states the selection vocabulary's fingerprint and the documents chosen; the reading script refuses
a treatment whose documents differ from the control's. A residual difference is declared: each pack drops its final partial block, so the characters scored
differ by at most one block per pack (≤ 256 identifiers).

**What "equal positions" means, and what it is not.** The treatment reads about 1.4× more characters in the same updates (design arithmetic, below). A gain
could be a better tokenizer or simply more text per update. This record's question is the first: *at equal compute*, is the tokenizer better. It does not separate
"better per character" from "more characters"; an equal-characters arm (fewer updates for the treatment) is the declared follow-up, run only if the result is V-a.

## Metrics

Primary: **O1**, the equal-weight mean of the five register partitions' bits per character (the convention E2615 declared and E2618 kept). Also decision-bearing:
the **whole-pack** bits per character. Reported, not decision-bearing: each partition; characters per identifier (held-out text); the share of identifiers that are
whitespace-only, by `scripts/measure_token_classes.py`; the training pack's size in identifiers and in characters; the vocabulary's fit time.

## Margins and decision rules, declared before the run

Margins are E2614's, for a difference between two single-seed arms (tie at or under 2.77·*s*, separable over 2.77·*s*<sub>hi</sub> or the kept band): O1 tie **0.0062**,
separable **0.02**; whole pack tie **0.00933**, separable **0.02683**. Let Δ = control − treatment (positive when the treatment is better).

* **V-a — better.** Δ(O1) > 0.02 **and** Δ(whole pack) > 0.00933. The tokenizer lowers loss at equal positions beyond seed noise on both measures. Consequence: the space-prefixed
  rule becomes the *candidate* for the next experiments; the follow-up equal-characters arm and two more seeds on each side are run before it replaces the default.
* **V-b — worse.** Δ(O1) < −0.02 **and** Δ(whole pack) < −0.00933. H17 is falsified at this scale in the strong direction; the record states how many identifiers per character were gained and lost.
* **V-c — no measurable difference.** |Δ(O1)| ≤ 0.0062 and |Δ(whole pack)| ≤ 0.00933. H17 is falsified as worded: 30–36 % fewer identifiers did not move held-out loss at equal updates. The text covered per update rose and the loss did not follow.
* **V-d — indeterminate.** Anything else. Reported with the numbers; seeds are the next step before any claim.

**Descriptive predictions (stated so a surprise is recognisable; not decision-bearing).**

* **P1.** Characters per identifier on the held-out text rises from the control's 2.1–2.3 to at least 2.8 (the arithmetic is 43–55 %). *Contradicted* by a rise under 30 %.
* **P2.** The whitespace-only share of identifiers falls under 10 %.
* **P3.** The mathematics partition improves least in relative terms, since its punctuation- and symbol-heavy text gains least from folding spaces.

**Stopping, failure and what may not change.** The arm runs its 2,400 updates with no early stopping. A crashed run is rerun once; a second failure is reported. A fit that stops short of 4,096
identifiers is a failed run, not a smaller vocabulary (the existing refusal). A diverged run is recorded as diverged. The margins, outcomes, predictions and the objective are not changed after the result is read.

## Known limits, declared in advance

* One seed per arm; the margins assume E2614's spread at the natural mixture applies to a different tokenizer. A new vocabulary may change the spread.
* The control and treatment differ in the vocabulary **and** in how much text the 19.66M positions cover (intended, but a confound with "more text").
* The text scored is the same documents; bits per character is the unit, so the tokenizer change does not move the metric's definition. The tails differ by at most one block per pack.
* One vocabulary fit per rule: the merge search is deterministic given its input, so a refit reproduces; the sensitivity of the outcome to the fitting sample is not tested.
* Truncated budget, archaic English, handful-of-works partitions, as every record in the series.

## Cost

The fit (~35 minutes for the default rule on this host; unmeasured for this one) plus one ~65-minute arm: about **1.7–2 hours**, sequential, queued after the E2617–E2619 campaign. The fitted vocabulary is committed
beside the record afterwards (ADR-0044) so the run stays re-trainable.

## Reproduction

```bash
chakaso experiment study --work-dir data/experiments --experiment E2620 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2620-space-prefixed.toml --write-report data/experiments/E2620/study.json
python scripts/read_tokenizer_arm.py --treatment data/experiments/E2620/seed-7/report.json \
  --control experiments/E2617/E2617-report.json --out experiments/E2620/results.json
```

## Results (added 2026-11-02)

Produced by `scripts/read_tokenizer_arm.py` from files that are committed: [`E2620-report.json`](E2620-report.json) (the treatment), [`../E2617/E2617-report.json`](../E2617/E2617-report.json) (the control) and [`results.json`](results.json). The fitted vocabulary is committed beside the record as
[`vocabulary-4a47f0281c81a9e5.json`](vocabulary-4a47f0281c81a9e5.json) ([ADR-0044](../../docs/decisions/ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md)); it loads, fingerprints as `4a47f0281c81a9e5`, and is 4,096 identifiers under the space-prefixed rule (frequency floor 3, fitted on the declared
12,000,454-character sample). The arm ran to the 2,400-update ceiling on this host (3,959 s, 5,531 positions per second); the fit and corpus preparation took 462 s of the run's own preparation plus the fit (about 32 minutes in all). The reading script's guards passed: same corpus, split, parameter count, updates and batch; the held-out documents
(34 test documents, every partition's count) are the control's, chosen by the control's own vocabulary (`434debe82f949347`).

| Partition | control (default rule) | treatment (space-prefixed) | Δ = control − treatment | relative |
| --- | ---: | ---: | ---: | ---: |
| mathematics | 2.5753 | 2.5530 | +0.0223 | 0.87 % |
| medicine | 1.8379 | 1.8102 | +0.0277 | 1.51 % |
| prose | 1.8392 | 1.8175 | +0.0217 | 1.18 % |
| science | 1.9488 | 1.9181 | +0.0307 | 1.57 % |
| technology | 1.9022 | 1.8725 | +0.0297 | 1.56 % |
| **O1, equal-weight mean** | **2.0207** | **1.9943** | **+0.0264** | 1.31 % |
| **whole pack** | **2.0555** | **2.0235** | **+0.0320** | 1.56 % |

Characters scored per partition differ between the arms by under 0.1 % (the declared tail residual), so the bits-per-character figures are over the same text. Characters per identifier on the held-out text: **2.099 → 2.933** (+39.7 %).

### The preregistered reading

* **Outcome: V-a.** Δ(O1) = **+0.0264** is over the 0.02 separable margin and Δ(whole pack) = **+0.0321** over the 0.00933 tie margin; the tokenizer lowers loss at equal positions beyond seed noise on both measures. Every one of the five partitions improves (by 0.022 to 0.031).
* **Predictions.** P1 held: characters per identifier 2.93 (≥ 2.8; +40 %, over the 30 % floor). P2 held: whitespace-only identifiers fall to 5.8–8.6 % of the fitted vocabulary's output on 1.5M-character samples of each register (against 34–39 %), and single spaces to 0.2–3.8 % (from 30–36 %); characters per identifier there are 2.61–3.34
  (from 1.99–2.31). P3 held: mathematics improves least in relative terms (0.87 %, against 1.2–1.6 % for the others), the register whose symbol-heavy text gains least from folding spaces.
* **Consequence under the declared rule.** The space-prefixed rule becomes the **candidate** for the next experiments. It does **not** replace the default on this record: the equal-characters arm and two more seeds on each side come first.

### What it does and does not show

* **It shows** that with the model, parameters, updates and positions per update unchanged, a vocabulary that does not spend a third of its identifiers on word gaps scores 1.3 % lower on the equal-weight mean (and 1.6 % on the whole pack) over identical held-out documents, on one seed.
* **It does not show** that the tokenizer is better *per character of training text*. The treatment's 19.7M positions cover 40 % more characters (about 57.7M against 41.3M of training text), so a model that simply saw more text would improve even with the default rule. The size of that effect is not measured here; [E2624](../E2624/record.md) holds the characters
  seen fixed instead.
* **It does not show** the gain survives another seed, a longer budget, a larger vocabulary or another corpus. The margins assumed here are E2614's, measured with the default tokenizer; the treatment's seed spread is unmeasured.
* The fitted vocabulary and the control share the 12.0M-character fitting sample and frequency floor; the sensitivity of the outcome to that sample is untested. Nothing here is capability (K-001).
