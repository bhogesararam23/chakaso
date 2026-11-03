# E2632 — does the space-prefixed rule's compression advantage survive a different fitting sample?

**Status:** preregistered — committed before the measurement runs.
**Type:** Research measurement, **preregistered**. It fits vocabularies and encodes text. Nothing is trained, no
checkpoint is loaded, and no loss number is produced.
**Tool:** [`scripts/measure_refit_sensitivity.py`](../../scripts/measure_refit_sensitivity.py).

## Question

[E2630](../E2630/record.md) made H17 adoption-grade on five paired seeds: Δ(O1) +0.026431 with a 95 % Student-*t*
interval of [+0.023212, +0.029650], every draw positive, and the treatment's own seed spread *smaller* than the
control's. [E2625](../E2625/record.md) and [E2628](../E2628/record.md) had already separated the gain from the update
count and reproduced it on a second host. What all five of those readings share is **one vocabulary**:
`4a47f0281c81a9e5`, the 4,096 identifiers learned from one 12,000,000-character sample of the training side
(`tokenizer_sample_characters = 12000000`, `min_frequency = 3`).

So the adoption question has a half that no number of seeds can answer, and it is the half
[ADR-0048](../../docs/decisions/ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md) leaves open:
**if the same rule is fitted on a different sample of the same corpus, does it still compress the same amount better?**
If a second fit of the space-prefixed rule lands at a different compression, then part of every measured gain is the
property of one merge history, and a default change would be shipping a sample as well as a rule.

This record answers it on the mechanism directly, because the mechanism is arithmetic: a pre-tokenizer decides where a
piece may start, and the number that follows is characters per identifier on text no fit was shown.

## What is fitted, and what is encoded

Corpus `gutenberg-multiregister-v1` (`312571f9fa8351a8`), split by whole work at split seed 7
(`adf650435c3a9068`) through `chakaso.training.subword_experiment.split_of_corpus` — the run's own rule, not a copy.

* **Four fitting samples**, drawn from the **training side** by
  [`tokenizer_sample`](../../src/chakaso/training/subword_experiment.py) at `sample_characters = 4,000,000` and seeds
  **7, 11, 13, 17** — the same seeded-shuffle-in-whole-documents rule the plans use, at a third of the size because the
  exact merge loop costs 2,116 s at 12.7M characters and 494 s at 4.0M (K-024's measurements), and eight fits is the
  budget this record can pay.
* **Each sample is fitted twice**, once per rule: `word|space|symbol` (the default, the one every record before E2620
  used) and `space-prefixed-word|space-prefixed-symbol|whitespace-run` (ADR-0048's rule), both at **4,096 identifiers
  and `min_frequency` 3** — the committed vocabularies' own parameters.
* **One fixed evaluation text**, drawn once from the **test side** at `eval_characters = 1,000,000` and
  `eval_seed = 20261104`, before any fit exists. No fit sees it; every vocabulary encodes exactly it.
* **Both committed vocabularies are measured on the same text** — `434debe82f949347` (E2612's, fitted at 12M
  characters under the default rule) and `4a47f0281c81a9e5` (E2620's, fitted at 12M under the space-prefixed rule) — so
  the small fits are tied to the artifacts the records cite rather than to a private idea of "the default".

The readings are: characters per identifier per vocabulary; the ratio space-prefixed over default **within each
sample** (that pairing is the point — two rules on one sample differ in the rule and nothing else); the Jaccard overlap
and Spearman rank agreement of the merge lists between fits of one rule; and the same compression figures for the two
committed files.

## Outcome rules, declared before the measurement

Let `r` be the within-sample ratio, over the four samples, and let `s = (max r − min r) / mean r` be its relative
spread. The pilot at 300k characters and 1,024 identifiers measured `mean r = 1.3284` with `s = 0.0028` on two samples;
the thresholds below are set against that magnitude, not against a guess about this run's own number.

* **R-a — the advantage is the rule's.** Every `r > 1.05` and `s ≤ 0.05`. Consequence: the space-prefixed gain cannot
  be a property of one merge history, and E2630's five seeds plus this is what
  [ADR-0048](../../docs/decisions/ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md)'s adoption
  record has to rest on. The ADR still decides; this record only removes one objection.
* **R-b — a second fit changes the answer.** Any `r ≤ 1.0`, or `s > 0.15`. Consequence: refit sensitivity is real at
  this scale, the rule stays a candidate, and what adoption would need is a vocabulary fitted on a second 12M sample
  and an arm trained with it — a ~35-minute fit plus an ~1-hour arm, not a default change.
* **R-c — in between** (`0.05 < s ≤ 0.15`, every `r > 1.05`). Consequence: the direction is robust and the magnitude is
  not, and the adoption record says so in those words rather than rounding either side.

**Descriptive, decision-free:** the merge-stability figures, the absolute characters-per-identifier of each fit, and the
two committed vocabularies' compression on the same text. P1 below is about the first, P2 about the third.

**Predictions (stated so a surprise is recognisable).**

* **P1.** `s` is small — a few per cent — while the Jaccard overlap between two fits of one rule is much below 1. Two
  samples of 4M characters will learn substantially different merge *lists* and very similar *compression*, because the
  frequent merges are the same ones and the tail differs.
* **P2.** Both committed vocabularies compress better than any 4M fit of their own rule, and the space-prefixed one
  stays around 2.9 characters per identifier against the default's 2.10 (E2620 measured those two numbers on its own
  evaluation text; the ratio, not the absolutes, is what should resemble this record's).
* **P3.** The four within-sample ratios agree in sign with each committed pair: `4a47f0281c81a9e5` buys more characters
  per identifier than `434debe82f949347` on the same evaluation text.

## Known limits, declared in advance

* **4M characters is not 12M.** The committed vocabularies were fitted on three times the text; a fit at 4M has fewer
  characters per identifier in absolute terms (P2 says how much fewer). What is compared is a *ratio between two rules
  on the same sample*, which is the quantity that has to be stable for the conclusion to transfer.
* Compression is the mechanism, not the loss. Four seeds of loss already measured the outcome; this record tests
  whether the mechanism depends on the sample. If R-b fires, the loss conclusion is *not* overturned — it becomes
  conditional on one vocabulary, and the arm that would un-condition it is named in the consequence.
* One corpus, one split, one language, one `min_frequency` and one identifier count. A different target size could
  change the balance: the space-prefixed rule spends identifiers on words rather than gaps, and the published optimum
  vocabulary grows with model size.
* Whole-document sampling means the four samples differ in which documents they include, which is exactly the variation
  the record is about; the evaluation text is one sample, held fixed, so no reading compares two different texts.
* No capability claim. This is a statement about how a text is cut (K-001 unchanged).

## Cost

Eight fits at 4,000,000 characters and 4,096 identifiers. Vocabulary preparation at that size was measured at
**494 s per fit** (K-024), so **roughly 65–75 minutes of CPU** for the eight, plus encoding a fixed 1M-character
evaluation text by ten vocabularies (seconds each). A pilot of four fits at 300k characters and 1,024 identifiers ran
end to end in 16 s on this host, which is what bounds the script's own overhead. It runs after E2631's arm, one job at a
time on this machine.

## Reproduction

```bash
.venv-nn/Scripts/python.exe scripts/measure_refit_sensitivity.py \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --fit-seeds 7,11,13,17 --sample-characters 4000000 --identifiers 4096 --min-frequency 3 \
  --split-seed 7 --eval-seed 20261104 --eval-characters 1000000 \
  --reference-vocabulary committed-default=experiments/E2612/vocabulary-434debe82f949347.json \
  --reference-vocabulary committed-space=experiments/E2620/vocabulary-4a47f0281c81a9e5.json \
  --out experiments/E2632/refit-sensitivity.json
```

The output is the record's evidence: every fit's fingerprint and size, the per-rule compression, the four ratios, the
merge-stability tables and the committed files' figures, plus the corpus, split and evaluation identities it was
measured under.
