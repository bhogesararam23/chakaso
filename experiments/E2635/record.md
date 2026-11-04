# E2635 — does counting works instead of items change what E2634's separations meant?

**Status:** preregistered — written and committed before the analysis ran. **Type:** Research measurement over
committed evidence. It scores nothing, trains nothing and re-reads three probe files [E2634](../E2634/record.md)
already published.

## Question

E2634 reported **G-a with a floor qualification**: at 745 items, `context_pmi_permuted` separated both material arm
pairs under Holm's correction (p = 0.013718 and p = 0.003658), and no reading whose untrained floor sits at chance
separated anything. Its own limits section named the assumption underneath those two numbers — *"McNemar treats each
item as an independent paired observation; the items of one held-out book are not independent … a reader should treat
a p-value here as optimistic by an unknown margin"* — and reported the works behind each rejection only after
re-cutting the item set by hand (deviation 2, K-035).

The margin now has a name and a computation: `chakaso.training.paired_verdicts`. So the question is narrow and it is
about the instrument, not about the models:

**On the same 745 items, the same three arms and the same committed verdict rows, does a work-clustered randomization
reading separate the pairs that item-level McNemar separated?**

Three things would make this measurement worth running, and they are stated before it:

* if both separations survive, E2634's G-a is *strengthened* on the one axis it was weakest on — its p-values were
  conditional on an independence assumption nobody could check;
* if either fails, then the only reading in this series that ever separated two independently trained arms was an
  artifact of counting 745 passages as 745 observations, and the campaign's conclusion returns to E2629's **Z-d**;
* the authored minimal-pair probe has three clusters when its usable phenomena are read (three phenomena, 300 pairs),
  and a randomization test over three clusters has a floor of 0.25 — a claim that can be derived before the run and
  checked after it.

## The evidence, fixed by digest

| File | Item set | Digest recorded in the file | Models |
| --- | --- | --- | --- |
| [`experiments/E2634/context-with-outcomes.json`](../E2634/context-with-outcomes.json) | 745 same-region passages | `b48c01eb0e40a65ff064eb32fbb6027a01c5d478cb39a5dd73c65a0c2b5bbbc0` | `E2612-T`, `E2612-P`, `E2613-H`, `untrained-control` |
| [`experiments/E2634/continuations-with-outcomes.json`](../E2634/continuations-with-outcomes.json) | 745 v1 passages | `f5ae1effc9a97f27db8dcc82fc3618b9146598f3c64d56beae611511bb479e66` | the same four |
| [`experiments/E2634/minimal-pairs.json`](../E2634/minimal-pairs.json) | 800 authored pairs, 300 of them usable | pair set `2487a4ab4424ce2feb2e7b570a0a99ab099e2a40f8f16f9486686c2d99f07208` | the same four |

The two continuation files are K-032 extracts, so their per-item arrays are rebuilt from the compact blocks by the
analysis command itself. None of the three carries a cluster block — they were written before K-035 landed — so the
works come from re-cutting each item set from the pinned snapshot with the flags E2634 used, and **the rebuild is
refused unless its digest equals the digest above**. The criterion's arm reports are the committed
[`E2629`](../E2629/record.md) extracts; the arms, the vocabulary (`434debe82f949347`), the split
(`adf650435c3a9068`, seed 7) and the corpus (`312571f9fa8351a8`) are unchanged from E2634, so this is the same
comparison with a different reading of the same rows.

## Method, declared before the run

Per reading, per pair of the four models (six pairs), from `scripts/clustered_probe_verdicts.py`:

* **the clustered reading** — the paired difference of the two arms' verdict vectors, plus a **sign-flip
  randomization p-value** over the works. With 24–26 works the assignment count is 2 ** 23 or more, so it is resampled
  rather than enumerated: `--seed 7 --resamples 2000`, both recorded in the artifact. The bootstrap interval on the
  same difference is the 95 % percentile interval of 2,000 work-resamples with the same seed.
* **the item-level reading** — exact McNemar on the same two vectors, as E2629 and E2634 reported it.
* **both corrected the same way** — Holm's step-down across the six pairs of one reading at α = 0.05, so "separated"
  means one thing in both columns. E2634's published separations were over three pairs (α/(3−i)); the wider family
  here is *more* conservative for the arm pairs, and that is declared rather than tuned after the fact. To keep
  E2634's own comparison exact, the arm pairs' clustered p-values are additionally reported against E2634's three-pair
  steps (0.016667, 0.025, 0.05) in the record's table.
* **the design effect** and the **leave-one-work-out** table, read as: how many independent observations 745 items are
  worth, and which work the estimate rests on.
* The minimal-pair probe's cluster unit is the **phenomenon**, which the artifact names; a separation there would be a
  statement about templates, not about books.

`untrained-control` is in every family as the **positive control**: it is 0.119 to 0.500 away from the trained arms on
the same items, so if a clustered reading cannot separate *that*, the reading is broken rather than the arms alike.

## Predictions (none decision-bearing)

* **P1.** The rebuild reproduces both digests, so the works are attributed to the 745 items that were scored. E2634
  measured this once by hand and got `b48c01eb…`; the gate now refuses rather than informs.
* **P2.** The arm-versus-control pairs separate under **both** readings, every instrument, no exception.
* **P3.** `context_pmi_permuted`'s two separations survive clustering. E2634 measured that their discordant items
  spread over 24 and 26 works with no work above 12 %, which bounds the design effect: with 745 items over ~25 works,
  perfect within-work correlation would still leave an effective sample near the work count, and the observed spread
  says the correlation is far from perfect.
* **P4.** No reading gains a separation from clustering that it lacked at item level. Clustering here can only reduce
  the effective sample when a work's items move together; the case where it *sharpens* the test (negative within-work
  dependence) would be the surprise, and it is what **D-c** exists to name.
* **P5.** The minimal-pair instrument separates nothing under phenomenon clustering, at either 300 usable pairs (3
  clusters, floor p = 0.25 against Holm's 0.008333 first step) or all 800 (8 clusters, floor 0.0078). Its E2629
  inversion is not a clustering artifact and does not become one.

## Decision rules and outcomes, declared now

The verdict is a statement about the instrument and its p-values. **D-a…D-d are read over the two material arm pairs
(`E2612-T`/`E2612-P` and `E2612-T`/`E2613-H`) on the continuation readings**, with the inside-the-band pair
(`E2612-P`/`E2613-H`) and the control pairs kept as the checks they are.

* **D-a — the separations were not item-count artifacts.** Every arm-pair separation that item-level McNemar reported
  also separates under the clustered reading, and no pair separates clustered that did not separate item-level.
  Consequence: E2634's G-a stands with its margin now measured rather than assumed, and the probe's ceiling claim is
  unchanged.
* **D-b — at least one was.** A pair that separated at item level does not separate clustered. Consequence: the
  campaign's one positive instrument result was produced by treating passages of one book as independent evidence,
  and the record that reported it gets an addendum rather than an edit. This is the outcome E2633's leave-one-out
  range made plausible.
* **D-c — the clustering sharpened the test.** A pair separates clustered that did not separate at item level.
  Consequence: within-work dependence is negative on that reading, which nobody has claimed; the record reports the
  design effect below 1.0 and re-derives the claim before using it.
* **D-d — nothing separates anywhere, control pairs excepted.** Consequence: at 745 items over ~25 works there is no
  paired reading of these verdicts that can separate arms 0.108 bits per character apart, and the boundary E2626 drew
  is the last word for this corpus.

**No outcome here is a capability claim.** A separation under clustering says two arms answered these items
differently, counted by work. Whether either model is useful is K-001 and is not changed by any letter.

## Known limits, declared in advance

* One corpus, one split, one tokenizer, one arm triple — the same three arms E2629 and E2634 compared. Medicine is
  absent from the item set for the reason E2634 gave (one held-out work cannot supply independent passages).
* The randomization null is *within-work exchangeability of the arm labels*. It is exact for that null; it is not a
  test of "these two models differ in general", and with 24–26 works its granularity is coarse. The resampled p-value
  carries Monte Carlo error, and its seed and resample count are in the artifact.
* The verdict rows are the probes' own 1.0/0.5/0.0 readings, so a pass-versus-tie counts as half an item here and no
  item for McNemar. Where the two readings differ for that reason rather than clustering, the record says so.
* The cluster assignment comes from a deterministic re-cut, digest-verified, not from the scored run's own artifact.
  The next probe run will write the block itself (K-035), and a record that wants this reading then will not rebuild
  anything.
* Re-analysis of published rows adds no new evidence about the models. Its whole value is that it changes what the
  existing numbers mean.

## Cost

No training, no scoring, no GPU. Two item-set rebuilds (encode the 1,443 held-out documents with the pinned
vocabulary, then cut 745 items) and three analyses over already-committed vectors. E2634 measured the same rebuild
inside its 81-second scoring pass, so the declared estimate is **1–3 minutes per probe, single process**, peak memory
one loaded subword vocabulary plus torch's import floor (~250 MB) — nothing like the 2.5 GB a checkpoint costs. The
statistic itself is dependency-free and enumerates or resamples in well under a second.

## Reproduction

```bash
cd /d/chakaso
QPYTHON=.venv-nn/Scripts/python.exe   # torch only because the rebuild imports the runner that splits the side
SNAP=data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json
VOCAB=experiments/E2612/vocabulary-434debe82f949347.json
QUOTAS=(--registers prose-:400,science-:170,technology-:120,mathematics-:55 --max-per-work 20)
MODELS=(--model E2612-T --model E2612-P --model E2613-H --model untrained-control)

"$QPYTHON" scripts/clustered_probe_verdicts.py \
  --probe experiments/E2634/context-with-outcomes.json \
  --reading pmi_prior_span --reading pmi_next_span --reading pmi_permuted --reading pmi_reversed \
  --clusters-from rebuild --builder context --prefix-length 64 --continuation-length 24 \
  --snapshot "$SNAP" --vocabulary "$VOCAB" --split-seed 7 --expected-split-fingerprint adf650435c3a9068 \
  "${QUOTAS[@]}" "${MODELS[@]}" --seed 7 --resamples 2000 \
  --out experiments/E2635/context-clustered.json

"$QPYTHON" scripts/clustered_probe_verdicts.py \
  --probe experiments/E2634/continuations-with-outcomes.json \
  --reading pmi_easy --reading pmi_hard \
  --clusters-from rebuild --builder continuation --prefix-length 64 --continuation-length 24 --hard-gap 1024 \
  --snapshot "$SNAP" --vocabulary "$VOCAB" --split-seed 7 --expected-split-fingerprint adf650435c3a9068 \
  "${QUOTAS[@]}" "${MODELS[@]}" --seed 7 --resamples 2000 \
  --out experiments/E2635/continuations-clustered.json

"$QPYTHON" .venv/Scripts/python.exe scripts/clustered_probe_verdicts.py \
  --probe experiments/E2634/minimal-pairs.json --reading outcomes --pair-probe --clusters-from rows \
  --model E2612-T --model E2612-P --model E2613-H --model untrained-control \
  --seed 7 --resamples 2000 --out experiments/E2635/minimal-pairs-clustered.json
```

The first two refuse if the re-cut set's digest differs from the one the probe file records. The third needs no
snapshot: the authored probe's phenomenon is already in every row.
