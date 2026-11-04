# E2635 — does counting works instead of items change what E2634's separations meant?

**Status:** measured — **D-b: one of E2634's two separations does not survive the works behind its own items, and the
other does.** Preregistered and committed on 2026-11-04 before the analysis ran; the Results section was added after.
**Type:** Research measurement over committed evidence. It scores nothing, trains nothing and re-reads three probe files
[E2634](../E2634/record.md) already published.

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

The seed check re-runs the same analysis for one reading with a different draw:

```bash
"$QPYTHON" scripts/clustered_probe_verdicts.py \
  --probe experiments/E2634/context-with-outcomes.json --reading pmi_permuted \
  --clusters-from rebuild --builder context --prefix-length 64 --continuation-length 24 \
  --snapshot "$SNAP" --vocabulary "$VOCAB" --split-seed 7 --expected-split-fingerprint adf650435c3a9068 \
  "${QUOTAS[@]}" --model E2612-T --model E2612-P --model E2613-H \
  --seed 11 --resamples 20000 --out experiments/E2635/context-permuted-seed11.json
```

with `--seed 13` for the second file. Three models, so the family is the three arm pairs — E2634's own — and the
clustered p-values are read against its steps.

## Results (added 2026-11-04)

**Outcome: D-b — one of E2634's two separations was an artifact of counting passages as independent.** On the reading
that separated anything, `context_pmi_permuted`, the pair `E2612-T` / `E2612-P` moves from p = 0.013718 under item-level
McNemar to **p = 0.030985** under the work-clustered randomization test and stops being a separation; the pair
`E2612-T` / `E2613-H` moves from 0.003658 to **0.008996** and stays one. No other continuation reading separates any
arm pair under either reading, no pair gains a separation from clustering, and every direction stays as E2634 left it.

The verdict does not turn on the choice of family: against E2634's own three-pair Holm steps (0.016667, 0.025, 0.05)
the surviving 0.008996 clears the first step and the lost 0.030985 clears neither. It does not turn on the seed either: the same reading re-cut at seeds 11 and 13 with 20,000 draws each
([`context-permuted-seed11.json`](context-permuted-seed11.json),
[`context-permuted-seed13.json`](context-permuted-seed13.json)) gives 0.0331 and 0.0320 for the lost pair, 0.0102 and
0.0109 for the surviving one — the p-values move in the third decimal, the verdicts do not move.

### Every arm pair, both readings, same rows

`items 745`, `clusters 61`, α = 0.05 over six pairs per reading. `disc` is McNemar's discordant split, `eff` the design
effect, `works` how many works hold the disagreement, and the interval is the 95 % percentile interval of 2,000
work-resamples. Full rows in [`context-clustered.json`](context-clustered.json),
[`continuations-clustered.json`](continuations-clustered.json) and
[`minimal-pairs-clustered.json`](minimal-pairs-clustered.json).

| Reading | Pair | accuracy | difference | disc | p item-level | p clustered | eff | works | 95 % interval |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prior_span` | T / P | 0.9409 / 0.9208 | +0.02013 | 29:14 | 0.031539 | 0.073963 | 1.28 | 29 | [0.0012, 0.0391] |
| `next_span` | T / P | 0.9597 / 0.9463 | +0.01342 | 25:15 | 0.15386 | 0.12694 | 0.83 | 29 | [−0.0014, 0.0279] |
| **`permuted`** | **T / P** | **0.9664 / 0.9436** | **+0.02282** | **30:13** | **0.013718 sep** | **0.030985 —** | **1.41** | **24** | **[0.0050, 0.0441]** |
| **`permuted`** | **T / H** | **0.9664 / 0.9396** | **+0.02685** | **32:12** | **0.003658 sep** | **0.008996 sep** | **1.16** | **26** | **[0.0088, 0.0456]** |
| `reversed` | T / P | 0.9597 / 0.9463 | +0.01342 | 25:15 | 0.15386 | 0.26387 | 1.64 | 25 | [−0.0075, 0.0356] |
| `prior_span` | P / H | 0.9208 / 0.9248 | −0.00403 | 17:20 | 0.74283 | 0.75712 | 1.04 | 27 | [−0.0209, 0.0121] |
| `permuted` | P / H | 0.9436 / 0.9396 | +0.00403 | 23:20 | 0.76079 | 0.76312 | 0.89 | 25 | [−0.0122, 0.0199] |
| v1 `easy` | T / P | 0.9879 / 0.9758 | +0.01208 | 14:5 | 0.063568 | 0.10495 | 1.17 | 15 | [0.0, 0.0249] |
| v1 `easy` | T / H | 0.9879 / 0.9732 | +0.01477 | 15:4 | 0.019211 | 0.037981 | 1.03 | 15 | [0.0036, 0.0262] |
| v1 `hard` | T / P | 0.9463 / 0.9342 | +0.01208 | 28:19 | 0.24296 | 0.14843 | 0.64 | 33 | [−0.0014, 0.0270] |

The **positive control behaved as a control must**: every arm against `untrained-control` separates under both readings
in all six continuation readings, with 372–448 discordant items spread over 58–60 works and clustered p-values at the
resampling floor (deviation 3). The authored pair probe is the opposite case, and it is the finding of this record:

| Instrument | items | clusters | separated item-level | separated clustered | design effect, arm vs control |
| --- | --- | --- | --- | --- | --- |
| same-region, four kinds | 745 | 61 works | 3 of 6 pairs (`permuted`: 5) | 3 of 6 (`permuted`: 4) | 0.85 – 1.41 |
| v1 continuation, two kinds | 745 | 61 works | 3 of 3 controls | 3 of 3 controls | 0.83 – 1.45 |
| authored minimal pairs | 800 | **8 phenomena** | 3 of 3 controls (p ≈ 2.8 × 10⁻³⁵) | **none** | **16.08 – 16.80** |

Eight hundred hand-written pairs of eight phenomena are worth about **49 independent observations** under the cluster
unit the artifact declares. The trained-versus-random contrast — the largest effect any instrument in this repository
has ever measured, item-level p ≈ 10⁻³⁵ — falls to p = 0.0234 with eight clusters, against Holm's first step of
0.008333, because a randomization test over eight units cannot produce a smaller p than 2/2⁸ = 0.0078. **A clustered
reading that reports nothing must therefore be read with its cluster count next to it**: on the corpus probes the count
is 61 and the floor is 2/2⁶¹, so the silence is the data's; on the authored probe the count is 8 and the silence is the
design's.

### The clustering the record promised, cross-checked

E2634 rebuilt its item set by hand and published that the two separations' discordant items spread over **24 and 26**
works with `mathematics-pg302` carrying **5 of 43 (11.6 %)**. This record's rebuild — a different code path, gated on
the digest rather than on the author's reading of the flags — returns 24 and 26 works with the same largest work at the
same 0.116 share. The hand figure was right, and the digest gate is what makes the new one reproducible: the re-cut
set digested to `b48c01eb0e40a65f…` and `f5ae1effc9a97f27…`, exactly the digests the probe files record, so the clusters
are paired to the 745 passages that were scored. The cluster listings themselves now have identities too —
`1b2fa9e975c2…` (same-region) and `dc64cdffc7aa…` (v1) — which is what a future record needs instead of a rebuild.

### The predictions

* **P1 — held.** Both rebuilds reproduced the probe files' digests; the gate had nothing to refuse.
* **P2 — held.** All 18 arm-versus-control continuation pairs separate under both readings.
* **P3 — held, for one of the two.** `permuted` kept `T`/`H` and lost `T`/`P`. The predicted mechanism was right in
  kind and wrong in size: the discordance is spread over 24–26 of 61 works, the design effect is 1.16–1.41, and the
  p-value moved by a factor of ~2.3 — not enough to keep both separations.
* **P4 — held as observed, wrong as stated.** No pair gained a separation from clustering. But the reason given — that
  clustering can only shrink the effective sample — is contradicted by **seven of the eighteen** continuation arm-pair
  readings, whose design effect is **below 1.0** (0.64 to 0.99). Deviation 2.
* **P5 — held, and by arithmetic rather than by measurement.** With 8 phenomenon clusters the floor is 0.0078 against
  a first Holm step of 0.008333, so the authored probe could separate nothing it had; at three usable phenomena the
  floor is 0.25 and separation is impossible for any verdict pattern. Derived before the run, confirmed by it.

### Deviations

1. **The preregistration used the wrong denominator for the works.** P3 reads "745 items over ~25 works"; the item set
   spans **61** held-out works, and the ~25 was E2634's count of works carrying the *discordant* items (and, by
   coincidence of wording, the number of works in the whole-pack loss sample per K-033). The measured design effects
   are the ones P3's argument needs and it survives; the stated count did not.
2. **Negative within-work dependence is real here.** Design effects below 1.0 mean the paired differences of one work
   *disagree with each other* more than chance would arrange, so clustering sharpens the estimate rather than blunting
   it. It did not produce a single new separation in this run, but any future reading must expect both directions and
   the module does not assume otherwise.
3. **The control pairs' clustered p-values are floors, not measurements.** With 61 clusters the assignment count is
   2⁶⁰, so the test is resampled and no draw was as extreme as the observed one: the printed 0.00049975 is
   1/(2,000 + 1), the smallest value the resampled path can report. The honest statement is p ≤ 0.0005.
4. **The authored pair probe loses even its unambiguous separation.** This was not among the four outcomes, because
   D-a…D-d were scoped to the continuation readings and the pair probe was kept as E2634's non-scaling control. It is
   recorded here as the clearest single number in the run: 800 pairs, 8 clusters, ~49 effective observations,
   nothing separated, including a trained network against a random one.
5. **Cost.** Declared 1–3 minutes per rebuild; measured **19 s** (same-region) and **29 s** (v1), the authored probe
   needing no rebuild at all (**0.2 s**). The three robustness re-runs at 20,000 resamples cost ~40 s each. Peak
   memory was one Python process holding the 149 MB snapshot's held-out side and a loaded vocabulary; no checkpoint
   was loaded and no model was scored.

## Interpretation

E2634's G-a was already qualified — the one reading that separated anything has an untrained floor 0.119 below chance.
This record qualifies the *other half* of that result, the half that was about statistics rather than about the probe:
of its two separations, one does not survive the works behind its own items. What is now believed about the instrument
is narrower than what E2634 published and stronger than what E2629 did: **at 745 passages from 61 held-out works, one
continuation reading separates one arm pair under a test that counts works, and it is the reading this project already
refused to use.** The remaining five readings separate nothing under either reading, and the authored probe separates
nothing once its own template structure is admitted.

The two uncertainty readings disagree on the lost pair, and that is a property rather than a defect: the bootstrap
interval for `T`/`P` on `permuted` is [0.0050, 0.0441] — clear of zero — while the randomization test says 0.031.
Resampling works answers "where is the difference, given that these are the works"; flipping a work's arm labels answers
"could the works alone have produced this". A percentile interval is not the inverse of a randomization test and this
record prints both rather than choosing the convenient one.

Nothing here is about whether either model is good. A clustered separation is a statement about items, works and
label swaps; the models' usefulness is K-001 and this run did not touch it.

## Conclusion

**Believed, at this strength:** the capability probes' arm-pair verdicts were optimistic, by a measurable but modest
amount on the corpus probes (design effects 0.64–1.64 over 61 works) and by a large one on the authored probe (16× over
8 phenomena). E2634's headline separation is now known to be one of two, and the surviving one belongs to the reading
with the off-chance floor, so the series' position is unchanged in kind: **no probe reading this repository trusts
separates two independently trained arms**, whether items are counted as items or as works. What changed is the
standard of evidence — a separation claim now requires the clustered reading, and `scripts/clustered_probe_verdicts.py`
is the command that produces it from committed rows.

## Follow-up

1. Any future probe record states both readings and declares which one its outcome letters are read on; a record that
   declares neither has no separation claim to make.
2. The next probe run writes its own cluster block (K-035), so this rebuild path becomes the compatibility path for
   E2626–E2634's files rather than the normal one.
3. The authored pair probe is not a candidate arm instrument at any item count this corpus and this template set
   support: eight clusters bound the test before the data is consulted. Reading it as an instrument needs pairs written
   to independent templates, which is an authoring task, not a scoring one.
