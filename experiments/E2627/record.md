# E2627 — the same-region continuation probe: is its floor at chance on real text, and does it have anywhere to go?

**Status:** preregistered — this record and the item set it names are committed **before** any checkpoint is scored on
version 2 of the continuation probe. The Results section will be added after, and nothing declared here will be changed.
**Type:** Measurement of existing and soon-to-exist checkpoints with a **new instrument**. Nothing new is trained by this
record: it scores rungs [E2626](../E2626/record.md) already writes and artifacts already on disk.
**Prerequisite:** the construction and its runner — [`chakaso/training/context_continuations.py`](../../src/chakaso/training/context_continuations.py),
[`scripts/evaluate_context_continuations.py`](../../scripts/evaluate_context_continuations.py), 23 tests — landed before this
record, as code and tests only. No model has been scored on it.

## Why this instrument

[E2623](../E2623/record.md) built the continuation probe and it came back **valid and near ceiling**: trained decoders prefer
the true next 24 identifiers to a span from another work of the same register 0.958–0.986 of the time, all six models' intervals
overlap, and the record's own conclusion is "it cannot rank the trained arms". Its limits section names the reason and the fix:

> A harder version — distractors from the *same passage region* (the preceding or following span, or the true span with its words
> permuted within the window) so topic and style are held fixed — is the way to make the probe rank models; it is a different
> construction and a new record, and is not started here.

This is that record. Four distractors per item, each exactly the continuation's length: the document's own text immediately
**before** the passage and immediately **after** the continuation, and the truth's own identifiers **permuted** and **reversed**.
The first two hold topic, register and hand fixed so nothing but discourse position remains; the last two hold the exact multiset
of identifiers fixed so only sequence remains. The pointwise-mutual-information subtraction is carried over unchanged and matters
more here, because a permuted span is far less predictable English than the text it was made from: without the subtraction a
reading could reward a model that merely dislikes disordered text.

## The item set, fixed now

[`scripts/evaluate_context_continuations.py`](../../scripts/evaluate_context_continuations.py) over
`chakaso.training.context_continuations`, on the **test side of the whole-work split** of the pinned
`gutenberg-multiregister-v1` snapshot (fingerprint `312571f9fa8351a8`, split seed 7, validation and test fractions 0.15, split
fingerprint `adf650435c3a9068` — checked by the script, and the split rebuild is imported from the v1 runner rather than
duplicated, so the two probes cannot sample different documents while their records say they are comparable). Items are cut with
the committed vocabulary `434debe82f949347`, passage 64 identifiers, continuation 24, at most 5 per work.

| Register | Items | Basis |
| --- | ---: | --- |
| prose | 100 | the same cap E2623 reached |
| science | 55 | as E2623 |
| technology | 40 | as E2623 |
| medicine | — | **not scored**: E2623 excluded it because one held-out work is not a test of generalisation |
| mathematics | 20 | as E2623 |
| **all** | **215** | item-set digest `f49c128e7b302827c4c5d790b1f685ed94dbfa02971fda17ca1c687e73e5bcc1` |

The counts and the digest were produced by building the set once, before this record was written, and no model was scored to
obtain them. The set comes to **exactly 215 items with the same per-register counts as v1**, which is not a coincidence the way
it sounds: the same anchor rule, the same document order, the same one-per-document and five-per-work caps, and every register
reached its cap. Where v1 needed a span 1,024 identifiers away, this needs the document to hold the span before the passage and
the span after the continuation; both requirements are satisfied at every anchor v1 used here, so a difference between the two
probes' scores is a difference in the task, not in the text sampled.

## Models, fixed now

**The floor**: seven untrained initialisations of the endpoint architecture, seeds 7, 11, 13, 17, 19, 23, 29 — E2622's and
E2623's floor, kept identical so the three probes' floors are comparable.

**The trained set**, all on the committed vocabulary:
* [E2626](../E2626/record.md)'s twelve rungs (steps 200 … 2400 of E2614 seed 7's configuration) — the ordering criterion, which
  is what makes a ranking claim here checkable rather than asserted;
* [E2614](../E2614/record.md)'s four further seeds (11, 13, 17, 19) — five independent trained arms of one configuration, whose
  held-out losses differ by up to 0.0067 bits per character;
* [E2612](../E2612/record.md)'s two arms (T the mixture, P the prose-only) and [E2613](../E2613/record.md)'s half-prose arm H, and
  [E2615](../E2615/record.md)'s 25 % arm — arms composed differently, whose loss ordering this project has already measured.

The E2608/E2611 artifacts are excluded by construction: they carry the `f99386c7…` vocabulary of a different corpus, and the
runner refuses a checkpoint whose tokenizer is not the one the items were cut with.

## Decision rules, declared before any model is scored

Each of the four kinds is judged **separately**. Nothing pools them, and no kind may rescue another; a construction with one
broken distractor is a construction with one broken distractor.

* **Floor (per kind).** The mean of the seven untrained accuracies lies in **[0.45, 0.55]** and the *highest* untrained accuracy
  lies below the endpoint model's lower Wilson bound. E2623's C-a, unchanged, so the two readings are comparable.
* **Headroom (per kind).** The endpoint model's accuracy is **below 0.95**. If it is at or above that, the kind is saturated in
  exactly the way E2623's easy distractor was, and no ranking claim about that kind is possible whatever the correlation says.
* **Ranking (per kind).** With the criterion fixed now as **E2626's rung test-pack bits per character** and materiality fixed
  as **0.02**, the same rule as E2626: McNemar's exact paired test over every rung pair, Holm-corrected within the kind, and
  :func:`chakaso.training.probe_sensitivity.compare_instruments`' verdict — **S-a**, **S-b**, **S-c**, **S-d** — read with its
  ρ, its coverage of the material pairs, and its smallest separated gap.
* **The record's own outcome**, stated for the construction as a whole rather than for any kind:
  * **T-a — the construction does what it was built for.** At least one kind clears floor, headroom and **S-a** or **S-b**.
    That kind becomes the repository's rankable context instrument, and the sharpest open question in
    [`chakaso-research-campaign-2026-11.md`](../../docs/research/chakaso-research-campaign-2026-11.md) §7 gets a measured answer
    for it. Consequence, and only then: it is declared usable for the *next* arm comparison, not retro-fitted onto the past ones.
  * **T-b — valid, still saturated.** Every kind clears its floor and none has headroom. E2623's ceiling was not the distractor's
    topic cue; it is something about a 24-identifier span after a 64-identifier passage at this scale, and the record says so.
  * **T-c — the floor is not at chance.** Some kind's untrained mean lies outside [0.45, 0.55]. The construction carries a bias
    the subtraction does not cancel; that kind's numbers are tabulated and not read, and the bias is looked for before a version
    3 is proposed.
  * **T-d — anything else**, with the numbers.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** `permuted` and `reversed` clear the floor and have headroom, because a model that has read 20M identifiers of prose
  should prefer ordered text it has just been primed with — but the subtraction removes the "ordered text is more likely" part,
  so what is left to measure is whether the *passage* predicts *this* order.
* **P2.** `prior_span` is the hardest kind: it is contiguous prose of the same document in the same hand, differing from the truth
  only in being the wrong end of it.
* **P3.** No kind separates a pair of E2614's five seeds: their held-out losses differ by under the band and E2622 measured the
  probe's own seed spread at 0.015–0.048 on 300 pairs.
* **P4.** The endpoint rung reproduces, to the last digit, the numbers **E2626** recorded for the same bytes on the *v1*
  items — the deterministic scorer over the same committed item files is what makes that a check rather than a hope.
  E2623's figures for `E2617` are **not** the reference: E2617 is the Linux host's repeat of seed 7, whose metrics agree
  with E2614's seed 7 to within 0.0025 bits per character but whose weights are not the same tensors. E2626 measured the
  consequence on this very instrument — all-800 minimal-pair accuracy 0.8250 for the endpoint here against 0.8350 for
  E2617 in E2622 — a gap inside E2622's own seed spread for the probe. A cross-host figure is a comparison, not a
  reproduction.

  *Amended 2026-11-03, before any model was scored on this construction.* As first committed this prediction said the
  endpoint must reproduce E2623's E2617 numbers "to the last digit (same weights, same items, deterministic scorer)".
  The parenthetical was wrong: the weights are not the same. The amendment changes which artifact is the reference and
  records the measured cross-host difference; it does not touch the item set and its digest, the four kinds, the floor
  models, the criterion, the materiality or the outcomes, and it is written now rather than after a result because a
  prediction adjusted to fit data has stopped being a prediction.

**What may not change after scoring:** the item set and its digest, the four kinds, the floor models, the criterion and
materiality, the outcomes and the predictions. A change is a further record.

## Known limits, declared in advance

* Items within a work are not independent (up to five per work) and the Wilson intervals treat them as independent — the same
  limit E2623 carried, unchanged.
* The `permuted` and `reversed` kinds test sequence, not discourse. A model can pass them by learning local transition statistics
  and that tells nothing about coherence; they are reported as what they are, and the ranking claim this record cares about is the
  one made by the two adjacent-span kinds.
* The truth is always the same span the model was trained to predict at that position; nothing here can distinguish "knows this
  passage" from "knows the register", because every distractor is from the same register.
* Archaic English, one corpus family, a truncated budget, a test pack the models' own split held out by whole work.
* Preference, not generation: a model can pass every kind and still write nothing usable (E2610), and none of this touches
  K-001.

## Cost

Scoring is seconds to minutes per model: 13 requests per item (5 candidates × 2 prefixes, deduplicated) over 215 items. E2623
measured 7 min 12 s for thirteen models batched; this has nineteen trained models plus seven controls, so about **15–20 minutes**
of CPU, after [E2626](../E2626/record.md)'s arm finishes.

## Reproduction

```bash
python scripts/evaluate_context_continuations.py \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --vocabulary experiments/E2612/vocabulary-434debe82f949347.json --expected-split-fingerprint adf650435c3a9068 \
  --registers prose-:100,science-:55,technology-:40,mathematics-:20 --max-per-work 5 \
  --control --control-seeds 11,13,17,19,23,29 \
  --checkpoint step200=data/experiments/E2626/seed-7/E2626-step200-decoder.pt … --checkpoint step2400=… \
  --checkpoint E2614-s11=… --checkpoint E2614-s13=… --checkpoint E2614-s17=… --checkpoint E2614-s19=… \
  --checkpoint E2612-T=… --checkpoint E2612-P=… --checkpoint E2613-H=… --checkpoint E2615-s25=… \
  --out experiments/E2627/context.json
python scripts/analyse_probe_sensitivity.py --report data/experiments/E2626/seed-7/report.json \
  --protocol configs/experiments/E2626-curve-ladder.toml --work-dir data/experiments/E2626/seed-7 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --split-seed 7 --expected-corpus-fingerprint 312571f9fa8351a8 \
  --pairs experiments/E2626/minimal-pairs.json --continuations experiments/E2626/continuations.json \
  --context experiments/E2627/context.json \
  --usable-phenomena auxiliary_agreement,determiner_noun_agreement,reflexive_pronoun_agreement \
  --materiality 0.02 --out experiments/E2627/results.json
```

The analysis is run with E2626's own pair and continuation files beside the new one, because the criterion table
comes from the curve and the command takes all three instruments at once: E2626's verdicts must reproduce exactly in this
output too, which is a free consistency check on the analysis rather than a second claim. The twelve further arms appear in
`context.json` and are named in the result's `models_scored_but_not_analysed` list — they carry no rung loss number here, so
they are not part of the ranking decision and are not silently dropped from the record either.

## Results

_Not yet. No model has been scored on this instrument at the time this record was committed._
