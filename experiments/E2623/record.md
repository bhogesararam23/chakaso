# E2623 — a held-out continuation probe: do the trained decoders use the passage they are given, and does an untrained decoder?

**Status:** measured — **C-a: the probe is valid at this scale (trained decoders use their passage; untrained ones do not), and it is near ceiling, so it cannot rank the trained arms.** Preregistered and committed (`568b482`) before any checkpoint was scored; the Results section was added after.
**Type:** Measurement of **existing** checkpoints with a **new instrument**. Nothing is trained. It addresses the benchmark specification's coherence item
([`chakaso-capability-benchmarks.md`](../../docs/research/chakaso-capability-benchmarks.md) §2) with a likelihood-based reading that needs no generation, no judge and no authored text.

**Question.** Given a 64-identifier passage, does a model give the *true* next 24 identifiers a higher probability **because of the passage** than it gives a span of the same length from (easy) another work of
the same register, or (hard) the same document at least 1,024 identifiers away? "Because of the passage" is made exact by pointwise mutual information: the score of a continuation *c* is
log p(*c* | passage) − log p(*c*), the second term being the same model's probability of *c* straight after a document separator. A candidate's own predictability cancels; a model that ignores its passage gives every pair a
tie and scores exactly one half.

It measures one narrow thing — use of preceding context to prefer the true continuation — on held-out text of one register family. It is not a measure of knowledge, reasoning or usefulness (K-001).

## The instrument, fixed now

[`scripts/evaluate_continuations.py`](../../scripts/evaluate_continuations.py) over `chakaso.training.continuation_items` / `nn_evaluation.continuation_log_probabilities`. Items are cut from the **test side of the whole-work split** of the
pinned snapshot (`gutenberg-multiregister-v1`, fingerprint `312571f9fa8351a8`; split seed 7, validation and test fractions 0.15, split fingerprint `adf650435c3a9068`, checked by the script), tokenized with the committed vocabulary
`434debe82f949347`. Passage 64 identifiers, continuation 24, hard gap 1,024. Selection is by SHA-256 order; one item per document, at most **5 per work**.

| Register | Items | Held-out works | Basis for the count |
| --- | ---: | ---: | --- |
| prose | 100 | 35 | |
| science | 55 | 12 | 12 works × 5 = 60 would exceed the eligible passages (59) |
| technology | 40 | 9 | 9 × 5 = 45 |
| mathematics | 20 | 5 | 5 × 5 = 25 |
| medicine | — | 1 | **not scored**: one held-out work is not a test of generalisation |
| **all** | **215** | | item-set digest `9557a327bfe61e2d747a826777b748ebe285b2dd352ba6a769aca0262cfacdcd` (built and checked before this record; the script reports it again) |

## Models, fixed now

`E2617`, `E2615-s00`, `-s25`, `-s50`, `-s75`, `E2616` (all seed 7, the committed vocabulary) and **seven untrained initialisations** of `E2617`'s architecture (seeds 7, 11, 13, 17, 19, 23, 29) as the floor — the floor is a distribution here from the start, which
E2622 showed to be necessary. Checkpoints that appear later (E2617b, the E2618 and E2619 arms) are added when scored, never dropped.

## Decision rules, declared before scoring

The decision-bearing quantity is the **PMI accuracy against the easy distractor** (the weaker the cue, the harder to pass by topic alone is the hard distractor; it is reported with the same rules but does not decide).

* **C-a — valid at this scale.** `E2617`'s Wilson interval excludes 0.5; **and** the mean PMI-easy accuracy of the seven untrained models lies in [0.45, 0.55]; **and** the highest untrained model's accuracy lies below `E2617`'s lower interval bound.
* **C-b — the trained model does not show use of its passage.** `E2617`'s interval includes 0.5. The probe cannot see context use at this scale.
* **C-c — the floor is not at chance.** The seven-initialisation mean lies outside [0.45, 0.55]. The cancellation does not hold for this construction; the headline is not read and the cause is looked for before a further version.
* **C-d — anything else**, reported with the numbers.

**Reported, not decision-bearing:** the hard-distractor PMI accuracy; the raw-conditional accuracy (log p(*c* | passage) alone) and the prefix-blind accuracy (log p(*c*) alone) for every model — they show how much of a plain likelihood comparison is
the candidate's own predictability; all of it per register.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** `E2617` scores at least 0.75 on PMI-easy.
* **P2.** Every trained model scores lower on PMI-hard than on PMI-easy (a same-document distractor shares topic and style, so it is the harder discrimination).
* **P3.** The prefix-blind accuracy of the untrained models is far from 0.5 for some initialisations (an untrained model prefers some token sequences regardless of context), which is why the raw-conditional reading would not do.
* **P4.** `E2616` (1,200 updates) scores below the 2,400-update arms on PMI-easy.

**What may not change after scoring:** the item set (digest above), the scoring rule, the models named, the outcomes and the predictions. A change is a new record.

## Known limits, declared in advance

* Up to five items come from one work, so the items are not independent and the Wilson intervals (which treat them as independent) are narrower than they should be; no work-clustered interval is computed here.
* Continuations of 24 identifiers (roughly 50 characters) after 64 (roughly 130): the probe sees short-range use of context. Nothing is said about long-range coherence.
* The distractors are real text of the same register; a model could prefer the true continuation by recognising a *style* shift within a document (hard) or *topic* shift (easy) without anything that deserves the name coherence. PMI removes the candidate's own predictability, not the passage's cues.
* Archaic English, one register family, one seed per model, a corpus the models trained on (the **test side** is held out by whole work, but the registers' vocabularies overlap with training).
* Preference, not generation: a model can pass and still generate nothing usable.

## Cost

Seconds of CPU per model (batched). No training.

## Reproduction

```bash
python scripts/evaluate_continuations.py \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --vocabulary experiments/E2612/vocabulary-434debe82f949347.json --expected-split-fingerprint adf650435c3a9068 \
  --registers prose-:100,science-:55,technology-:40,mathematics-:20 --max-per-work 5 \
  --checkpoint E2617=data/experiments/E2617/seed-7/E2617-decoder.pt \
  --checkpoint E2615-s00=… --checkpoint E2615-s25=… --checkpoint E2615-s50=… --checkpoint E2615-s75=… --checkpoint E2616=… \
  --control --control-seeds 11,13,17,19,23,29 --out experiments/E2623/results.json
```

## Results (added 2026-11-02)

[`results.json`](results.json), from the command in *Reproduction*. Item set: 215 items, digest `9557a327…cfacdcd` as declared. Scoring took 7 min 12 s for thirteen models on a host busy with a training run. Checkpoint checksums are those of E2621.

| Model | PMI accuracy, easy distractor [95 % Wilson] | PMI, hard distractor | raw conditional (easy) | prefix-blind (easy) |
| --- | ---: | ---: | ---: | ---: |
| E2617 | 0.986 [0.960, 0.995] | 0.953 [0.917, 0.975] | 0.795 | 0.498 |
| E2615-s00 | 0.972 [0.940, 0.987] | 0.949 [0.911, 0.971] | 0.795 | 0.479 |
| E2615-s25 | 0.958 [0.922, 0.978] | 0.921 [0.877, 0.950] | 0.795 | 0.507 |
| E2615-s50 | 0.977 [0.947, 0.990] | 0.953 [0.917, 0.975] | 0.786 | 0.479 |
| E2615-s75 | 0.977 [0.947, 0.990] | 0.916 [0.872, 0.946] | 0.795 | 0.507 |
| E2616 | 0.972 [0.940, 0.987] | 0.926 [0.883, 0.954] | 0.702 | 0.484 |
| untrained, mean of 7 (range) | 0.484 (0.451–0.516) | 0.469 (0.423–0.498) | 0.469 | 0.484 (0.428–0.540) |

By register, PMI easy / hard (items in brackets; the first four columns are trained models, the last the mean of seven untrained):

| Register | E2617 | E2615-s00 | E2615-s75 | E2616 | untrained mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose (100) | 1.00 / 0.96 | 0.98 / 0.97 | 0.99 / 0.94 | 0.99 / 0.96 | 0.47 / 0.46 |
| science (55) | 0.96 / 0.95 | 0.98 / 0.96 | 0.96 / 0.93 | 0.96 / 0.95 | 0.48 / 0.49 |
| technology (40) | 0.97 / 0.97 | 0.95 / 0.95 | 0.97 / 0.90 | 0.97 / 0.90 | 0.50 / 0.47 |
| mathematics (20) | 1.00 / 0.90 | 0.95 / 0.80 | 0.95 / 0.80 | 0.90 / 0.75 | 0.51 / 0.45 |

No pair was a tie for any model.

### The preregistered reading

* **Outcome: C-a.** E2617's interval [0.960, 0.995] excludes 0.5; the mean of the seven untrained models' PMI-easy accuracies is **0.484**, inside [0.45, 0.55]; the highest untrained model (0.516) lies far below E2617's lower bound (0.960). The pointwise-mutual-information construction did what it was built for:
  untrained models sit at chance whatever their per-candidate biases are.
* **P1 held** (0.986 ≥ 0.75). **P2 held**: every trained model scores lower on the same-document distractor than on the other-work one (E2617 0.953 against 0.986; E2615-s75 0.916 against 0.977). **P3 did not hold as worded**: the untrained models' prefix-blind accuracy ranges 0.428–0.540, not "far from 0.5" — a random
  initialisation's own preferences between these particular spans are small, though the raw-conditional reading is still not usable for ranking (see next). **P4 did not hold on PMI** (E2616 0.972 is not below every 2,400-update arm; E2615-s25 is lower, 0.958); it holds on the *raw-conditional* column (0.702 against 0.786–0.795).

### What the three readings show

* **Why PMI.** For the trained models the raw-conditional accuracy (0.70–0.80) is far below their PMI accuracy (0.96–0.99): a quarter of the time the true continuation is *less likely in absolute terms* than a distractor that is simply more predictable text, and the passage is what puts the
  true one ahead. A plain likelihood comparison would have understated context use by 0.2 and mixed it with the candidates' own predictability.
* **The probe is near ceiling.** Six trained models lie between 0.958 and 0.986 on the easy distractor (all intervals overlap) and 0.916–0.953 on the hard one. **It cannot order the trained arms**; the differences between 0.972 and 0.986 on 215 items are well inside the interval. The one place with headroom is
  mathematics, hard distractor: E2617 0.90 against 0.80 for E2615-s00, E2615-s75 and 0.75 for E2616 — on 20 items, intervals of about ±0.15, not a finding.
* **What it does show.** Trained decoders at this scale use a 64-identifier passage to prefer the true next 24 identifiers over a same-register span from another work (0.96–0.99) and over a same-document span far away (0.92–0.95); untrained ones do not (0.45–0.52). That is a statement about **use of context**, which held-out bits per character
  cannot make by itself, and it is the first non-loss capability measurement here that passes its own floor test at the headline level.

### Limits that stand, and the next version

Items within a work are not independent and the intervals are narrower than a work-clustered interval would be. The task is short-range and a topic or style cue suffices for the easy distractor. A harder version — distractors from the *same passage region* (the preceding or following span, or the true span with its words
permuted within the window) so topic and style are held fixed — is the way to make the probe rank models; it is a different construction and a new record, and is not started here. Nothing here is knowledge, reasoning, instruction following or usefulness (K-001).
