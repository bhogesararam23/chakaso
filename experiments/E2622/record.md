# E2622 — the minimal-pair probe, version 2: with identifier counts balanced, is the untrained floor at chance, and does the probe then see what the trained decoders learned?

**Status:** measured — **I-a holds at the headline level, but the headline floor is not stable across random initialisations; the readable result is the three usable phenomena.** Preregistered and committed (`5ed623a`) before any checkpoint was scored on version 2; the Results section was added after.
**Type:** Measurement of **existing** checkpoints with a **corrected instrument**. Nothing is trained. It is the re-measurement [E2621](../E2621/record.md) named as its consequence: E2621's
untrained floor sat at 0.606 (interval [0.572, 0.640]), above chance, because in some constructions the bad sentence tokenizes into more identifiers, and a summed log-probability falls with each one.

**What changed, and only that.** [`configs/probes/grammatical-minimal-pairs-v2.json`](../../configs/probes/grammatical-minimal-pairs-v2.json), digest
`2487a4ab4424ce2feb2e7b570a0a99ab099e2a40f8f16f9486686c2d99f07208`: the same eight phenomena, lexicon, templates, scoring rule and 100 pairs each, but only frames whose two sentences have the **same identifier count under the
committed vocabulary** `434debe82f949347` are eligible (`scripts/build_minimal_pairs.py --balance-with`). The version-1 file and its numbers are unchanged. The scorer now also reports the **token-count-only reading** with each model's
own tokenizer; for these pairs under this vocabulary it is exactly 0.5 by construction (a test shows it).

It measures one narrow thing: preference for a grammatical sentence over a minimally different ungrammatical one, on constructions common in 19th-century prose. It is not a measure of understanding, knowledge,
reasoning or usefulness (K-001).

## Models, fixed now

The seven scored in E2621 — `E2617`, `E2615-s00`, `-s25`, `-s50`, `-s75`, `E2616` and an untrained control (a random initialisation of `E2617`'s architecture, seed 7) — plus every further checkpoint the campaign has
produced **at the time of scoring**, named in the results: `E2617b` (the same-host repeat of seed 7) and the new seeds of `E2615-s00/-s25/-s50` and the three `E2619` arms, each scored once, with its seed. Scoring a checkpoint
that did not exist when this record was written is allowed because it adds models, not conditions; no model is dropped once scored.

## Decision rules, declared before scoring

Identical in form to E2621's, with version 2:

* **I-a — valid at this scale.** `E2617`'s Wilson interval **excludes 0.5**; the **untrained control's interval includes 0.5**; and the length-only reading (0.5 in characters at version 1; reported again here) lies **below** `E2617`'s interval.
* **I-b — the trained model is at chance.** `E2617`'s interval includes 0.5.
* **I-c — the floor is still not at chance.** The control's interval excludes 0.5. The per-phenomenon controls name the phenomena that carry it; the headline is not read, and a further version is a further record.
* **I-d — anything else**, reported with the numbers.

**Per-phenomenon usability** (as E2621): a phenomenon is *usable* if the control's interval for it includes 0.5 and `E2617`'s excludes it; a phenomenon that fails either test is excluded from any per-phenomenon reading, and its
numbers are still tabulated.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** The control's overall accuracy is within 0.05 of 0.5 (the balance removes the token-count bias; other biases, such as a preference for some frequent word, could remain).
* **P2.** `E2617` scores at least 0.75 overall (E2621: 0.8175 on version 1, whose headline was not read) and again scores at chance on plain subject–verb agreement and below chance across an attractor.
* **P3.** `E2616` (1,200 updates) again scores at chance on auxiliary agreement, where every 2,400-update arm scored 0.92–0.99 at version 1.

**If I-a holds,** the per-arm numbers are read as descriptions with their intervals; arms with five seeds (when E2618 completes) are additionally summarised by the mean and sample standard deviation of their accuracy,
which is a measurement of the probe's own seed-to-seed spread and the first seed spread of any capability measure in this repository.

**What may not change after scoring:** the version-2 pair set, the scoring rule, the models already named, the outcomes and the predictions.

## Known limits, declared in advance

* The balance is under the committed vocabulary only; a model with another tokenizer would need its own floor. All models scored here share it.
* Templated pairs from a 16-noun, 10-verb lexicon: 800 pairs are fewer degrees of freedom than their count suggests; the Wilson interval is for the sampling of those sentences, not for the sampling of English.
* One register; one seed for each model scored before E2618 completes; preference, not generation.
* Balancing identifier counts removes one bias that the untrained floor revealed; it does not prove there is no other. The floor is how the next one would show.

## Cost

Seconds to minutes of CPU per checkpoint (batched scoring); no training.

## Reproduction

```bash
python scripts/evaluate_minimal_pairs.py --pairs configs/probes/grammatical-minimal-pairs-v2.json \
  --checkpoint E2617=data/experiments/E2617/seed-7/E2617-decoder.pt  --control \
  --checkpoint E2615-s00=… --checkpoint E2615-s25=… --checkpoint E2615-s50=… --checkpoint E2615-s75=… --checkpoint E2616=… \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --out experiments/E2622/results.json
```

## Results (added 2026-11-02)

[`results.json`](results.json), from the command in *Reproduction* plus `--control-seeds 11,13,17,19,23,29`: six further untrained models, added after the first scoring showed how the seed-7 control reached chance (models are only ever added,
as the record allows). `E2617b` and the later seeds were not yet complete at scoring time and are not in this table. Contamination screen: **1 of 800** pairs (`determiner_noun_agreement-2c585b9e`) shares a 5-word n-gram with the corpus. Scoring: 5 min 38 s for
thirteen models on a host busy with a training run (35 min for eight, one sentence at a time, in E2621).

| Phenomenon | E2615-s00 | -s25 | -s50 | -s75 | E2616 | E2617 | untrained control (seed 7) | untrained, 7 inits: min – max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| auxiliary agreement | 0.92 | 0.98 | 0.94 | 0.99 | 0.49 | 0.93 | 0.49 | 0.44 – 0.51 |
| determiner–adjective order | 0.93 | 0.97 | 1.00 | 0.96 | 0.99 | 0.98 | 0.82 | **0.02 – 0.82** |
| determiner–noun agreement | 0.93 | 0.97 | 0.98 | 0.95 | 0.78 | 0.92 | 0.48 | 0.44 – 0.52 |
| modal takes base form | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.09 | **0.00 – 0.97** |
| past-participle form | 0.94 | 0.86 | 1.00 | 0.97 | 0.94 | 1.00 | 0.38 | 0.35 – 0.70 |
| reflexive-pronoun agreement | 0.73 | 0.67 | 0.81 | 0.91 | 0.69 | 0.85 | 0.52 | 0.44 – 0.54 |
| subject–verb agreement | 0.46 | 0.46 | 0.46 | 0.49 | 0.46 | 0.46 | 0.54 | 0.46 – 0.55 |
| subject–verb agreement, attractor | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.54 | 0.46 | 0.46 – 0.54 |
| **all 800** | 0.8063 | 0.8063 | 0.8413 | 0.8512 | 0.7362 | **0.8350** | **0.4725** | 0.406 – 0.578 |
| 95 % interval | [0.777, 0.832] | [0.777, 0.832] | [0.814, 0.865] | [0.825, 0.874] | [0.705, 0.766] | [0.808, 0.859] | [0.438, 0.507] | |
| **usable phenomena only (300 pairs)** | 0.860 | 0.873 | 0.910 | 0.950 | 0.653 | **0.900** | 0.497 | 0.470 – 0.507 |
| 95 % interval | [0.816, 0.895] | [0.831, 0.906] | [0.872, 0.937] | [0.919, 0.969] | [0.598, 0.705] | [0.861, 0.929] | [0.440, 0.553] | |

Token-count-only reading: exactly 0.5 for every model, as constructed. Length-only (characters): 0.502. Checkpoint checksums are those of E2621.

### The preregistered reading

* **Outcome: I-a.** E2617's interval [0.808, 0.859] excludes 0.5; the seed-7 control's [0.438, 0.507] includes it; the length-only reading 0.502 lies below E2617's interval. The balance removed the token-count bias E2621 found.
* **Usable phenomena** (control includes 0.5, E2617 excludes it): **auxiliary agreement, determiner–noun agreement, reflexive-pronoun agreement.** Unusable: *modal takes base form* (control 0.09, below chance), *determiner–adjective order* (control 0.82),
  *past-participle form* (control 0.38), and both subject–verb phenomena, where E2617 itself is at chance (0.46 [0.366, 0.557] and 0.54 [0.443, 0.634]).
* **Predictions.** P1 (control within 0.05 of chance overall): held, 0.4725. P2 (E2617 ≥ 0.75 overall; at chance on plain subject–verb agreement): held (0.835; 0.46). The extension that it would be *below* chance across an attractor **did not
  hold**: 0.54, at chance — the 0.38 in E2621 belonged to the version-1 pair subset, and is withdrawn as a finding. P3 (E2616 at chance on auxiliary agreement): held, 0.49.

### What the extra controls add, and what they limit

* **The headline floor is not a stable quantity.** Across seven untrained initialisations the overall accuracy is 0.406 to 0.578; three of the seven intervals exclude 0.5. The seed-7 control reached chance partly by cancellation (modal 0.09, order 0.82,
  past participle 0.38). The preregistered rule is met by the seed-7 control, and the other six controls say that it could easily not have been. **The all-800 numbers are therefore reported and not leaned on.**
* **The three usable phenomena are robust to the control.** Their untrained floor stays within 0.44–0.54 per phenomenon in all seven initialisations, and 0.470–0.507 pooled. On those 300 pairs the trained models separate from every untrained one by a wide margin
  (0.86–0.95 against at most 0.507; E2616 0.653 is also above every control's upper interval end of 0.563).
* **Descriptively, and with one seed per model**, the usable-phenomena accuracy rises with prose share (0.860, 0.873, 0.910, 0.950 at 0, 25, 50, 75 %) and E2617 sits at 0.900; the 0 % and 75 % intervals do not overlap. E2616, with half the updates, is far lower (0.653), and it is
  at chance on auxiliary agreement while the 2,400-update arms are at 0.92–0.99. These are descriptions of seven checkpoints. They are not a finding about composition until the seeds of E2618 give the probe's own seed-to-seed spread.
* **Limits that stand:** templated pairs, a small lexicon (so within a phenomenon the 100 pairs share a handful of distinct word forms, which is also why a single random initialisation can swing a whole phenomenon), one register, preference rather than generation. 300 usable pairs is a small instrument.
  Nothing here is understanding, knowledge or usefulness (K-001).

### Consequence

Version 2 is usable for **three phenomena**, with a seven-initialisation untrained floor. The two phenomena where the trained model is itself at chance (subject–verb agreement) are the most informative negative result of the probe: the decoder does not prefer the
agreeing verb in simple intransitive sentences, in a register full of them. The next steps are in the E2618 seeds (the probe's spread) and a version 3 that draws the lexicon's distinct word forms more widely, if the probe is to be extended; neither is started here.
