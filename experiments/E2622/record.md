# E2622 — the minimal-pair probe, version 2: with identifier counts balanced, is the untrained floor at chance, and does the probe then see what the trained decoders learned?

**Status:** preregistered, not yet run. Committed before any checkpoint is scored on version 2; the Results section is added after.
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
