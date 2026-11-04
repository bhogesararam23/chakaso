# E2629 — can either instrument rank two independently trained arms? The prose-share pair is the test and the half-prose arm is the null

**Status:** measured — **Z-d: neither instrument ranks two independently trained arms.** No decision-bearing
instrument separated either material pair at a 0.108246-bit gap (≈32 of E2614's measured seed standard deviations);
the grammatical probe on its usable phenomena put the better model *last* (ρ = −0.866, and −1.000 across all 800
pairs), the two continuation readings ordered the arms correctly and reached no separation (closest p = 0.049 against
Holm's 0.016667 step over three pairs), and the pair inside the band was left alone by all eight instruments.
Preregistered and committed (`9277702`) before any model was scored for this record; the Results section was added
after `results.json` was written, with one deviation named there.
**Type:** Research experiment, **preregistered**. Nothing is trained. It is the question E2626 could not ask, answered
with artifacts that already exist.

## Why this experiment exists

[E2626](../E2626/record.md) measured what the two capability probes can see and closed on a limitation it wrote
itself: its ordering was a training curve, and *rungs of one trajectory are not arms* — they share an initialisation,
a data order and nearly all their weights, which makes a rung pair the easiest ranking an instrument is ever given.
Its sensitivity figures are therefore upper bounds, and its sharpest number — the minimal-pair probe resolving a
0.0218-bit gap while the continuation probe resolves nothing below 0.294 — is a statement about a curve, not about
arms. [E2627](../E2627/record.md) tried the other available population, E2614's five independently trained seeds, and
found no kind separating any of the ten pairs; that is the correct answer for five models of **one** configuration,
whose true gap is zero by construction, and it measures resolution rather than difference.

What has never been done here is the comparison that matters: **two arms, trained independently, on data that is
known to change held-out loss, scored on one item set, with a paired test.** The gap has to be known, which means it
has to come from a loss measurement that is already committed, and the two arms have to share a vocabulary, because
the probe runners refuse a checkpoint whose tokenizer is not the one the items were cut with — and cutting two item
sets would leave the comparison unpaired, which is the weaker test the paired statistics were written to replace.

[E2612](../E2612/record.md) and [E2613](../E2613/record.md) supply exactly that pair, and one more arm to use as a
null.

## The arms, fixed now

| Name | Checkpoint | Whole pack bpc | O1 bpc | What it is |
| --- | --- | --- | --- | --- |
| `E2612-T` | `data/experiments/E2612/mixture/E2612-T-decoder.pt` | 2.05544739454246 | 2.021357 | prose + technical mixture, 20,000,000-token cap |
| `E2612-P` | `data/experiments/E2612/prose/E2612-P-decoder.pt` | 2.163693122795988 | 2.218646 | prose-only, same cap |
| `E2613-H` | `data/experiments/E2613/half-prose-e2612-vocab/E2613-H-decoder.pt` | 2.1682177974655765 | 2.232937 | prose-only at half the cap, `E2612`'s vocabulary |

These are the numbers the records report; `scripts/analyse_arm_pairs.py` reads them from the reports rather than from
this table, and the table is here so a reader can see what the script will refuse.

**The matchedness guard runs before any statistic.** `chakaso.training.arm_comparison` is the module E2613 used to
prove its arms were matched, and this record applies it again: corpus `312571f9fa8351a8`, whole-work split
`adf650435c3a9068`, vocabulary `434debe82f949347`, seed 7, 2,400 steps, cosine schedule, learning rate 0.0006, batch
32, 3,413,760 parameters, and the same held-out denominators — 254,976 positions over 535,282 characters in every
arm. The conditions that differ are the ones E2612 and E2613 declared as the experiment: `training_group_prefixes`
(the mixture reads all five registers; the other two read `prose-` only) and the token cap (`E2613-H` is at
10,000,000 against the other two's 20,000,000). An undeclared difference in any matched condition refuses the
analysis, and that refusal is tested rather than asserted.

**What these arms are not, stated now rather than after the fact.** They share an initialisation — seed 7 in all
three — so this is not a five-seed population. But unlike rungs they are three separate optimisation trajectories
over different data, ending at different weights, and that is the difference E2626 said a curve could not give.
One seed per arm means the gap itself carries [E2614](../E2614/record.md)'s measured noise: s = 0.00337 bits per
character on the whole pack, so the 0.108246 gap is about 32 s and the 0.004525 one is about 1.3 s. The material
pair is not noise. The null pair is inside the noise as well as inside the band, which is why it is asked to be left
alone rather than to be ranked.

## The pairs, and the reversal available in them

Three arms give three pairs, all tested, all corrected as one family:

| Pair | Whole-pack gap | Material under the 0.02 band? |
| --- | --- | --- |
| `E2612-T` vs `E2612-P` | 0.108246 | yes — this is the test |
| `E2612-T` vs `E2613-H` | 0.112770 | yes — the same claim at half the prose dose |
| `E2612-P` vs `E2613-H` | 0.004525 | **no** — the false-positive check |

The third pair is the reason this record is not a foregone conclusion. An instrument that "separates" it is
reporting a difference the quality measure does not have, and E2629 treats that as a failure of the instrument, not
as a small success.

There is also a reversal available on one register, and it is named here so nobody has to invent it afterwards: on
held-out **prose** the ordering is the other way round — `E2613-H` 1.806212, `E2612-P` 1.809261, `E2612-T` 1.839520.
That is E2612's trade-off, measured and recorded there, and it is the most plausible explanation if a probe
contradicts the whole-pack criterion: the criterion this record uses is the whole pack, because that is the unit
E2614's band was measured on.

## The instruments, fixed now

Decision-bearing, three of them, each judged by its own per-pair McNemar cells:

* `minimal_pairs_usable` — the three phenomena E2622 found to have a floor at chance
  (`auxiliary_agreement`, `determiner_noun_agreement`, `reflexive_pronoun_agreement`), 300 pairs, passed as an
  explicit list to the script because the record decides, the script does not infer.
* `continuation_pmi_easy` and `continuation_pmi_hard` — E2623's version-1 held-out continuation probe, items cut
  from the same split's test side with the same register quotas (prose 100, science 55, technology 40, mathematics 20,
  at most 5 per work).

Reported and **not** decision-bearing: `minimal_pairs_all_800` (five of its eight phenomena have a floor that is not
at chance, and E2626 said so), and the four `context_pmi_*` kinds read from E2627's committed `context.json` without
re-scoring — where `permuted` is known to fail its floor at 0.4425 and must not be read as a ranking.

Rank correlation and the S-a…S-d sensitivity verdicts are recomputed and reported for continuity with E2626. They do
not decide this record: with three models a rank correlation is a description of three points, and the question was
never "does it order them smoothly" but "does it get each pair right".

## Decision rules, declared before any model is scored

Holm's step-down over the three pairs within one instrument at alpha 0.05; a pair is *separated* when Holm rejects it,
and its direction is read against the whole-pack criterion.

* **Z-a — the instruments rank arms.** Some decision-bearing instrument ranks **every** material pair in the
  criterion's direction, **and** no decision-bearing instrument contradicts the criterion on any pair, **and** no
  decision-bearing instrument separates the pair inside the band.
* **Z-b — one arm ranking, no contradiction.** Some decision-bearing instrument ranks at least one material pair in
  the criterion's direction, and none contradicts. Weaker than Z-a by design: it admits that the 0.108 and 0.113 gaps
  are the same claim twice and may be visible once.
* **Z-c — they cannot be cited at arm level.** A decision-bearing instrument either ranks a pair the wrong way or
  separates the pair the band calls equal. This is a result, not a embarrassment, and the record names which
  instrument and which pair.
* **Z-d — nothing was seen.** No decision-bearing instrument separated a material pair at all.

## Predictions

* **P1 (the campaign's hope, and falsifiable).** `minimal_pairs_usable` separates `E2612-T` from `E2612-P` in the
  criterion's direction. Along one curve that instrument resolved a 0.0218-bit gap; 0.108246 is five times that. If it
  cannot see this pair between two independently trained arms, E2626's sensitivity figures were a property of rungs,
  and that is the finding.
* **P2 (the ceiling claim, put to the sword).** Neither continuation reading separates anything. E2626 resolved no gap
  below 0.294 along the curve, and both material gaps here are below it. **If either separates a material pair, then
  E2627's reading — that the ceiling is the span construction and not the population — is wrong**, and this record
  will say so.
* **P3.** No instrument separates the `E2612-P` / `E2613-H` pair. E2627's forty tests found no separation where the
  true gap is zero; here it is 0.0045, inside both the band and the seed noise. Any separation is a named false
  positive and lands the record on Z-c.
* **P4 (conditional, and stated so a contradiction is explained rather than rationalised).** If a decision-bearing
  instrument contradicts the whole-pack criterion, the likeliest source is the continuation probe rather than the
  grammatical one, because the continuation items are drawn register-by-register from the held-out text where
  `E2612-P` and `E2613-H` are the *better* models on prose (1.809261 and 1.806212 against 1.839520). A contradiction
  from `minimal_pairs_usable` would be the surprise, since its items are this project's own sentences and carry no
  register at all.

## Cost

No training. Two scoring passes over three checkpoints plus the untrained floor: E2626 ran the same two scripts over
thirteen rungs and each pass was tens of minutes, so this is minutes each, and the same-region instrument is read
from a committed file at no cost. Peak memory is one loaded 3,413,760-parameter decoder at a time, 2.5 GB class, and
this machine had 5.1 GB free when this record was written.

## Reproduction

```bash
cd /d/chakaso
QPYTHON=.venv-nn/Scripts/python.exe   # torch is needed to load the checkpoints; nothing is trained
ARMS=(--checkpoint "E2612-T=data/experiments/E2612/mixture/E2612-T-decoder.pt"
      --checkpoint "E2612-P=data/experiments/E2612/prose/E2612-P-decoder.pt"
      --checkpoint "E2613-H=data/experiments/E2613/half-prose-e2612-vocab/E2613-H-decoder.pt")

"$QPYTHON" scripts/evaluate_minimal_pairs.py \
  --pairs configs/probes/grammatical-minimal-pairs-v2.json \
  --control "${ARMS[@]}" --out experiments/E2629/minimal-pairs.json

"$QPYTHON" scripts/evaluate_continuations.py \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --vocabulary experiments/E2612/vocabulary-434debe82f949347.json \
  --expected-split-fingerprint adf650435c3a9068 \
  --registers prose-:100,science-:55,technology-:40,mathematics-:20 --max-per-work 5 \
  --control "${ARMS[@]}" --out experiments/E2629/continuations.json

"$QPYTHON" scripts/evaluate_context_continuations.py \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --vocabulary experiments/E2612/vocabulary-434debe82f949347.json \
  --expected-split-fingerprint adf650435c3a9068 \
  --registers prose-:100,science-:55,technology-:40,mathematics-:20 --max-per-work 5 \
  --control "${ARMS[@]}" --out experiments/E2629/context.json

# The same-region instrument was re-scored here rather than read from E2627's committed file; see Deviation 1. The
# item digest came back f49c128e7b302827c4c5d790b1f685ed94dbfa02971fda17ca1c687e73e5bcc1, the one E2627 fixed.
"$QPYTHON" scripts/analyse_arm_pairs.py \
  --arm "E2612-T=data/experiments/E2612/mixture/report.json" \
  --arm "E2612-P=data/experiments/E2612/prose/report.json" \
  --arm "E2613-H=data/experiments/E2613/half-prose-e2612-vocab/report.json" \
  --reference E2612-T \
  --pairs experiments/E2629/minimal-pairs.json \
  --continuations experiments/E2629/continuations.json \
  --context experiments/E2629/context.json \
  --usable-phenomena auxiliary_agreement,determiner_noun_agreement,reflexive_pronoun_agreement \
  --materiality 0.02 --out experiments/E2629/results.json

for arm in "E2612-T=data/experiments/E2612/mixture/report.json" \
           "E2612-P=data/experiments/E2612/prose/report.json" \
           "E2613-H=data/experiments/E2613/half-prose-e2612-vocab/report.json"; do
  "$QPYTHON" scripts/extract_report.py --report "${arm#*=}" \
    --out "experiments/E2629/${arm%%=*}-report.json"
done
```

The three arm reports are the criterion, and they are git-ignored artifacts, so each is committed beside this record
as a compact extract from `scripts/extract_report.py` and `results.json` cites those extracts. The arm *names* are
identical in the checkpoints, the reports and E2627's committed context result; that is what lets the same-region
instruments be read without re-scoring them, and the script refuses any instrument that did not score every arm.

## Known limits, declared in advance

* **One seed per arm.** The gaps are single draws; E2614's spread says the material pair is 32 s and the null pair
  1.3 s, so the ordering is not noise, but the size of either gap is not measured here.
* **All three arms share seed 7 and one vocabulary.** They are independent in their data and their weights, not in
  their initialisation. This is the arm pair this repository can afford; a tokenizer-differing pair (E2614 seed 11
  against E2628) is unpaired by construction and is not what a McNemar test is for.
* **A "yes" is narrower than it sounds.** The arms differ by training composition, which is what E2612 and E2613 were
  measuring. An instrument that ranks them shows the instrument moves with a distributional change the loss also
  moves with; it does not show the instrument detects a *capability* gain, and it certainly does not say the model is
  useful — [K-001](../../docs/agent/KNOWN_ISSUES.md) is untouched by this record.
* **Loss is still the only criterion**, and nothing here validates it against anything.
* **The items are archaic English** and the pairs are this project's own sentences; `permuted` has a floor that is not
  at chance and is reported, not read.
* **Z-d is a result.** If no instrument separates a 0.108-bit gap between two independently trained arms, the honest
  conclusion is that this project has no capability instrument that has ever ranked two models which differ, and
  that changes what the next instrument may claim.

## Results (added 2026-11-03)

**Outcome: Z-d — no decision-bearing instrument separated a material pair, and the gap was 0.108246 bits per
character, about 32 of E2614's measured seed standard deviations.** Two independently trained arms, one item set, a
paired test, and neither probe can rank them. The null pair was left alone, which is the one part of the design that
went as planned.

### The readings

Whole-pack bits per character is the criterion; the accuracy columns are this record's own scoring runs.

| Arm | Whole pack | O1 | Pairs, usable 300 | Pairs, all 800 | v1 easy | v1 hard | prior | next | permuted | reversed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `E2612-T` | 2.055447 | 2.021357 | 0.89000 | 0.82500 | 0.98605 | 0.95349 | 0.9395 | 0.9674 | 0.9581 | 0.9581 |
| `E2612-P` | 2.163693 | 2.218646 | 0.89000 | 0.83375 | 0.96279 | 0.91163 | 0.9209 | 0.9442 | 0.9302 | 0.9209 |
| `E2613-H` | 2.168218 | 2.232937 | 0.90000 | 0.83750 | 0.95814 | 0.92558 | 0.9256 | 0.9395 | 0.9256 | 0.9163 |
| untrained | — | — | 0.49667 | 0.47250 | 0.49302 | 0.49302 | 0.4512 | 0.4791 | 0.3628 | 0.4326 |

**The grammatical probe puts the better model last.** On its three usable phenomena the mixture arm and the
prose-only arm tie exactly (0.89000 each) and the half-prose arm — the worst model by held-out loss — is the best of
the three (0.90000). Rank correlation against the criterion is **ρ = −0.866** for the usable subset and **ρ = −1.000**
across all 800 pairs: over three independently trained arms the instrument orders them the opposite way to the loss.

**The continuation probes order them correctly and cannot resolve a thing.** Both v1 readings rank the arms in the
criterion's order exactly (ρ = +1.000 for easy, +0.500 for hard), and every material pair points the right way — but
the smallest p anywhere in the whole analysis that agreed with the criterion was 0.03125, and Holm's first step over
three pairs is 0.05/3 = 0.016667. Nothing was rejected.

### The pair tests, exactly

Holm's step-down inside each instrument over the three pairs; a cell is `separated` only when Holm rejects it.

| Pair | Gap | usable 300 | all 800 | v1 easy | v1 hard | prior | next | permuted | reversed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T vs P | 0.108246 | 13:13 p 1.0 | 13:20 p 0.296 | 5:0 p 0.063 | 13:4 p 0.049 | 8:4 p 0.388 | 9:4 p 0.267 | 10:4 p 0.180 | 13:5 p 0.096 |
| T vs H | 0.112770 | 4:7 p 0.549 | 4:14 p 0.031 | 6:0 p 0.031 | 11:5 p 0.210 | 10:7 p 0.629 | 9:3 p 0.146 | 10:3 p 0.092 | 13:4 p 0.049 |
| P vs H | 0.004525 | 9:12 p 0.664 | 10:13 p 0.678 | 3:2 p 1.0 | 6:9 p 0.607 | 7:8 p 1.0 | 8:7 p 1.0 | 9:8 p 1.0 | 9:8 p 1.0 |

Every cell in the third row is a non-separation: **the null pair was never close**, which is what P3 asked for. The
closest anything came to a separation was `v1 hard` and `v1 easy` on T against P/H, both pointing the right way, both
defeated by the family correction rather than by the data.

### Predictions, accounted for

* **P1 failed, and it failed flatly.** `minimal_pairs_usable` did not separate `E2612-T` from `E2612-P`: it found
  thirteen items one way and thirteen the other. E2626's ability to resolve a 0.0218-bit gap along a curve did not
  transfer across a 0.108-bit gap between arms, because along a curve the two models being compared are almost the
  same weights and here they are not.
* **P2 held, with the closest miss of the record.** Neither continuation reading separated anything, so E2627's
  reading — the ceiling is the span construction, not the population — survives an arm pair. But `continuation_pmi_hard`
  reached p = 0.049 on the 0.108 pair, which is under 0.05 and over the 0.016667 Holm step. **Had this record declared
  one pair instead of three, that would have been called a separation.** The correction is what stopped it, and the
  record says so rather than reporting the 0.049 as a near-win.
* **P3 held.** No instrument, decision-bearing or not, separated the inside-the-band pair.
* **P4 did not fire, and its direction was wrong.** No instrument contradicted the criterion *after* the correction,
  so the conditional was never reached; but the uncorrected tendency contradicts, and it contradicts in the
  **grammatical** probe, not the continuation one — 13:20 and 4:14 against the loss across all 800 pairs, ρ = −1.0.
  The prose/technical reversal I named as the likely source of contradiction did show up, but in the instrument whose
  items "carry no register at all" according to this record's own reasoning. That reasoning was the error: an
  800-pair set of self-written sentences still samples constructions a prose-only model has seen more of, and the
  prediction treated "not drawn from the corpus" as "not drawn from the same distribution".

### Two consistency checks the record did not plan for

* `E2612-T` scores 0.82500 across all 800 pairs and 0.89000 on the usable subset — **equal to E2626's `step2400` rung
  to every digit** (0.825 there), the model E2614's seed 7 reproduced. Two records, a month of code changes between
  them, and the same trained distribution reads the same probe identically.
* The same-region items were **re-scored**, not read from E2627's committed file, and the item digest came back
  `f49c128e…`, the one E2627 fixed before it scored anything, with `E2612-T`'s four kind accuracies equal to the
  endpoint values that record prints. That is E2627's deviation, recorded below, and an unplanned reproduction.

### Deviations from the preregistered plan

1. **The same-region instrument was re-scored rather than reused.** The plan was to read `experiments/E2627/context.json`
   at no cost. That committed file is a compact extract, and the extractor stubs per-item arrays by design, so
   `outcomes` — the raw material a paired test is computed from — is not in it. `scripts/analyse_arm_pairs.py` refused
   the file rather than working around it, and the three arms were scored again here. The re-scoring is a check on
   E2627 as much as on this record: the items and the accuracies came back identical. Filed as
   [K-032](../../docs/agent/KNOWN_ISSUES.md): E2626's and E2627's committed probe extracts cannot be re-paired from
   the repository, because the 1 MiB tracked-file bound and the extractor's design pull in opposite directions.
2. **The criterion is read from git-ignored reports**, so all three are committed beside this record as compact
   extracts; the partition numbers E2629 quotes are in them.

### What the refused reading became (added 2026-11-04)

K-032 is closed, and this record is the demonstration. [`context-from-E2627-extract.json`](context-from-E2627-extract.json)
is `scripts/analyse_arm_pairs.py` run with `--context experiments/E2627/context-with-outcomes.json` — the same-region
instrument taken from **E2627's** committed bytes instead of re-scored here, which is the reading that was refused and
deviation 1 worked around. Every one of its eight instrument rows is byte-identical to the row in
[`results.json`](results.json) above, including the four `context_pmi_*` rows, whose discordant counts and p-values now
come from a scoring run this record did not perform. Two things follow and neither changes a verdict: the compact
encoding preserves the paired test exactly, and E2627's scoring of these three checkpoints reproduces E2629's fresh
re-scoring of the same 215 items on the same item digest, item for item.

### What this changes, and what it does not

1. **E2626's caveat is now a measurement, not a caveat.** It published its sensitivity figures as *upper bounds* on
   what a probe would resolve between two independently trained models, because rungs share an initialisation, a data
   order and nearly all their weights. The bound is now known to be loose to the point of inversion: the same
   instrument that resolved 63 of 74 material rung pairs, down to 0.0218 bits, resolves none of the two material arm
   pairs at 0.108 — and orders three arms the wrong way while doing so.
2. **No capability claim in this repository is corroborated at arm level by anything but loss.** E2621–E2623 showed
   trained models beat random initialisations on both probes, and that stands: the untrained rows above sit at
   0.36–0.50 against the arms' 0.83–0.99. It is a floor-versus-ceiling statement, not a ranking. From E2612 onward,
   every sentence of the form "this arm is better than that one" rests on held-out bits per character and on nothing
   else measured here.
3. **A ranking instrument at this scale has to beat 0.016667, not 0.05.** Two of the eight instruments came within a
   factor of two of that on the right side of the gap; a fourth scoring pass over the same arms with more items is the
   obvious next lever, and it is cheap — the item sets are what limit the p-value here, not the models.
4. **The prose/technical reversal is now visible in the grammatical probe too**, uncorrected (13:20, 4:14). That is
   E2612's finding reproduced in an instrument E2612 never touched, and it says the register of the *evaluation items*
   matters even when the items are this project's own sentences.
5. **Nothing about the models moved.** The mixture is still the better arm by the criterion the project uses, E2612's
   trade-off stands as recorded, every arm is still truncated at the step ceiling, and no sample from any of them is
   usable text. [K-001](../../docs/agent/KNOWN_ISSUES.md) is untouched by this record.

### Addendum (2026-11-04) — re-run under the record's own declaration, the verdict is unchanged

Nothing above is edited. `scripts/analyse_arm_pairs.py` used to embed this record's `Z-a`…`Z-d` rules; under
[ADR-0052](../../docs/decisions/ADR-0052-an-analysis-prints-the-verdict-its-record-declared-or-none-at-all.md) they
live in [`configs/analyses/arm-pairs-e2629-v1.json`](../../configs/analyses/arm-pairs-e2629-v1.json) and the analysis
prints no letter unless it is handed one (or told `--statistics-only`). The command above therefore needs two more
flags:

```bash
  --instrument-vocabulary configs/analyses/probe-sensitivity-e2626-v1.json   --outcome-vocabulary configs/analyses/arm-pairs-e2629-v1.json
```

Re-run that way over the committed arm reports and the three probe files, the analysis returns **Z-d** with its `pairs`
array byte-identical to [`results.json`](results.json) and all eight instrument verdicts identical. One block differs:
`matchedness` now also reports `decay_applied_to` as an unequal condition, because `chakaso.training.arm_comparison`
learned to track it after this record ran ([E2631](../E2631/record.md) is where the vector-decay arm was measured). That
is the guard doing its job on a later schema, not a change to this record's arms or its verdict.
