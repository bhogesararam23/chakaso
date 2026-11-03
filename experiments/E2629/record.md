# E2629 — can either instrument rank two independently trained arms? The prose-share pair is the test and the half-prose arm is the null

**Status:** preregistered — the analysis landed as code and tests in `d03b69c`, the three checkpoints are already on
this machine, and the same-region probe has already scored them (its result is committed as
[E2627](../E2627/record.md)'s `context.json`). **No model has been scored for this record**: neither the pair probe
nor the version-1 continuation probe has ever been run on these three arms, and the decision rules and predictions
below are fixed before that happens.
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

"$QPYTHON" scripts/analyse_arm_pairs.py \
  --arm "E2612-T=data/experiments/E2612/mixture/report.json" \
  --arm "E2612-P=data/experiments/E2612/prose/report.json" \
  --arm "E2613-H=data/experiments/E2613/half-prose-e2612-vocab/report.json" \
  --reference E2612-T \
  --pairs experiments/E2629/minimal-pairs.json \
  --continuations experiments/E2629/continuations.json \
  --context experiments/E2627/context.json \
  --usable-phenomena auxiliary_agreement,determiner_noun_agreement,reflexive_pronoun_agreement \
  --materiality 0.02 --out experiments/E2629/results.json
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
