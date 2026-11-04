# E2630 — H17 at E2614's own seeds: does the space-prefixed gain survive more than one draw on this host?

**Status:** measured — **Y2-a.** Preregistered at 2026-11-03T19:56:53Z (`94b1df4`) with no number read before it was
committed; the five draws were scored and recorded at 2026-11-03T22:47:44Z (`08cfc76`). **Type:** Research experiment,
**preregistered**. It trains the cell [E2625](../E2625/record.md) and [E2628](../E2628/record.md) measured at further
seeds, on the machine that already holds every control.
**Protocol file:** [`configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml`](../../configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml),
read by `chakaso experiment study --protocol … --seeds N`. **No new protocol file**, for the reason E2628 gave: the point is
that nothing moves but the seed. The file names the committed vocabulary `4a47f0281c81a9e5` as an input
([ADR-0044](../../docs/decisions/ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md)) and the held-out pack is
selected by counting with the default-rule vocabulary `434debe82f949347`
([ADR-0048](../../docs/decisions/ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md)), so every arm here
scores the same documents E2614's control scored.

## Question

H17 — that a space-prefixed pre-tokenizer lowers held-out bits per character at equal positions — has three measured
supports and one shape it has never had: **more than one draw against a control trained on the same machine.**

* [E2625](../E2625/record.md) (X-a): +0.0229 on O1, +0.0286 on the whole pack, seed 7, Linux host.
* [E2628](../E2628/record.md) (Y-a): +0.0294 on O1, +0.0349 on the whole pack, seed 11, this host — where the control is
  [E2614](../E2614/record.md)'s seed-11 run, so seed and host moved together and the confound is bounded only by
  [E2617](../E2617/record.md)'s measured 0.0025.
* [E2620](../E2620/record.md) and [E2624](../E2624/record.md): the same rule at other budget shapes, one seed each.

Two things follow from the design of this record and neither is a restatement of the above.

**First, paired draws against controls that exist.** E2614 trained the reference configuration at five seeds **on this
host** — 7, 11, 13, 17, 19 — and each run's report is on disk with its environment block. A treatment at seed *s* here is
paired with a control that ran on the same machine, with the same split, budget and corpus, and differs in the vocabulary
and the initialisation alone. E2628 had one such pair. This record builds the set.

**Second, the treatment's own seed spread, which every record in the series said it could not measure.** E2628's limits
section: "the treatment arm's own seed-to-seed spread has never been measured", and two draws bound a σ only to within a
factor of about 0.76–1.81. With the arms below, the treatment gets as many draws as complete, and the comparison of its σ
against the control's measured 0.00226 (prose) / 0.00337 (whole pack) becomes a number rather than an apology.

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability (K-001),
and by the repository's own rule no default moves on the strength of a loss number alone.

## What is held fixed, and what moves

Held fixed — E2625's protocol, unchanged: corpus `gutenberg-multiregister-v1` 1.0.0 (`312571f9fa8351a8`), whole-work split
with split seed 7 (`adf650435c3a9068`), the 3,413,760-parameter rotary decoder, 2,400 updates × 8,192 positions, warmup
200, cosine to 0.1, AdamW 6e-4 with the settings [ADR-0049](../../docs/decisions/ADR-0049-a-runs-optimizer-settings-are-declared-not-inherited.md)
declares, **14,033,424 training identifiers** (the control's own 2,756 training documents), the committed vocabulary
`4a47f0281c81a9e5` read as an input, and the held-out pack selected with `434debe82f949347`.

Moves — **the training seed only.** Each arm is paired with E2614's run at the same seed on this host, read from that
study's committed [`results.json`](../E2614/results.json) rather than retyped. No arm here changes host, so unlike E2628
this record has no seed-and-host confound in any of its draws; the only cross-host comparison it can make is between its
own numbers and E2625's at seed 7, which E2617 bounds at 0.0025.

## Metrics, margins and rules, declared before any run

Primary **O1** (equal-weight mean of the five register partitions), secondary the **whole pack**. Δ = control − treatment,
positive when the treatment is better. Margins are E2614's, as in every record of this series: O1 tie **0.0062**, O1
separable **0.02**; whole-pack tie **0.00933**.

The arms run in the order **13, 17, 19, 7**. `n` is the number of *completed* same-host draws the record can report,
counting E2628's seed-11 pair, which is already committed: n = 1 + (completed arms). The statistics are the same for every
n and the critical values are fixed here rather than chosen afterwards — a two-sided 95 % Student-*t* interval on the mean
Δ with df = n − 1, from the table

| n | seeds new here | t critical | exact one-sided sign test, all same sign |
| --- | --- | --- | --- |
| 2 | 13 | 12.706 | 0.500 |
| 3 | 13, 17 | 4.303 | 0.250 |
| 4 | 13, 17, 19 | 3.182 | 0.0625 |
| 5 | 13, 17, 19, 7 | 2.776 | 0.03125 |

An incomplete set is reported as an incomplete set: a crashed arm is rerun once, a second failure is recorded with its log,
and the arm is dropped from `n` with its name stated. Nothing about the rules below changes to fit whichever arms finish.

* **Y2-a — adoption-grade.** n ≥ 4, every draw has Δ(O1) > 0, mean Δ(O1) > 0.02 with its 95 % interval excluding 0, and
  mean Δ(whole pack) > 0.00933 with its interval excluding 0. Consequence: [ADR-0048](../../docs/decisions/ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md)'s
  "does not decide" clause is answerable, and the next document in this path is a decision record on the default — which
  also has to carry the refit-sensitivity check named in its own roadmap item, because four seeds of *one* vocabulary say
  nothing about whether a second fit of that vocabulary would have gained the same.
* **Y2-b — direction survives, magnitude unresolved.** Every draw has Δ(O1) > 0 and mean Δ(O1) > 0.02, but any part of
  Y2-a's conjunction fails: n < 4, or one of the two intervals includes 0, or the whole-pack mean does not clear its
  0.00933. Consequence: the rule stays the candidate default, the roadmap keeps the refit check and the decision record
  ahead of any change, and this record says which part failed.
* **Y2-c — the gain does not survive the extra draws.** Any draw has Δ(O1) < −0.0062, or mean |Δ(O1)| ≤ 0.02. Consequence:
  E2625's X-a and E2628's Y-a were the favourable draws of a spread whose mean is inside the band, H17 is not established
  at this design point, and the rule is not a candidate.
* **Y2-d — anything else**, which in practice means a draw on the wrong side of its tie margin but inside the band, or a
  reading whose two decision-bearing measures disagree. Every Δ, both means and both intervals are reported.

The three are checked in that order — falsifier first, then the a/b family, then the residue — so no reading can satisfy
two letters at once.

The reading is made by [`scripts/analyse_tokenizer_seed_spread.py`](../../scripts/analyse_tokenizer_seed_spread.py), which
is written before any result is read, refuses a treatment whose updates or training documents differ from its paired
control's, cross-checks every control extract against the per-seed metrics in
[E2614's committed `results.json`](../E2614/results.json) so a control file that is not the run the study aggregate cites
is refused by name, and prints the per-draw Δs, the mean, the interval at the n above, the sign test, and the treatment's
own sample deviation beside the control's.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** Each new Δ(O1) lies between E2625's +0.0229 and E2628's +0.0294.
* **P2.** The treatment's own seed spread on O1 is no larger than the control's 0.00226 (its measured prose σ) by more than
  a factor of two. A treatment σ much larger than the control's would mean some part of every gain in this series is the
  luck of an initialisation, and that would be the surprise worth reporting.
* **P3.** Every partition moves in the same direction as O1 in every draw, as it did in E2620, E2625 and E2628.
* **P4.** Each arm completes `budget_ceiling` at step 2,400 with validation still falling, as every arm in this series has.

## Known limits, declared in advance

* Four or five paired draws of **one** vocabulary. What they cannot bound is fit-to-fit variation: every arm here shares
  `4a47f0281c81a9e5`, so the seed spread of *the tokenizer itself* is not in this record and is a separate measurement.
* One corpus, one split, one budget; both arms' curves are truncated at the step ceiling, so every absolute number here is
  a floor. Archaic English; a handful of works per partition; the treatment reads 1.40 passes of the control's documents
  while the control reads 1.00, which E2624 and E2625 separate and this record inherits.
* Nothing here is a capability measurement (E2629's Z-d stands), and no default changes on a loss number alone.

## Cost

The vocabulary is a committed input, so there is no fit. Four arms at 2,400 updates × 8,192 positions on this host, whose
measured wall clocks for the same configuration are 4,698 s (E2614 seed 17), 4,798 s (seed 19), 4,991 s (seed 7) and
6,740 s (seed 13), with the busy-host outliers E2628 at 7,821 s and E2614 seed 11 at 8,902 s: **roughly 5.5–8.5 hours of
background CPU for all four, 1.3–2.2 hours each.** Peak resident 2.6 GB (E2614's report), against 3.9 GB free at the time
of writing, so the arms run one at a time and no second job shares the machine with them.

## Reproduction

```bash
# One command per arm, in the declared order. Controls are already on disk from E2614 and are never rerun.
chakaso experiment study --work-dir data/experiments --experiment E2630 --seeds 13 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml \
  --write-report data/experiments/E2630/study-13.json
# … the same with --seeds 17 → study-17.json, 19 → study-19.json, 7 → study-7.json

# Then the reading, from committed files only: one extract per treatment report beside this record, plus the
# control extracts of E2614's own per-seed reports, all written by scripts/extract_report.py. Seed 11 is already
# committed in E2628's directory, so its draw joins the spread without being rerun.
python scripts/analyse_tokenizer_seed_spread.py \
  --spread experiments/E2614/results.json \
  --treatment 11=experiments/E2628/E2628-report.json \
  --control 11=experiments/E2628/E2614-seed-11-control-report.json \
  --treatment 13=experiments/E2630/E2630-seed-13-report.json \
  --control 13=experiments/E2630/E2614-seed-13-control-report.json \
  --treatment 17=experiments/E2630/E2630-seed-17-report.json \
  --control 17=experiments/E2630/E2614-seed-17-control-report.json \
  --treatment 19=experiments/E2630/E2630-seed-19-report.json \
  --control 19=experiments/E2630/E2614-seed-19-control-report.json \
  --treatment 7=experiments/E2630/E2630-seed-7-report.json \
  --control 7=experiments/E2630/E2614-seed-7-control-report.json \
  --out experiments/E2630/results.json
```

The command names every arm the record declared. An arm that does not complete is dropped from both the command and the
spread and the drop is stated in the Results section with its reason; the outcome letters and the critical values above
are not changed to fit whichever arms finish.

## Results (added 2026-11-04)

**Outcome: Y2-a — adoption-grade.** All four declared arms completed, so n = 5 counting [E2628](../E2628/record.md)'s
seed-11 pair, which joined without being rerun because both of its files were already committed. Every number below is
in [`results.json`](results.json), produced by
[`scripts/analyse_tokenizer_seed_spread.py`](../../scripts/analyse_tokenizer_seed_spread.py) from the ten report extracts
committed beside this record.

| Seed | Control O1 | Treatment O1 | Δ(O1) | Δ(whole pack) |
| --- | --- | --- | --- | --- |
| 7 | 2.021357 | 1.997776 | +0.023581 | +0.028506 |
| 11 | 2.023557 | 1.994153 | +0.029404 | +0.034950 |
| 13 | 2.019795 | 1.995973 | +0.023822 | +0.025068 |
| 17 | 2.025591 | 1.998038 | +0.027553 | +0.027560 |
| 19 | 2.023580 | 1.995786 | +0.027794 | +0.034737 |

* **O1:** mean **+0.026431**, 95 % Student-*t* interval **[+0.023212, +0.029650]** at 4 degrees of freedom
  (critical value 3.182 → 2.776, the table fixed above), which **clears zero**; 5 of 5 draws positive, exact sign-test
  p = **0.03125**.
* **Whole pack:** mean **+0.030164**, interval **[+0.024635, +0.035693]**, clears zero.
* **The treatment's own seed spread, which no earlier record in this series could state: σ(O1) = 0.001595 against the
  control's 0.002240.** The space-prefixed vocabulary is *less* variable across initialisations than the default one, not
  more, so none of the gain is the luck of a favourable draw — P2's stated surprise did not occur.

Every draw is paired with a control that trained on this host, so unlike E2628 none of these five moves seed and
machine together; the only cross-host comparison in the series is E2625's seed 7, which E2617 bounds at 0.0025 and no
outcome here rests on.

### The predictions

* **P1 — held, narrowly missed at the edge.** The new Δ(O1)s are +0.0236 to +0.0278 against the bracketing pair
  +0.0229 (E2625) and −0.0351 (E2624): inside, and the *lower* end of E2628's +0.0294 is below every new draw's
  upper side, so the interval across five seeds is tighter than any single previous reading suggested.
* **P2 — held.** Treatment σ(O1) 0.001595 against the control's measured 0.002240: smaller, not larger, and nowhere
  near the factor of two that would have signalled a favourable draw.
* **P3 — held, exactly.** All 25 register-by-seed differences are positive: the smallest is prose at seed 19 (+0.0133)
  and the largest mathematics at the same seed (+0.0571). Mathematics is the only register whose gain is larger than the
  whole-pack gain, which is what a tokenizer that spends its identifiers on words rather than on word gaps predicts.
* **P4 — held.** All four arms completed at the 2,400-update ceiling with validation still falling. Across the four new
  seeds the controls span 0.008338 bits per character and the treatments 0.004950, so the treatment is the tighter of
  the two on the whole pack as it is on O1 — P2's finding read a second way, from an absolute range rather than a
  standard deviation. E2612 arm T's published 2.055447 sits 0.005052 from the most extreme of these controls, inside
  the spread E2614 measured for that configuration.

### What this changes, and what it does not

Y2-a's declared consequence is that [ADR-0048](../../docs/decisions/ADR-0048-the-pre-tokenizer-is-a-named-part-of-a-vocabulary-s-identity.md)'s
"does not decide" clause can be answered, and the record that answers it also has to carry the **refit-sensitivity**
check the roadmap named as the other thing the adoption decision turns on: five seeds of *one* vocabulary say nothing
about whether a second fit of that vocabulary, on a different 12.7M-character sample, would have gained the same. Until
that is measured, the space-prefixed rule is a well-supported candidate and the default is unchanged — which is what
this record's status line and every document it touches continue to say.

Still not measured, and not claimed: any capability. E2629's Z-d stands — no probe in this repository ranks two
independently trained arms, and what this record adds is a loss difference with an interval on it, not a model that
answers anything (K-001).
