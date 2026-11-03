# E2630 — H17 at E2614's own seeds: does the space-prefixed gain survive more than one draw on this host?

**Status:** preregistered — no number in this record was read before it was committed. **Type:** Research experiment,
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
