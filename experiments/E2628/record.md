# E2628 — H17 at a second seed, on the other host: does the space-prefixed tokenizer's gain survive a different initialisation?

**Status:** preregistered — this record is committed **before** the arm runs. The Results section will be added after the report is on disk, and the margins, outcomes and predictions below will not change.
**Type:** Research experiment, **preregistered**. It is the second seed of the cell [E2625](../E2625/record.md) measured, run on the
machine that has not carried a space-prefixed arm yet.
**Protocol file:** [`configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml`](../../configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml),
read by `chakaso experiment study --protocol … --seeds 11`. **No new protocol file**: the point is that nothing moves but the seed and
the host, exactly as [E2617](../E2617/record.md) reused E2614's file for the same reason. The `--seeds 11` argument is the only difference
from E2625's own command, and the study's protocol digest checks that claim rather than the record asserting it.

## Question

[E2620](../E2620/record.md), [E2624](../E2624/record.md) and [E2625](../E2625/record.md) triangulated the space-prefixed pre-tokenizer
(H17) from three directions and every one of them rests on **one seed**. E2625 is the cleanest design — the control's 2,400 updates over the
control's own 2,756 training documents, so only the tokenizer and the number of passes differ — and it gained **+0.0229 on O1** and
**+0.0286 on the whole pack**, outcome X-a, against E2617 on the Linux host.

H17's falsification rule is a margin measured from E2614's five-seed spread: a one-seed difference is material above 2.77·*s*, which is 0.02
on O1 and 0.00933 on the whole pack. That band belongs to the **control** tokenizer. The treatment arm's own seed-to-seed spread has never
been measured, and E2620's and E2625's records say so in their limits sections. A vocabulary that compresses differently can train
differently: it changes which parameters do work (the embedding table is the same 4,096 × 256 but is now spent on words rather than on word
gaps) and it changes how many distinct tokens a position sees. If the treatment's spread is larger than the control's, some part of the
measured gain is the luck of one initialisation.

**This arm is the second draw.** Δ₁₁ = E2614 seed 11 (default tokenizer, this host) − E2628 (space-prefixed, this host, seed 11). Compared with
Δ₇ = E2617 − E2625 (both the Linux host), the two Δs are two independent draws of the same quantity under the same design, on two machines.

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed — E2625's protocol, unchanged: corpus `gutenberg-multiregister-v1` 1.0.0 (`312571f9fa8351a8`), whole-work split with split seed 7
(`adf650435c3a9068`), the 3,413,760-parameter rotary decoder, 2,400 updates × 8,192 positions, warmup 200, cosine to 0.1, AdamW 6e-4 with the
settings ADR-0049 now declares, **14,033,424 training identifiers** (the control's own 2,756 documents, so the treatment reads 1.40 passes of
the same distinct text), the committed vocabulary `4a47f0281c81a9e5` read as an input (ADR-0044), and the held-out pack selected by counting
with the default-rule vocabulary `434debe82f949347` (ADR-0048) so the documents scored are the control's.

Moves — **the training seed (7 → 11) and the host** (the Linux container that ran E2625 → this Windows machine, 14 threads, torch 2.13.0+cpu).
The control for this arm is therefore not E2617 but [E2614](../E2614/record.md)'s seed-11 run, which trained on this host with the default
vocabulary, and whose numbers are in that study's committed `results.json`: whole pack **2.0587330335285547** bits per character and O1
**2.0235569480223940** (the mean of its five committed partition values).

**Seed and machine move together here, and that is a confound, not a cleverness.** It cannot be taken apart on this machine without running a
second seed on the Linux host, which is not available in this session. What bounds it is E2617's own measurement: the same protocol and seed on
the two hosts differ by at most **0.0025** bits per character, and its equal-weight register mean by **0.00067**, which is 17 % of its tie margin
at the very most. So the difference between Δ₁₁ and Δ₇ is the seed effect plus at most that, and no conclusion below rests on separating them.

## Metrics, margins and rules, declared before the run

Primary **O1** (equal-weight mean of the five register partitions), secondary the **whole pack**. Δ = control − treatment, positive when the
treatment is better. Margins are E2614's, as in E2620 and E2625: O1 tie **0.0062**, separable **0.02**; whole pack tie **0.00933**.

* **Y-a — the gain survives a second draw.** Δ(O1) > 0.02 **and** Δ(whole pack) > 0.00933, i.e. this arm reproduces E2625's direction beyond the
  band. With E2625 that is two independent seeds of the same design both material. Consequence: the space-prefixed rule becomes the candidate
  default on two seeds, and the remaining step before it replaces anything is a decision record, not another arm.
* **Y-b — the gain does not survive it.** |Δ(O1)| ≤ 0.0062 and |Δ(whole pack)| ≤ 0.00933, or Δ(O1) < 0. Then the two draws are 2.0235 → ? and
  whatever E2625 said, and **the H17 gain is not established beyond one initialisation**: the record says which of the two Δs is larger and the
  consequence is that the default does not change and neither does the roadmap's ranking.
* **Y-c — the arm is worse by more than the band.** Δ(O1) < −0.02 and Δ(whole pack) < −0.00933. E2625's X-a was a one-seed artifact and H17 is
  falsified in the strong direction at this design point; this outcome is reported as such and the rule is not a candidate.
* **Y-d — anything else**, with both Δs and the two numbers beside them.

The reading is made by [`scripts/read_tokenizer_arm.py`](../../scripts/read_tokenizer_arm.py) `--same-documents`, which refuses an arm whose
updates or training documents differ from the control's, and the control is read from E2614's committed `results.json` rather than retyped.

**Descriptive predictions (stated so a surprise is recognisable; none decision-bearing).**

* **P1.** Δ(O1) lies between E2625's +0.0229 and E2624's −0.0351, the two arms that bracket this design on updates and distinct text.
* **P2.** The treatment's seed-11 O1 is close to its seed-7 O1 (1.9943 on the Linux host): the treatment's own spread is expected to resemble
  the control's 0.0024 on O1, so a difference of more than 0.01 between the two seeds of the *treatment* would be the surprise.
* **P3.** Every partition moves in the same direction as O1, as it did in E2620 and E2625.
* **P4.** The whole-pack number is worse than E2626's 2.05544739454246 for the same seed family on this host by less than the band, because the
  treatment differs from the control only in its tokenizer and reads the same documents.

**Stopping, failure and what may not change.** The arm runs its 2,400 updates with no early stopping and is expected to complete
`budget_ceiling`, as every arm in this series has. A crashed run is rerun once; a second failure is reported with its log. The margins,
outcomes, predictions and the objective are not changed after the result is read.

## Known limits, declared in advance

* Two seeds is still two. A treatment spread estimated from two draws is known only to within a factor of about 0.76–1.81 on σ at 95 % (E2614's
  own note on what three and five seeds buy), so this arm can confirm the direction of the gain and cannot bound its variability.
* Seed and host move together, bounded above by E2617's measured 0.0025 and not separable on this machine.
* One corpus, one split, one budget; both arms' curves are truncated at the step ceiling, so every absolute number here is a floor (E2612's
  reading, unchanged); archaic English; a handful of works per partition.
* Nothing here is a capability measurement and nothing here changes the default: that needs a decision record (K-001 stands).

## Cost

The vocabulary is a committed input, so there is no fit. One arm at 2,400 updates × 8,192 positions on this host: E2614's seed 7 took 4,991 s
and its seed 11 took 8,902 s with other work sharing the machine, so **1.4–2.5 hours**, and E2626's ladder ran at 3,766 positions per second
alongside a busy session. Peak resident 2.6 GB (measured in E2614's report), which this machine has.

## Reproduction

```bash
# The control is E2614's seed-11 run on this host. read_tokenizer_arm.py reads a run report, not a study
# aggregate, so the committed evidence is a compact extract of that run's report.json carrying the same
# `extract` block (source file sha256, note) E2617 and E2625 cited for their control.
chakaso experiment study --work-dir data/experiments --experiment E2628 --seeds 11 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2625-space-prefixed-equal-updates-and-documents.toml \
  --write-report data/experiments/E2628/study.json
python scripts/read_tokenizer_arm.py --treatment experiments/E2628/E2628-report.json \
  --control experiments/E2628/E2614-seed-11-control-report.json --same-documents \
  --out experiments/E2628/results.json
```

Both reports are committed beside this record after the run: the treatment's own extract and the control's
extract of `data/experiments/E2614/seed-11/report.json`, so the comparison is aggregated from files in the
repository rather than from numbers typed into prose, and `results.json` is rebuilt from them by
`--collect-only`-style reading without training anything.

## Results

_Not yet. The arm had not been launched when this record was committed._
