# E2618 — seeds on the unresolved arms of the prose-share sweep: can 0 %, 25 % and 50 % prose be ordered?

**Status:** preregistered, not yet run. Committed before any new seed starts; the Results section is added after the reports are on disk.
**Type:** Research experiment, **preregistered**. It is task T11 of the active ledger and rule 3 of [E2615](../E2615/record.md),
which found H9 unresolved with three arms — `s00`, `s25` and `s50` — that one seed each could not tell apart.
**Protocol files:** E2615's, unchanged — [`E2615-s00.toml`](../../configs/experiments/E2615-s00.toml),
[`E2615-s25.toml`](../../configs/experiments/E2615-s25.toml) and [`E2615-s50.toml`](../../configs/experiments/E2615-s50.toml) — read by
`chakaso experiment study --protocol … --seeds …`. No new protocol file: the seeds are the only addition, and the study refuses a seed
whose protocol digest differs.

**Question.** E2615 measured the equal-weight mean of the five register partitions (O1) at 0, 25, 50, 75 and 100 % prose, one seed per
arm. O1 was lowest at 25 % (2.00981); 0 % (+0.00911) and 50 % (+0.01181) were inside the 0.02 margin of it, so no share was named. The
margins came from E2614's spread at the *natural* mixture and were assumed to hold at every share. This experiment gives each of the
three arms five seeds, measures the spread at those shares directly, and asks whether the three can now be ordered or declared tied.

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed — E2615's configuration, unchanged: corpus `gutenberg-multiregister-v1` 1.0.0 (`312571f9fa8351a8`), whole-work split with split
seed 7 (`adf650435c3a9068`), vocabulary `434debe82f949347` from the committed file, the 3,413,760-parameter rotary decoder, 2,400 × 8,192
positions, cosine to 0.1, AdamW 6e-4, warmup 200, 20,000,000 training identifiers per arm, the same evaluation packs, the same host
(4 threads, torch 2.15.0+cpu) as the existing seed-7 runs.

Moves — **the training seed**, which moves initialisation and data order together (E2614). Seeds: **7, 11, 13, 17, 19**, E2614's list. Seed 7
is E2615's run and is reused as is (same code, same host); **four new seeds per arm, twelve new runs**.

**Not moved: which documents are drawn.** The share-composed training set is a function of the split seed (7), so every seed of an arm trains
on the same documents. The spread measured here is the spread over initialisation and order at one selection, as in E2614. E2615's P3
found the nested `s50` selection a tie with the natural mixture's five-seed mean; no other selection effect is tested.

## Run order

Round-robin by seed so that any prefix of the campaign is balanced across the three arms: seed 11 on `s00`, `s25`, `s50`; then 13; then 17;
then 19. The analysis uses the complete rounds on disk and states *n*. A campaign that stops after round 2 or 3 is reported at *n* = 3 or 4
under the same rules, labelled as such; the headline rules below are written for *n* = 5.

## Metrics

Per seed: bits per character on the five register partitions and the whole pack, from each report. **Primary: O1, the equal-weight mean of the five
partitions** (a convention, declared in E2615 and kept: the model is meant to be general and equal weights are the only weighting that is not an
accident of one sample). Secondary, reported and not decision-bearing: each partition, the whole pack, the scored-characters-weighted technical aggregate.
Aggregated by `scripts/analyse_arm_replication.py` from the reports on disk.

## Statistics, declared before any new seed runs

For each arm: mean and sample standard deviation of O1 over its seeds. The three arms' within-arm variances are pooled
(*s*<sub>p</sub>, df = 3(*n* − 1) = 12 at *n* = 5). For a pair of arms the difference of means *d* has standard error *s*<sub>p</sub>·√(2/*n*), and its confidence
interval uses a Student-*t* quantile with the pooled df at level **1 − 0.05/3** (Bonferroni over the three pairs, so the family-wise level is 95 %).
The quantile is computed numerically in the script and tested against published values. The equivalence margin is **δ = 0.0062**, E2614's O1 tie
margin (2.77 × 0.00224): a difference smaller than one seed's own noise.

Each pair is graded:

* **ORDERED** — the interval excludes 0. The arm with the lower mean is better on O1. If the whole interval also lies inside ±δ the grade carries
  the flag *negligible*: detectable and practically small.
* **TIED** — the interval lies entirely inside ±δ and includes 0: the two arms are equivalent at δ.
* **UNRESOLVED** — anything else (the interval includes 0 and reaches beyond ±δ).

## Decision rules, declared before any new seed runs

Among `s00`, `s25` and `s50`, with *m* the arm with the lowest mean O1:

* **H9-a′ — an interior share is better than none.** *m* is `s25` or `s50` and is graded ORDERED (not negligible) against `s00`. With the
  100 % endpoint separably worse than all three in E2615 (+0.2088 against `s25`; unchanged, not re-run), this is an interior optimum on O1.
* **H9-b′ — no prose is best of the three.** *m* is `s00` and is ORDERED (not negligible) against both `s25` and `s50`.
* **H9-c′ — still unresolved.** Anything else. The record lists the pairs graded TIED and UNRESOLVED and concludes nothing about which arm is better.

Rules that follow:

1. **H9-a′**: the best arm becomes the *candidate* working mixture for the next experiments; the natural mixture (ADR-0041) is replaced only by a
   decision with its own record, because O1 is a convention and a prose-weighted objective would name a different share.
2. **H9-b′**: as E2615's rule 2 — `s00` is not proposed as a mixture, since a model that never saw a prose work is not general; the record states
   its prose and technical numbers beside the best arm that includes prose.
3. **H9-c′**: the three shares are reported as indistinguishable on O1 at this budget and noise level; the weighting, not the data, decides, and the
   record says so. A finer grid is not proposed.

**Noise check.** The pooled O1 standard deviation is compared with E2614's 0.00224 (upper end of its 95 % interval 0.00644 = 2.874 × 0.00224). If it exceeds the upper
end, the spread at low-prose shares is larger than E2615 assumed, E2615's margins are restated in this record's results, and the E2615 gradings are re-read
against them, with each reading that changes listed.

**Descriptive predictions (stated so a surprise is recognisable as one; not decision-bearing).**

* P1′. Prose bits per character is lower at `s50` than at `s25`, and lower at `s25` than at `s00` (E2615's single-seed ordering), with both contrasts ORDERED.
* P2′. Technology bits per character is higher at `s50` than at `s25`, and higher at `s25` than at `s00`.
* P3′. The per-arm seed standard deviation of O1 is within E2614's 95 % interval (0.0013 to 0.0064) in every arm.

**Stopping, failure and what may not change.** Every run goes to the 2,400-step ceiling with no early stopping. A crashed run is rerun once; a second failure
is reported, not dropped. A diverged run is recorded as diverged, not retried with other settings. The seeds, δ, the quantile level, the grades, the outcomes,
the objective and the predictions are not changed after any new seed's result is read; a change is a new record.

## Known limits, declared in advance

* Five seeds estimate a standard deviation to within a factor of about 0.6 to 2.9 at 95 %. Pooling across three arms (df 12) narrows that, **under the assumption
  that the three arms share one variance**, which the per-arm standard deviations are shown against so a reader can reject it.
* The documents each arm trains on are fixed across seeds. The spread over *which documents* is not measured.
* One split: the held-out side has 1 to 35 works per register (medicine: one); mathematics and medicine partitions are a handful of works.
* Truncated budget (every decoder run so far ended at its step ceiling with validation still falling) and archaic English. Numbers from the two hosts are
  never mixed in this experiment: all fifteen runs (three arms × five seeds) are from this host.
* O1 is a convention. Another weighting can order the arms differently, and the secondary tables are there for that reader.

## Cost

Twelve runs at 2,859–3,090 s each (E2615's observed range on this host) plus about three minutes of corpus preparation per invocation:
roughly **10–11 hours** of CPU, sequential, resumable (a finished seed's report is reused; an interrupted one is rerun from scratch).

## Reproduction

```bash
# one invocation per (arm, seed) in the order above; repeat for s00, s25, s50 and seeds 11, 13, 17, 19
chakaso experiment study --work-dir data/experiments --experiment E2615-s25 --seeds 11 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2615-s25.toml --write-report data/experiments/E2615-s25/study-seed-11.json
# the aggregate over all five seeds of an arm, training nothing
chakaso experiment study --work-dir data/experiments --experiment E2615-s25 --seeds 7,11,13,17,19 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 --protocol configs/experiments/E2615-s25.toml \
  --collect-only --write-report data/experiments/E2615-s25/study-5-seeds.json --overwrite
python scripts/analyse_arm_replication.py --arm 0.0=data/experiments/E2615-s00/study-5-seeds.json \
  --arm 0.25=data/experiments/E2615-s25/study-5-seeds.json --arm 0.5=data/experiments/E2615-s50/study-5-seeds.json \
  --noise-study experiments/E2614/results.json --out experiments/E2618/results.json
```
