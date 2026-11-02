# E2618 — seeds on the unresolved arms of the prose-share sweep: can 0 %, 25 % and 50 % prose be ordered?

**Status:** measured — **H9-a′: an interior prose share beats none, on the equal-weight register mean; the 25 % arm is lowest and ordered against both others.** Preregistered and committed (`f29bd00`) before any new seed ran; the Results section was added after all fifteen reports were on disk.
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

## Results (added 2026-11-02)

Produced by `scripts/analyse_arm_replication.py` from files that are committed: the three arm aggregates [`E2618-s00-study.json`](E2618-s00-study.json), [`E2618-s25-study.json`](E2618-s25-study.json), [`E2618-s50-study.json`](E2618-s50-study.json) (each rebuilt by
`chakaso experiment study … --collect-only` from the five reports on disk), E2614's `results.json` as the noise study, and [`results.json`](results.json). Compact extracts of the twelve new reports are in [`reports/`](reports/) (seed 7 is E2615's, in [`../E2615/reports/`](../E2615/reports/)),
each with the SHA-256 of the full report and checkpoint. All fifteen runs: this host (4 threads, torch 2.15.0+cpu), the 2,400-update ceiling, none early-stopped, none diverged, none re-run except as disclosed below. *n* = 5, df = 12, *t* = 2.7795 at the Bonferroni level, δ = 0.0062.

### The equal-weight register mean O1, by seed

| Prose share | seed 7 | 11 | 13 | 17 | 19 | mean | sample sd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 2.01892 | 2.02053 | 2.01427 | 2.01915 | 2.02001 | **2.01858** | 0.00249 |
| 0.25 | 2.00981 | 2.00224 | 2.01597 | 2.01417 | 2.00422 | **2.00928** | 0.00600 |
| 0.5 | 2.02162 | 2.02045 | 2.02344 | 2.02528 | 2.02616 | **2.02339** | 0.00240 |

Pooled within-arm standard deviation **0.00400** (E2614's was 0.00224, interval 0.00134–0.00644): inside it, so E2615's margins are not restated. The 25 % arm is the noisiest (sd 0.0060, near the interval's upper end).

### The preregistered reading

| Pair (difference of means) | *d* | interval (Bonferroni, 95 % family) | grade | lower O1 |
| --- | ---: | --- | --- | --- |
| 0 % vs 25 % | +0.00930 | [+0.00227, +0.01633] | **ORDERED** | 25 % |
| 0 % vs 50 % | -0.00482 | [-0.01185, +0.00222] | **UNRESOLVED** | 0 % |
| 25 % vs 50 % | -0.01411 | [-0.02114, -0.00708] | **ORDERED** | 25 % |

* **Outcome: H9-a′.** The lowest mean is the 25 % arm, and it is graded ORDERED, not negligible, against the 0 % arm (*d* = 0.0093, interval [0.0023, 0.0163], the interval excludes zero and reaches beyond the 0.0062 margin, so it is not negligible). It is also ORDERED against the 50 % arm
  (*d* = 0.0141, interval [0.0071, 0.0211]). The 0 % and 50 % arms are UNRESOLVED from one another (*d* = 0.0048, interval [−0.0119, 0.0022]).
* **What that says and no more.** On the equal-weight mean O1, an interior share (25 %) is better than none and better than 50 %. With E2615's 100 % endpoint separably worse than all three (+0.2088 against 25 %, one seed, not re-run) the curve has an interior minimum near a quarter of the budget.
  The size is small: 0.009 and 0.014 bits per character, under 1 % of O1.
* **Rule 1 applies.** The 25 % arm becomes the **candidate** working mixture for the next experiments. It does not replace the natural mixture (ADR-0041) by this record: that is a decision with its own record, because O1 is a convention (equal weight to five registers, four of them technical) and a prose-weighted objective names a different share.
* **Predictions.** P1′ held (prose is lower at 50 % than 25 % than 0 %, both contrasts ORDERED). P2′ held (technology is higher at 50 % than 25 % than 0 %, both ORDERED). P3′ held (each arm's O1 standard deviation lies inside E2614's interval).

### Per register and whole pack (descriptive, not decision-bearing)

| Metric | 0 % | 25 % | 50 % | pairs (0–25, 0–50, 25–50) |
| --- | ---: | ---: | ---: | --- |
| prose | 1.9840 | 1.8790 | 1.8401 | 0–25 ordered, 0–50 ordered, 25–50 ordered |
| science | 1.9375 | 1.9356 | 1.9435 | 0–25 unresolved, 0–50 ordered, 25–50 ordered |
| technology | 1.8409 | 1.8639 | 1.8975 | 0–25 ordered, 0–50 ordered, 25–50 ordered |
| medicine | 1.7882 | 1.8043 | 1.8242 | 0–25 ordered, 0–50 ordered, 25–50 ordered |
| mathematics | 2.5423 | 2.5636 | 2.6116 | 0–25 unresolved, 0–50 ordered, 25–50 ordered |
| whole pack | 2.1052 | 2.0622 | 2.0600 | 0–25 ordered, 0–50 ordered, 25–50 unresolved |

Prose and the technical registers trade off monotonically and every pair is separated except science (0–25) and mathematics (0–25): prose falls with prose share, technology, medicine and mathematics rise, as E2615's single seeds showed. The **whole-pack** number, whose weights are an accident of one seeded sample, would have named
the 50 % arm as tied with the 25 % arm and the 0 % arm as worse; the record fixed O1 for that reason, and the disagreement is the reason the answer depends on the weighting.

### Deviations and disclosures

* One launch (the 0 % arm, seed 11) was lost to a container restart and rerun from scratch, as the record declared; the lost attempt wrote no checkpoint and no report, and its directory is kept aside (`seed-11.interrupted-0457Z`). Wall clock per run was 54–70 minutes depending on other work on the host and is not a result.
* The spread is over initialisation and data order at one document selection and one split: **which documents were drawn is the same for every seed of an arm**, so selection noise is not in these intervals (E2615's P3 found the nested 50 % selection a tie with the natural mixture, which is the only check).
* The Bonferroni correction covers the three pairs of this analysis, not the later comparisons that will use its result.

### What this does not say

It does not say 25 % is the best mixture for any other objective or corpus, that it will stay best at a longer budget or a larger model, or that a model trained on it is more capable: nothing here is capability (K-001). The held-out side has 1–35 works per register (medicine one); the mathematics and medicine partitions are a handful of works.
