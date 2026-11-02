# E2619 — passes versus updates at fixed updates: does reading the same prose more times cost held-out prose?

**Status:** measured — **Q-a: the smallest pool (about four passes) is worst on held-out prose by 0.011–0.015 bits per character; one and two passes tie.** Preregistered and committed (`91bb931`) before any arm ran; the Results section was added after the three reports were on disk.
**Type:** Research experiment, **preregistered**. It is task T13 of the active ledger, the follow-up [E2616](../E2616/record.md) named when its
one-pass arm could not separate passes from updates.
**Protocol files:** [`E2619-prose-5m.toml`](../../configs/experiments/E2619-prose-5m.toml),
[`E2619-prose-10m.toml`](../../configs/experiments/E2619-prose-10m.toml) and [`E2619-prose-20m.toml`](../../configs/experiments/E2619-prose-20m.toml),
read by `chakaso experiment study --protocol … --seeds 7`. A test holds each to E2614's on every field except the prose-only filter and the cap
(the 20M file differs from E2614's only in the filter).

**Question.** [E2613](../E2613/record.md) compared a prose-only arm at 10M identifiers (about two passes in 2,400 updates; prose **1.80621**) with the natural
mixture T (**1.83952**) and a 20M prose control P (about one pass; **1.80926**). [E2616](../E2616/record.md) then gave a prose-only model one pass over the same
10M with a *halved* schedule (1,200 updates) and it scored **1.93944**: worse than T by 0.10, which the record read as "the halved schedule explains the
whole gap, so passes and updates are not separated". This experiment holds the updates fixed at 2,400 and moves only how many distinct identifiers they
draw from, so the number of passes over the data is the only thing that changes: 5M (about four passes), 10M (about two), 20M (about one).

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed — E2614's configuration: corpus `312571f9fa8351a8`, whole-work split seed 7 (`adf650435c3a9068`), vocabulary `434debe82f949347`, the
3,413,760-parameter rotary decoder, **2,400 updates × 8,192 positions = 19,660,800 positions in every arm**, cosine to 0.1, AdamW 6e-4, warmup 200, seed 7,
the same evaluation packs, this host (4 threads, torch 2.15.0+cpu). The training text is **prose works only**, drawn in the order of one seeded shuffle of the
prose works (split seed 7) and cut after the first whole unit that reaches the cap, so the 5M pack is a prefix of the 10M pack, which is a prefix of the 20M pack.

Moves — the cap: **5,000,000**, **10,000,000** and **20,000,000** identifiers. Prose holds 24.5M identifiers on the training side, so no arm needs more
than the pool holds. Passes at 2,400 updates: about 3.93, 1.97 and 0.98.

**References already on disk, and why they are not substituted for runs.** E2613 arm H *is* the 10M configuration, and E2612 arm P *is* the 20M one. They ran
on the other host (14 threads, torch 2.13.0+cpu) under earlier code. This experiment re-runs both on this host so that all three arms share one machine; the
earlier numbers are reported beside them, and the difference between each pair is a cross-machine check on those two configurations (read against
[E2617](../E2617/record.md)'s result for the natural mixture).

## Metric, margins and decision rules, declared before any arm runs

**Primary metric:** held-out bits per character of the **prose partition**, *X*<sub>5</sub>, *X*<sub>10</sub>, *X*<sub>20</sub>. Tie margin **0.00625** and separable margin
**0.02**, from E2614 (prose spread *s* = 0.00226 at the natural mixture, **assumed** to hold for a prose-only arm; E2618 and this experiment's single seeds do not test it).

* **Q-a — repetition costs prose.** *X*<sub>5</sub> − *X*<sub>20</sub> > 0.00625, and *X*<sub>5</sub> is the largest of the three. Reading the same prose more times, at equal
  updates, scores worse; passes are a real axis, with the 10M arm placing the curve.
* **Q-b — repetition is free in this range.** The three values lie within 0.00625 of one another (max − min ≤ 0.00625). Distinct identifiers do not matter between
  4 passes and 1 at this budget; the E2616 gap is then entirely a matter of updates.
* **Q-c — anything else.** Reported with the three values and the pairwise differences graded tie / indeterminate / separable; no mechanism is claimed.

**What each outcome changes.** Q-a: any future prose budget should be spent on more distinct text before more updates over the same text, and E2616's R-a becomes
"updates, and also passes" rather than "updates". Q-b: updates are the binding quantity and the data pool can be re-read up to four times with no measurable cost here. Q-c:
nothing already recorded changes.

**Also reported, not decision-bearing:** the other four partitions and the equal-weight mean (a prose-only arm is expected to sit worse than the mixture on the
technical registers, as H did); each arm's training pack size in identifiers and the realised passes (positions ÷ packed identifiers); the validation curve at the six
evaluation points; the comparison with E2613-H and E2612-P as described above.

**Descriptive predictions (not decision-bearing).** (i) Prose bits per character is non-decreasing as the pool shrinks. (ii) E2613-H and E2612-P differ by 0.00305, a tie, which
predicts that the 10M and 20M arms here tie. A separable difference between them would contradict that prediction and is reported as a surprise.

**Stopping, failure and what may not change.** Every arm runs its 2,400 updates with no early stopping. A crashed arm is rerun once; a second failure is reported. A diverged
arm is recorded as diverged. The caps, the margins, the outcomes and the predictions are not changed after a result is read; a change is a new record.

## Known limits, declared in advance

* One seed per arm; the margin is E2614's natural-mixture spread, assumed for a prose-only arm. A Q-b reading therefore means "not shown", not "small".
* Prose only, so the arms say nothing about repetition of technical text, whose pools are 3.3–10.6M identifiers and are read more than once in every mixture arm.
* The pool is nested and drawn by one shuffle: a different shuffle could change the 5M pack's content. The selection effect is not measured.
* Truncated budget, archaic English, a held-out prose partition of 35 works.
* Epochs, optimizer updates and tokens are three different quantities here: positions are fixed, updates are fixed, and passes vary only because the pool varies. Nothing in this
  design separates a pool-size effect from a repetition effect inside one arm.

## Cost

Three runs at roughly 50 minutes each on this host plus corpus preparation: about **2.5–3 hours**, sequential, resumable.

## Reproduction

```bash
for tag in 5m 10m 20m; do
  chakaso experiment study --work-dir data/experiments --experiment E2619-prose-$tag --seeds 7 --split-seed 7 \
    --expected-corpus-fingerprint 312571f9fa8351a8 \
    --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
    --protocol configs/experiments/E2619-prose-$tag.toml --write-report data/experiments/E2619-prose-$tag/study.json
done
python scripts/read_passes_at_fixed_updates.py --arm 5m=data/experiments/E2619-prose-5m/seed-7/report.json \
  --arm 10m=data/experiments/E2619-prose-10m/seed-7/report.json --arm 20m=data/experiments/E2619-prose-20m/seed-7/report.json \
  --out experiments/E2619/results.json
```

## Results (added 2026-11-02)

Produced by `scripts/read_passes_at_fixed_updates.py` from files that are committed: the three compact extracts in [`reports/`](reports/), E2613's `results.json` as the reference, and [`results.json`](results.json). All three arms: this host (4 threads, torch 2.15.0+cpu), seed 7, 2,400 updates × 8,192 positions = 19,660,800,
`budget_ceiling`, none early-stopped, none diverged, none re-run. Wall clock 65–70 minutes each on a host also running other work (not a result).

| Pool | Passes (positions ÷ packed identifiers) | **Prose** bits per character | science | technology | medicine | mathematics | equal-weight mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5m (5,005,218 identifiers) | 3.928 | **1.82055** | 2.0775 | 2.2078 | 2.0132 | 3.1829 | 2.2604 |
| 10m (10,001,764 identifiers) | 1.966 | **1.80570** | 2.0591 | 2.1889 | 1.9988 | 3.1351 | 2.2375 |
| 20m (20,000,402 identifiers) | 0.983 | **1.80928** | 2.0589 | 2.1872 | 1.9951 | 3.0472 | 2.2195 |

### The preregistered reading

* **Outcome: Q-a.** The range of the three prose values is 0.01485, over the 0.00625 tie margin; the 5M arm is the largest (1.82055) and is worse than the 20M arm by **0.01127** (over the tie margin, under the 0.02 separable margin: graded *indeterminate*).
  Reading the same prose about four times, at equal updates, scores worse than reading it once.
* **The curve is not a line.** 10M against 20M: −0.00358 (the 10M arm is numerically *better*), a tie. 5M against 10M: +0.01485, indeterminate. So **one and two passes are indistinguishable and four passes cost about 0.011–0.015 bits per character on prose** — the placing the declaration reserved for the 10M arm.
* **Predictions.** (i) Prose is non-decreasing as the pool shrinks: **did not hold as worded** (10M is below 20M, by an amount that is a tie). (ii) E2613-H and E2612-P tie, hence the 10M and 20M arms here tie: **held** (−0.00358).
* **The technical registers worsen as the pool shrinks** (the equal-weight mean of a prose-only arm is 2.2195, 2.2375, 2.2604 for 20M, 10M, 5M), as expected of arms that never saw those registers; it is reported, not decision-bearing.

### The cross-host check this record promised

The 10M and 20M arms are E2613-H and E2612-P re-run here. This host scores **1.80570** against H's 1.80621 (**−0.00051**) and **1.80928** against P's 1.80926 (**+0.00002**): prose-only runs agree across the two machines to within 0.0005, consistent with E2617's prose difference of −0.00033 for the natural mixture.
Two more configurations with a machine effect on prose below the 0.0005 reporting tolerance.

### What it changes

* **E2616 is read again.** Its one-pass 1,200-update arm scored 1.93944 against 1.8057 for two passes of the same 10M at 2,400 updates: the 0.134 gap is **updates**, since passes between one and two cost nothing measurable here. E2616's R-a ("a halved schedule explains the whole gap") stands, and "passes and updates are not separated" is now answered in one direction:
  at this scale a halved schedule costs an order of magnitude more than a doubled pass count (to four) does.
* **H10 gets a first measured point.** Four passes over 5M identifiers cost 0.011–0.015 bits per character of prose relative to one; one to two passes cost nothing measurable. A single seed, a single corpus, a prose-only arm.
* No mixture, default or budget is changed by this record.

### Limits that stand

One seed per arm (the margin is E2614's natural-mixture spread, assumed for a prose-only arm; E2618's pooled spread of 0.004 on O1 suggests it is not smaller, so the 0.011 indeterminate gap is about 1.5–2 pair-standard-errors wide at best). The pool is nested and drawn by one seeded shuffle: a different shuffle could change the 5M pack's
content. Prose only. Truncated budget, archaic English, a held-out prose partition of 35 works. Epochs, updates and tokens are three quantities here; this design varies the pool at fixed positions and updates and cannot separate a pool-size effect from a repetition effect inside an arm. Nothing here is capability (K-001).
