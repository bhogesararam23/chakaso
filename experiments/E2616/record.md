# E2616 — the one-pass control: does a prose-only model reading the mixture's prose once score the mixture's prose?

**Status:** measured — **R-a, with the record's own short-schedule flag raised**. Preregistered and committed before the arm
ran; the Results section was added after the report was on disk.
**Type:** Research experiment, **preregistered**. It is the follow-up [E2613](../E2613/record.md) declared and did not
run, and its result is read against numbers already on disk.
**Protocol file:** [`configs/experiments/E2616-one-pass-prose.toml`](../../configs/experiments/E2616-one-pass-prose.toml),
read by `chakaso experiment study --protocol … --seeds 7`. A test holds it to E2614's on every field except five.

**Question.** E2613 found that a prose-only arm at 10M identifiers (H) scored held-out prose **1.80621**, against the
mixture arm T's **1.83952** (−0.03331, outside the 0.02 band) and the 20M prose control P's **1.80926**. It also
declared a confound it could not remove: H made about **two passes** over its 10M identifiers in 2,400 steps, while T
made one pass over each of its two ~10M halves. So H read roughly twice as many prose *positions* as T. The finding was
therefore: a step budget partly spent on technical text costs prose, **by an amount about equal to what spending those
positions on prose buys** — not that technical text harms prose.

This record asks the remaining question: **give a prose-only model exactly one pass over the same ~10M prose
identifiers, so it reads the same number of prose positions T did, and compare its prose to T's and to H's.** If it
scores like T, the second pass was the difference. If it scores like H, technical text displaced prose learning
even at equal prose positions.

It is a measurement of one small model's held-out loss at a truncated budget. It says nothing about capability
(K-001).

## What is held fixed, and what moves

Held fixed — E2614 / E2613-H's configuration: corpus `312571f9fa8351a8`, whole-work split seed 7
(`adf650435c3a9068`), vocabulary `434debe82f949347` from the committed file, the 3,413,760-parameter rotary decoder,
batch 32, AdamW 6e-4, weight decay 0.01, cosine to 0.1, the same evaluation packs, seed 7, and **the training text**:
prose works only, shuffled with split seed 7 and capped at 10,000,000 identifiers, which is E2613 arm H's selection
(10,001,764 packed identifiers).

Moves — five values, all about how long the schedule is, none about what is read:

| Field | E2613-H | This arm | Why |
| --- | ---: | ---: | --- |
| `max_steps` | 2,400 | **1,200** | 1,200 × 8,192 = 9,830,400 positions: 0.983 passes over the pack |
| `warmup_steps` | 200 | 100 | the same fraction of the schedule |
| `eval_every` | 400 | 200 | the same six evaluations |
| `training_group_prefixes` | `prose-` | `prose-` | unchanged in effect; declared in the file because E2614's file has none |
| `max_training_tokens` | 10,000,000 | 10,000,000 | unchanged; E2614's file says 20M |

(The last two rows are changes **relative to the E2614 file**, which is why they are in the file's diff; relative to
arm H they change nothing.)

## What this design cannot remove, declared before the run

* **It changes the number of optimiser updates as well as the number of passes.** The arm makes 1,200 updates; T made
  2,400, about half of whose positions were prose. Equal prose positions is the point, but "equal prose positions" and
  "equal updates" cannot both hold, and this record chooses the first. A result that scores worse than T can be a
  short-schedule effect, and the record will say so rather than read it as composition.
* **One seed.** E2614 measured a prose standard deviation of 0.00226 at the natural mixture; the spread of a
  prose-only arm at 1,200 steps is not measured and is assumed to resemble it.
* **Truncated budget, archaic English, whole-work split of 62 held-out works**, as every record in the series.

## Metric, margins and decision rules, declared before the run

**Primary metric:** the held-out bits per character of the prose partition, *X*. The reference values are fixed now:
**T = 1.83952** (E2612 arm T, reproduced exactly as E2614 seed 7; five-seed mean 1.83986), **H = 1.80621** (E2613) and
P = 1.80926 (E2612). The tie margin for the prose partition is E2614's central band, **0.00625** (2.77 × 0.00226).
T − H = 0.03331, more than twice that margin, so the two readings below cannot overlap.

* **R-a — the second pass was the difference.** *X* ≥ T − 0.00625 (that is, *X* is within the tie margin of T or
  worse). A prose-only model reading the mixture's prose once scores the mixture's prose; technical text is not what
  costs prose at this budget. If *X* is worse than T by more than the separable margin (0.02), the arm is also
  evidence that 1,200 updates are fewer than the budget needs, and the record says so.
* **R-b — technical text displaced prose learning.** *X* ≤ H + 0.00625. One pass of prose alone, at half the compute,
  already scores like two passes: the mixture's prose cost is not explained by its prose positions.
* **R-c — in between.** T − 0.00625 > *X* > H + 0.00625. Reported with *X*'s position between H and T as a fraction;
  no mechanism is claimed and the question is not closed.

**What each outcome changes.** R-a reinforces E2613's reading and makes H9's prose curve a function of prose positions
seen. R-b makes the mixture itself the cost, and the prose-share sweep ([E2615](../E2615/record.md)) the place to
measure it. R-c changes nothing already recorded. None of them selects a mixture.

**Also reported, not decision-bearing:** the other four partitions and the equal-weight register mean against H and P
(an arm that never saw technical text should sit with them, as H did); the realised packed identifiers; the
determinism of the training text against E2613-H (the packed identifier count must equal 10,001,764).

**Stopping and failure.** The arm runs its 1,200 steps with no early stopping. A crash is rerun once; a second failure
is reported. A non-finite loss is recorded as diverged, not retried.

## Cost

About 45 minutes of CPU for 1,200 steps, plus corpus preparation of about three minutes; one seed.

## Reproduction

```bash
chakaso experiment study --work-dir data/experiments --experiment E2616 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2616-one-pass-prose.toml --write-report data/experiments/E2616/study.json
# compare_arms reports max_steps as unmatched against H and T, by design: it is the declared difference.
python scripts/compare_arms.py --arm one_pass=data/experiments/E2616/seed-7/report.json \
  --arm H=data/experiments/E2613/half-prose-e2612-vocab/report.json --arm T=data/experiments/E2612/mixture/report.json \
  --reference H --aggregate technical=science-,technology-,medicine-,mathematics-
```

## Results (added 2026-11-02, after the report was on disk)

Produced by [`scripts/read_one_pass_control.py`](../../scripts/read_one_pass_control.py) from the committed extract
[`E2616-report.json`](E2616-report.json) and E2613's committed `results.json` (the T, H and P values the record fixed),
into [`results.json`](results.json). The arm ran 1,200 of 1,200 steps (`budget_ceiling`, no early stop, no divergence, no
re-run) in 1,485 s. Its packed training text is **10,001,764 identifiers, equal to E2613-H's**, so the training text is the
same; this is the determinism check the record asked for. It ran on the same host and library as the E2615 arms (see
[E2615's execution notes](../E2615/record.md)); the references T, H and P come from the earlier machine.

| | prose bits per character | against X |
| --- | ---: | ---: |
| **X, this arm (1,200 updates, one pass)** | **1.93944** | |
| T, the mixture (2,400 updates) | 1.83952 | X − T = **+0.09992** |
| H, prose only, two passes (2,400 updates) | 1.80621 | X − H = **+0.13323** |
| P, 20M prose (2,400 updates) | 1.80926 | X − P = +0.13018 |

**Outcome: R-a.** X is far above T − 0.00625, so a prose-only model reading the mixture's prose once does **not** score the
mixture's prose: it scores about a tenth of a bit worse. R-b and R-c are excluded by the same numbers. X − T exceeds the
separable margin (0.02) five times over, so the record's flag applies as written: **this arm is also evidence that 1,200
updates are fewer than the budget needs.**

### What this licenses, and what it does not

* **It does not separate the two explanations the question set up.** The record declared that "equal prose positions" and
  "equal updates" cannot both hold and chose the first; the result is that the arm with half the updates is much worse than
  either reference. A gap of 0.0999 to T, three times the 0.0333 gap that E2613 set out to explain, is what a shortened
  schedule is expected to produce, so the shortened schedule is a sufficient explanation of the whole difference and the
  composition question is left where E2613 left it. R-a's formal reading ("the second pass was the difference") is true of
  the outcome, but this design cannot tell a second pass from a second thousand updates, and the record said so beforehand.
* **The other four partitions, not decision-bearing,** are also all worse than H's, including the technical registers neither
  arm trained on: technology 2.3007 against H's 2.1900, medicine 2.1171 against 2.0006, science 2.1873 against 2.0593,
  mathematics 3.1168 against 3.1086. The equal-weight mean is 2.3323 for X against 2.2329 (H), 2.2186 (P) and 2.0214 (T). That
  the unseen registers worsen along with prose is what a general under-training would do, and is in line with the first point.
* **Observation across records, not a result of this one.** E2615 held updates fixed at 2,400 and moved the prose share:
  prose falls steadily with share, and `s50` (10M prose identifiers) ties the natural mixture T on prose, while `s100` (20M)
  is within 0.004 of H (10M read twice). At a fixed number of updates, prose bits per character therefore tracks the number of
  prose positions read more than their uniqueness. That is consistent with E2613's reading and is not a test of it, because
  E2615's arms were not designed to hold passes and positions apart.
* **A cleaner version of the question would hold updates fixed and vary the passes**, for example prose-only at 2,400 updates
  over 5M identifiers (four passes) against 10M (two) and 20M (one). That is a new record; nothing here is changed by it.
* **One seed.** A prose-only arm's spread at 1,200 steps is unmeasured, as declared. The gap is more than forty times
  E2614's prose standard deviation (0.00226), so a different seed is not a plausible explanation of its size.
