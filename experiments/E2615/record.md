# E2615 — the prose-share sweep: how do held-out bits per character depend on the share of training identifiers drawn from prose?

**Status:** preregistered — committed before any arm ran. No result exists; the Results section is added only after
the reports are on disk.
**Type:** Research experiment, **preregistered**. It is hypothesis **H9** of the 2026-11-01 campaign, restated below
because its original wording does not name a quantity that can be measured.
**Protocol files:** [`E2615-s00`](../../configs/experiments/E2615-s00.toml), [`s25`](../../configs/experiments/E2615-s25.toml),
[`s50`](../../configs/experiments/E2615-s50.toml), [`s75`](../../configs/experiments/E2615-s75.toml), each read by
`chakaso experiment study --protocol … --seeds 7`. This record and those files are one declaration; the files run. A
test holds them equal to [E2614's](../../configs/experiments/E2614-decoder-seed-spread.toml) on every field except the
share.
**Mechanism:** [ADR-0045](../../docs/decisions/ADR-0045-training-composition-is-declared-as-a-share-of-the-token-budget.md).

**Question.** E2612 changed the training mixture at one point and found a trade-off: technical text cost nothing it
could not repay on technical held-out text (−0.27584) and cost prose 0.03026. E2613 showed the prose cost belongs to the
mixture, not to seeing less prose. The *shape* of the trade-off is unknown: whether training on more prose than the
natural 51 % buys prose back cheaply, whether training on less prose costs prose steeply, and whether any share is
better than both ends. This experiment draws that curve at five points.

It is a measurement of how one small model's held-out loss responds to training composition at a truncated budget.
It says nothing about capability (K-001).

## Why H9 is restated before it is run

H9 said "held-out bits per character over a mixed register corpus is minimised at an interior prose share." That is
not one quantity, for a reason the pool measurement below found: **the repository's "whole test pack" is not a
register-balanced objective.** It is the held-out documents shuffled by the split seed and packed until 250,000
identifiers, and replaying that shuffle (`scripts/measure_training_pool.py`, replay labelled as such in its output)
gives its register weights as **prose 41.9 %, science 45.3 %, technology 5.8 %, mathematics 7.0 %, medicine 0.0 %**
(256,125 identifiers). Nobody chose that, and medicine, a register E2612–E2614 trained on and scored, is absent from it.
An optimum of the whole-pack number would be an optimum of one random sample's composition.

A single "best share" also does not exist without a weighting, because the registers pull in opposite directions:
every weighting has its own optimum, and the optima differ. So this record fixes the objective **before** the run, and
reads everything else per register.

## What is held fixed, and what moves

Held fixed — E2614 / E2612 arm T's configuration, unchanged, one seed (7):

| Condition | Value |
| --- | --- |
| Corpus | `gutenberg-multiregister-v1` 1.0.0, fingerprint `312571f9fa8351a8` |
| Split | whole-work, split seed 7, 312 / 56 / 62 works, zero on more than one side; fingerprint `adf650435c3a9068` |
| Vocabulary | `434debe82f949347`, from the committed file `experiments/E2612/vocabulary-434debe82f949347.json`, checked by fingerprint ([ADR-0044](../../docs/decisions/ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md)) |
| Model | rotary decoder, 3,413,760 parameters, context 256 |
| Budget | 2,400 steps × 8,192 positions = 19,660,800, cosine to 0.1, AdamW 6e-4, warmup 200, batch 32 |
| Training pool per arm | 20,000,000 identifiers (`max_training_tokens`), units shuffled with split seed 7 |
| Evaluation | the same packs as E2612–E2614: 250,000 identifiers per register partition |

**What moves: the share of the 20M identifiers drawn from works whose group starts with `prose-`.**

| Arm | Prose share | Prose identifiers | Other identifiers | Source |
| --- | ---: | ---: | ---: | --- |
| `E2615-s00` | 0.00 | 0 | 20.0M | run here |
| `E2615-s25` | 0.25 | 5.0M | 15.0M | run here |
| `E2615-s50` | 0.50 | 10.0M | 10.0M | run here (nested control) |
| `E2615-s75` | 0.75 | 15.0M | 5.0M | run here |
| `s100` | 1.00 | 20.0M | 0 | **E2612 arm P, reused, not re-run** |

*Why `s100` is reused:* the share-1 selection is exactly the filter-and-shuffle selection E2612-P used (ADR-0045 (3),
held by a test), and the pool measurement replays it at **20,000,402** prose identifiers, the number E2612's report
records. The arm comparison proves the conditions matched before any number is read, and refuses a mislabelled arm.

*Why a nested `s50` is run when the natural mixture T already exists:* T draws the first 20M identifiers of the whole
shuffled side (prose 51.03 % by replay), which is a different document selection from "the first 10M prose and the first
10M other". Running `s50` in the same nested family as the others removes that difference from every neighbouring
comparison, and comparing `s50` with T's five seeds measures how much the document selection itself matters. **T is
therefore a reference, not an arm.**

**Why no finer grid and no 0.125:** the grid is fixed now. Quarters are as fine as seven-hour CPU budgets and a
single seed per arm can usefully resolve, and the margins below are larger than the differences a finer grid would
tell apart. An arm added later is a new record.

**Why a 0 % endpoint:** without it, a curve that falls toward the low-prose end cannot be told from a boundary optimum,
and the question the edge arm answers — what a model that never saw a prose register scores on prose — is itself one
this project has not measured. It is not proposed as a mixture to train.

### What each register can supply (training side, identifiers under the committed vocabulary)

Measured by `scripts/measure_training_pool.py` before any arm was declared; no arm needs more than its pool holds, and
every arm stays under one pass over each class.

| Register | Works | Documents | Identifiers | Largest need (arm) | Passes at that need |
| --- | ---: | ---: | ---: | --- | ---: |
| prose | 150 | 3,354 | 24,523,613 | 20.0M (`s100`) | 0.82 |
| science | 69 | 1,527 | 10,601,631 | | |
| technology | 44 | 676 | 4,776,975 | | |
| medicine | 23 | 487 | 3,311,285 | | |
| mathematics | 26 | 468 | 4,526,846 | | |
| **non-prose total** | 162 | 3,158 | 23,216,737 | 20.0M (`s00`) | 0.86 |

Within the non-prose class the four technical registers arrive in the proportions the shuffle gives, not in equal
shares; each arm's report states the realised counts by register.

The held-out side holds: prose 35 works / 791 documents / 5,851,632 identifiers; science 12 / 437 / 3,145,450;
technology 9 / 136 / 969,745; mathematics 5 / 72 / 540,385; **medicine 1 / 7 / 47,008**. The medicine and mathematics
partitions are a handful of works; they are scored as measured, and the noise below is the training noise only.

## Metrics

From each arm's report: bits per character on each of the five register partitions
(`transformer.test_partitions[prefix].metrics`) and on the whole pack; the technical aggregate weighted by scored
characters (K-027); the realised training composition. Aggregated by `scripts/analyse_share_sweep.py` from the reports on
disk, never typed in.

**Primary objective (O1): the equal-weight mean of the five register partitions**, lower is better. It is chosen because
the model is meant to be general and the only weighting that is not an artefact is "each register once". It is a
convention, stated so a reader can reject it; it is also dominated numerically by the four technical registers.
Because the weighting decides the answer, O1 is **not** the only reading: the per-register curves and the dominance
table below are decision-relevant for anyone with another weighting.

## Noise and margins, from E2614 (declared before any arm ran)

A difference between two single-seed arms is judged against the seed spread E2614 measured at the natural mixture, in
three grades: **tie** (at or under 2.77·*s*), **indeterminate** (between that and 2.77·*s*<sub>hi</sub>, the upper end
of the 95 % interval for σ at five seeds), **separable** (over 2.77·*s*<sub>hi</sub>). The 0.02 prose band and the 0.035
mathematics band E2614 set are kept: a kept band can only raise the separable margin, a replacement band also stands in
for the central one.

| Metric | Tie at or under | Separable over | Basis |
| --- | ---: | ---: | --- |
| prose | 0.00625 | **0.02000** | E2614 R1; the 0.02 band is kept |
| science | 0.00933 | 0.02681 | R2: no declared band |
| technology | 0.00291 | 0.00838 | R1 numerically; no declared band |
| medicine | 0.01036 | 0.02977 | R2: no declared band |
| mathematics | 0.03500 | 0.09808 | R3: the replacement band 0.035 is the tie margin |
| **O1, equal-weight mean** | 0.00620 | **0.02000** | s = 0.00224 over the five E2614 seeds; 2.77·*s*<sub>hi</sub> = 0.01783 is under 0.02, so by the same rule as prose the 0.02 band is kept |

These are computed by the analysis script from E2614's `study.json`; the table is its output, copied here before any arm
ran. **The spread was measured at the natural mixture** and is assumed to apply at every share; a mixture with little
prose may be noisier on prose. The document-selection noise (which works were drawn) is *not* in these numbers: all five
E2614 seeds drew the same documents.

## Hypothesis and decision rules, declared before any run

**H9 (restated).** The equal-weight mean O1 has an interior optimum in prose share.

The outcome is the one `classify_share_sweep` returns for O1 over the five arms, with margin 0.02:

* **H9-a — supported.** The arm with the lowest O1 is `s25`, `s50` or `s75`, and it is lower than **both** `s00` and
  `s100` by more than 0.02.
* **H9-b — falsified at a boundary.** The arm with the lowest O1 is `s00` or `s100`, and it is lower than **every other
  arm** by more than 0.02.
* **H9-c — unresolved.** Anything else: a best arm that clears only one endpoint, or arms within 0.02 of the best. No
  share is named. The record lists the arms the data cannot tell apart from the best, and nothing is concluded about
  which of them is better.

Rules that follow, declared now:

1. **H9-a**: the best share becomes the working mixture for the next experiments, and the next unit adds seeds at that
   share and at its best neighbour before anything is built on the difference between them.
2. **H9-b**: H9 as worded is false for O1. The edge arm is not proposed as a training mixture: a model that never sees
   prose (or only prose) is not a general model, and a boundary result under a convention-weighted objective is a fact
   about the weighting. The record states the prose and technical numbers of the edge arm beside the best interior arm,
   and any choice among them is a separate decision, with its own record.
3. **H9-c**: the natural mixture stays the working default (ADR-0041), and the next unit adds seeds on the arms
   in the unresolved set rather than on a finer grid.

**Descriptive predictions (not decision-bearing; stated so a surprise is recognisable as one).**

* **P1.** Prose bits per character is non-increasing in prose share across the five arms, and `s00` is the worst by a
  large margin. *Contradicted* by an increase between adjacent arms graded separable.
* **P2.** The weighted technical aggregate is non-decreasing in prose share. *Contradicted* likewise.
* **P3.** The nested `s50` is a tie with T's five-seed mean on O1 and on prose. This is the document-selection check;
  the pair margin against a five-seed mean is the table's margins scaled by √((1 + 1/5)/2) = 0.775. A separable
  difference means the selection matters, and every comparison with T or P then carries that caveat.
* **No prediction** is made about *where* O1's minimum sits. Four of five registers are technical, so equal weights favour
  a low prose share; how low, and whether an edge arm wins, is what the sweep is for.

**Dominance.** An arm *dominates* another if it is separably better on **every** register. The analysis lists the pairs.
The expected absence of any pair (a trade-off) is itself the finding.

**Stopping, failure and what may not change.** Every arm runs to the 2,400-step ceiling with no early stopping
(`patience = 99`). A crashed arm is rerun once; a second failure is reported, not dropped. A diverged run (non-finite
loss) is recorded as diverged, not retried with other settings. The grid, margins, objective, rules and predictions above
are not changed after a result is read; a change is a new record. An arm's realised prose share is checked against its
declared one; a deviation of more than one percentage point is reported as a deviation.

## Known limits, declared in advance

* **One seed per arm.** The margins above carry the uncertainty of that, under the assumption that E2614's spread applies
  at every share. Where a difference is "indeterminate", the right reading is "not shown", not "small".
* **Truncated budget.** Every decoder run so far stopped at its step ceiling with validation still falling; the curve is
  the curve at 19.66M positions, not at convergence.
* **Archaic public-domain English**, five registers of it; no partition is modern.
* **A handful of held-out works per technical register** (medicine: one). The sweep ranks mixtures by how they score on
  *these* works.
* **Share is in identifiers, not characters**, and the technical registers' internal proportions are not controlled.
* **The existing reference points were run under earlier code** (E2612 under the pre-share mechanism). E2614 reproduced
  E2612 arm T to the last digit under the current code, which is the evidence the comparison is sound; the same check
  for P is the arm comparison's matched-conditions proof, not a re-run.

## Cost

Four arms at about 90 minutes of CPU each when the host is idle (E2614's seeds took 4,698–8,903 s on a shared host),
run sequentially: roughly **6–9 hours**, plus about three minutes of corpus preparation per arm. The vocabulary is read
from the committed file and nothing is fitted.

## Reproduction

```bash
# 1. the pool inventory this record cites (no training; ~3 minutes)
python scripts/measure_training_pool.py --protocol configs/experiments/E2614-decoder-seed-spread.toml \
  --snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --work-dir data/experiments/pool --out data/experiments/pool/training-pool.json

# 2. one arm (repeat for s25, s00, s50, s75), a one-seed study per protocol file
chakaso experiment study --work-dir data/experiments --experiment E2615-s25 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2615-s25.toml --write-report data/experiments/E2615-s25/study.json

# 3. the reading, from the reports on disk
python scripts/analyse_share_sweep.py \
  --arm 0.0=data/experiments/E2615-s00/seed-7/report.json --arm 0.25=data/experiments/E2615-s25/seed-7/report.json \
  --arm 0.5=data/experiments/E2615-s50/seed-7/report.json --arm 0.75=data/experiments/E2615-s75/seed-7/report.json \
  --arm 1.0=data/experiments/E2612/prose/report.json \
  --noise-study data/experiments/E2614/study.json --natural-mixture-study data/experiments/E2614/study.json \
  --out experiments/E2615/results.json
```

## Execution notes (added 2026-11-01, second session, before any arm had finished)

Nothing above is changed: not the grid, the objective, the margins, the rules, the predictions or the protocol files. These
notes record how the preregistered arms are being run, and what differs from the conditions the reference runs had. They are
dated and committed while the first arm is still training.

**What happened to the previous session's launch.** The previous session reported that this experiment had been started in
the background. This session found none of it: the container it ran in was gone, `data/` did not exist, no process was
running, and no arm report, checkpoint or log survives. No result from that launch exists anywhere, and none is used. Every
arm below is therefore a fresh run from the committed protocol files.

**Machine and library, different from the reference runs.** E2612 and E2614 were run on a 14-thread host with torch
2.13.0+cpu (the records state it). This session's host has 4 CPU cores and torch 2.15.0+cpu (numpy 2.4.6, Python 3.13.18).
Thread count and kernel selection are not pinned (K-030 item 4), so a bitwise match with the reference runs is **not
expected and not promised**. All four new arms share this one machine and library, so differences *between them* are not
confounded by it. The s100 endpoint and the noise margins come from the other machine, which is a stated caveat on every
comparison that involves them, not a measured one: no arm run here is a same-protocol repeat of an earlier arm, so this
sweep does not measure the cross-machine effect.

**The s100 endpoint is E2612's arm P as committed.** The raw E2612 arm reports did not survive, but
`experiments/E2612/results.json` carries arm P's per-arm report block (budget, corpus identity, test partitions, curve),
with the report file's SHA-256. The endpoint is read from there. The matched-conditions proof (`compare_arms`) can therefore
be applied only to the fields that block holds; whatever it cannot check is reported as unchecked, not as matched.

**The corpus bytes differ in one source; the corpus identity does not.** The corpus was re-acquired from the 430 pinned
URLs at Project Gutenberg's declared pacing. 429 files match their pinned SHA-256. `science-pg3489` ("Fabre, Poet of
Science") does not: upstream's `Last-Modified` for it is 2026-11-01 09:00 UTC, after the 2026-10-30 retrieval. The original
bytes could not be recovered (the Internet Archive is unreachable from this host). The drift guard was not weakened: the
committed catalogue is untouched. A scratch catalogue carrying the new digest was used to build the snapshot, and the result
was accepted only because it reproduces the pinned identity exactly: **fingerprint `312571f9fa8351a8`, 9,388 documents,
149,000,499 characters**, the values the data card records. The one thing that differs is the snapshot's own checksum
(`3a446f54bdac7a6c…` here, `236c8d5a8e4eb6f2…` in the tracked data card), because it covers the raw digest. Reports from this
sweep will carry the new snapshot checksum.

**Ordering and handling.** Arms run sequentially in the order s25, s00, s50, s75, then E2616 ([record](../E2616/record.md)).
The source tree is not edited while an arm runs. Per-arm results are committed as small summaries; checkpoints and reports
stay in `data/`, which is git-ignored, and whether `data/` survives the end of the session is not known (it survived one
worker restart, which killed every process).
