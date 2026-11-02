# E2617 — a cross-machine repeat: does E2614's seed 7 reproduce on a different host?

**Status:** measured — **X-b, with the contingent same-host repeat run: this host reproduces itself bit for bit, so the difference from E2614 is a machine effect, and it is at most 0.0025 bits per character.** Preregistered and committed before the run (`b6362c9`..`d8072ff`); the Results section was added after both reports were on disk.
**Type:** Research experiment, **preregistered**. It is task T12 of the active ledger and the direct test K-030 item 4 asked for.
**Protocol file:** [`configs/experiments/E2614-decoder-seed-spread.toml`](../../configs/experiments/E2614-decoder-seed-spread.toml),
unchanged, read by `chakaso experiment study --protocol … --seeds 7`. No new protocol file is needed: the point is that nothing
moves but the machine.

**Question.** E2615 and E2616 ran on a 4-core Linux host (torch 2.15.0+cpu, 4 threads). E2612–E2614 and the E2612-P endpoint ran
on a Windows host (torch 2.13.0+cpu, 14 threads). Several readings subtract numbers produced on the two (E2615's 0.75 → 1.0 step,
its P3 check, E2616 and E2613 against T, H and P), and the only evidence about the machine effect is a bound on the *sum* of it and
document selection. E2614 showed the protocol is deterministic **on one machine** (its seed 7 repeated E2612 arm T to the last
digit). This experiment repeats that one seed on the other machine and reads the difference.

It is a measurement of numerical reproducibility. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed — E2614 seed 7, entirely: corpus `gutenberg-multiregister-v1` 1.0.0, fingerprint `312571f9fa8351a8`; whole-work split,
split seed 7 (`adf650435c3a9068`); vocabulary `434debe82f949347` from the committed file; the 3,413,760-parameter rotary decoder;
2,400 steps × 8,192 positions, cosine to 0.1, AdamW 6e-4, warmup 200; the same evaluation packs; seed 7; the same code revision
as E2615/E2616 (current `main`). The protocol digest must equal E2614's, which the study checks.

Moves — **the machine, as a bundle**: operating system (Windows 11 → Linux), torch (2.13.0+cpu → 2.15.0+cpu), numpy (2.4.0 → 2.4.6),
Python patch version, CPU model, and intra-op thread count (14 → 4). These are confounded by construction; a difference cannot
be attributed to one of them. Each report records `torch`, `numpy`, Python, platform and `torch_threads`, which is what makes
the bundle readable afterwards.

One difference in the *corpus bytes* is disclosed and not a moved condition of the protocol: `science-pg3489` changed upstream after
the pin, with identical ingested text (K-031, ADR-0047), so the snapshot checksum differs (`3a446f54…` against `236c8d5a…`) and the
corpus fingerprint does not.

## Reference values, fixed now

From the committed [`E2614/results.json`](../E2614/results.json), `per_run["7"].metrics` (Windows, torch 2.13.0+cpu, 14 threads,
4,991.29 s):

| Metric | Reference |
| --- | ---: |
| prose | 1.839519816 |
| science | 1.948623905 |
| technology | 1.902731768 |
| medicine | 1.838141372 |
| mathematics | 2.577769477 |
| whole pack | 2.055447395 |
| **O1, equal-weight mean of the five partitions** | 2.021357268 |

(O1 is computed by the reading script from the five partitions; the number above is that arithmetic, stated so it can be checked.)

## Decision rules, declared before the run

Let Δ be the new run's value minus the reference, per metric. The thresholds are not new: 0.0005 is E2606's declared reproduction
tolerance (`REPRODUCTION_TOLERANCES`, a reporting rule, not an empirical bound), and the tie margins are E2614's 2.77·*s* per metric
(prose 0.00625, science 0.00933, technology 0.00291, medicine 0.01036, mathematics 0.035, whole pack 0.00933, O1 0.0062).

* **X-a — reproduced.** |Δ| ≤ 0.0005 on **every** metric above. The two machines agree at the project's reporting resolution; the
  cross-machine readings in E2615/E2616/E2613 carry no machine effect larger than that, for this configuration.
* **X-b — within seed noise, not reproduced.** Some |Δ| > 0.0005, and every |Δ| ≤ its tie margin. A machine effect exists (or this
  host is not deterministic) but is smaller than one seed's noise on every metric; cross-machine readings are safe only for
  differences larger than the observed |Δ| plus the seed margin.
* **X-c — beyond seed noise.** Any |Δ| > its tie margin. The machine effect on that metric is at least that large. Every comparison
  that subtracts a number from one host from a number from the other is flagged, and the record names which of the existing
  readings it touches: E2615's 0.75 → 1.0 step and its P3 contrast; E2616 and E2613 against T, H and P.

**Contingent step, declared now.** Under X-b or X-c, seed 7 is run once more **on this host** (same command, new experiment directory
`E2617b`), to separate "this host is not deterministic" (the second run differs from the first) from "the machines differ" (the
second run equals the first and both differ from the reference). A deviation between the two runs here is a floor for the seed
margin on this host. Under X-a no second run is made. A crashed run is rerun once; a second failure is reported.

**What may not change after a result is read:** the reference values, the thresholds, the metrics and the outcome names.

## Metrics

Bits per character on the five register partitions and the whole pack (`transformer.test_partitions[prefix].metrics`,
`transformer.test`), plus the equal-weight mean; the environment block of both runs; wall clock and positions per second as observed
conditions, not results. Read by `scripts/read_cross_machine_repeat.py` from the report and the committed reference; never typed in.

## Known limits, declared in advance

* One run on one other machine. The result is about this pair of machines, not about machines in general.
* The bundle of differences cannot be taken apart: a thread-count-only repeat is impossible on a 4-core host (14 threads would
  oversubscribe it and say nothing about the Windows host).
* Seed 7 only. Whether the effect depends on the seed is not tested.
* The reference is E2614's committed per-run summary, not its raw report (the raw reports are not on this host); the comparison is
  therefore on the summarised metrics, which is all the rules use.

## Cost

One run: 2,859–3,090 s per arm was observed on this host for the same configuration (E2615), about **50 minutes**, plus about three
minutes of corpus preparation. A contingent second run adds the same.

## Reproduction

```bash
chakaso experiment study --work-dir data/experiments --experiment E2617 --seeds 7 --split-seed 7 \
  --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2614-decoder-seed-spread.toml --write-report data/experiments/E2617/study.json
python scripts/read_cross_machine_repeat.py --report data/experiments/E2617/seed-7/report.json \
  --reference experiments/E2614/results.json --out experiments/E2617/results.json
```

## Results (added 2026-11-02)

Produced by `scripts/read_cross_machine_repeat.py` from files that are committed: [`E2617-report.json`](E2617-report.json) (the first run), E2614's committed `results.json` as the reference, and [`results.json`](results.json). The same command on the
raw `data/` report gives an identical reading. Both runs: 4 threads, torch 2.15.0+cpu, numpy 2.4.6, Python 3.13.18, Linux 6.18.44-fc, against the reference's 14 threads, torch 2.13.0+cpu, numpy 2.4.0, Python 3.13.5, Windows 11.

| Metric | E2614 seed 7 (Windows) | E2617 (this host) | Δ | Tie margin | Δ as a share of the margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose | 1.839520 | 1.839188 | −0.000332 | 0.00625 | 5 % |
| science | 1.948624 | 1.948785 | +0.000161 | 0.00933 | 2 % |
| technology | 1.902732 | 1.902224 | −0.000508 | 0.00291 | 17 % |
| medicine | 1.838141 | 1.837905 | −0.000236 | 0.01036 | 2 % |
| mathematics | 2.577769 | 2.575317 | −0.002452 | 0.03500 | 7 % |
| whole pack | 2.055447 | 2.055532 | +0.000084 | 0.00933 | 1 % |
| **O1** | 2.021357 | 2.020684 | **−0.000674** | 0.00620 | 11 % |

### The preregistered reading

* **Outcome: X-b.** Three metrics (technology, mathematics, O1) differ by more than the 0.0005 reporting tolerance; **none** is within reach of its tie margin (the largest share is 17 %). The two machines do not agree to the last digits and agree far inside one seed's noise.
* **The contingent step was required and was run.** E2617b is the same command on this host, run as a second seed-7 repeat while other jobs shared the cores (6,217 s against 3,799 s wall). Its report is [`E2617b-report.json`](E2617b-report.json). **Every metric above is identical to the
  last printed digit, the validation curves are equal, and all 43 tensors of the two checkpoints are bit-equal** (the checkpoint files differ in bytes, `cb95186b…` against `c911879d…`, because their metadata carries the experiment name and the run's timings; the report digests differ for the same reason, and the
  protocol digests differ because the experiment name is part of one). So **this host does not fail to reproduce itself**, and the differences in the table are a **machine effect**, not nondeterminism here. Concurrency with other jobs did not change a digit.
* **What the effect is and is not.** It is the sum of everything that differs between the two hosts (operating system, torch 2.13.0 → 2.15.0, numpy, Python patch, thread count 14 → 4, CPU); the experiment cannot take that bundle apart, and nothing here says which part matters. One seed, one pair of machines.

### Consequences for readings that subtract numbers from the two hosts

* The observed machine effect on O1 is **0.00067**, on prose **0.00033**, on mathematics **0.00245** (the largest). Every cross-host comparison in E2613, E2615 and E2616 that was read as a finding is larger than that by at least an order of magnitude: E2615's 0.75 → 1.0 step on O1 is +0.1656, E2616's prose gap to T is 0.0999, the 0 → 0.25 prose step 0.1013.
* The readings that are **within** the effect are the ones that were already read as ties: E2615's P3 contrast (the nested 50 % arm against the natural mixture's five-seed mean: O1 −0.0012, prose +0.0014) is the same size as this effect (O1 −0.0007, prose −0.0003). P3's "document selection does not measurably matter at this resolution" therefore stands, and
  the two possible explanations it could not separate (selection and machine) are now bounded: the machine accounts for at most about half of the O1 difference observed there.
* E2613's H (10M prose, 1.80621) against P (20M prose, 1.80926) differ by 0.00305 on one host pair; that is larger than the machine effect on prose, so it is not explained by it, and it is read as the tie the margin already made it.
* No threshold, margin or outcome of any earlier record changes.

### What this does not establish

A different pair of hosts, a different seed or a longer budget could show a different effect. Floating-point reductions order differently across torch builds and thread counts, so a difference is expected in principle; this measures its size for this protocol once.
