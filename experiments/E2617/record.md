# E2617 — a cross-machine repeat: does E2614's seed 7 reproduce on a different host?

**Status:** preregistered, not yet run. Committed before the run starts; the Results section is added after the report is on disk.
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
(prose 0.00625, science 0.00933, technology 0.00291, medicine 0.01036, mathematics 0.035, whole pack 0.00934, O1 0.0062).

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
