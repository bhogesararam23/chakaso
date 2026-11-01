# E2614 — a decoder seed spread: how large is the noise the 0.02-bit band is supposed to clear?

**Status:** preregistered — committed before any seed ran. No result exists; the Results section is added only
after the reports are on disk.
**Type:** Research experiment, **preregistered**. It takes the place of **E2609**, which E2608 declared (arm A across
five seeds on its 919-document corpus) and nobody ran. *Why a different configuration, stated before any run:* arm A's
corpus is split by chapter, so a spread there would be a spread of within-work numbers (K-023, ADR-0042); E2614 runs
at E2612 arm T's configuration, which is the one every comparison still to come (H8–H13) is measured at. The E2608
reference checkpoint therefore still has no seed spread of its own.
**Protocol file:** [`configs/experiments/E2614-decoder-seed-spread.toml`](../../configs/experiments/E2614-decoder-seed-spread.toml),
read by `chakaso experiment study --protocol`. This record and that file are the same declaration; the file is
what runs.

**Question.** Every decoder record in the repository (E2608, E2611, E2612) judges a difference against a
**0.02 bits-per-character band** that is "inherited from E2611", not measured. If the seed-to-seed standard
deviation of the same configuration is large next to 0.02, then some verdicts already recorded as "outside the
band" are single draws from noise. If it is small, the band is conservative and the verdicts stand. This
experiment measures the quantity the band assumes.

It is a measurement of the training path's variability. It says nothing about capability (K-001).

## What is held fixed, and what moves

Held fixed (one corpus, one split, one vocabulary, one budget — E2612 arm T's configuration, unchanged):

| Condition | Value |
| --- | --- |
| Corpus | `gutenberg-multiregister-v1` 1.0.0, fingerprint `312571f9fa8351a8`; all five registers train |
| Split | whole-work, split seed 7, 312 / 56 / 62 works, zero on more than one side |
| Vocabulary | `434debe82f949347`, 4,096 identifiers, from the committed file `experiments/E2612/vocabulary-434debe82f949347.json` ([ADR-0044](../../docs/decisions/ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md)), checked by fingerprint before any step |
| Model | rotary decoder, 3,413,760 parameters, context 256 |
| Budget | 2,400 steps × 8,192 positions = 19,660,800, cosine to 0.1, AdamW 6e-4, warmup 200 |
| Evaluation | the same packs: 250,000 identifiers per register partition and for the whole pack |

**What moves:** the training seed — and it moves two things together. Chakaso's seed is "the master seed for
initialization and for the batch plan" (`nn_training.py`), so each seed changes **both the initial weights and the
data order**. The spread measured here is the spread over *initialisation and order jointly* at a fixed split and
vocabulary. It is not the spread over splits, tokenizers or corpora, and it is not comparable to a seed spread
that holds the data order fixed (see the [Pythia note](../../docs/internal/engineering/open-model-research-pythia-2026-11-01.md) §2.2).

**Seeds, declared now:** `7, 11, 13, 17, 19` — E2606's list. **Five, not three:** with three runs the 95 % interval
for a standard deviation is 0.52 s to 6.29 s, which cannot decide whether 0.02 sits above noise; with five it is
0.60 s to 2.87 s (computed from chi-square quantiles, assuming approximately normal seed variation, and checked
against the published table by a test). Five is the smallest list for which the interval is narrower than a factor
of three. Seed 7 repeats E2612 arm T and doubles as a determinism check (below).

## Metrics

From each report: bits per character on the whole test pack (`transformer.test`) and on each of the five register
partitions (`transformer.test_partitions[prefix].metrics`), aggregated by `chakaso experiment study` as sample mean,
sample standard deviation *s*, range, and the 95 % interval for σ. Aggregated from the reports on disk
(`--collect-only` rebuilds it); never typed in.

Primary quantities: **prose partition**, **whole pack**, and **mathematics partition** (the partition with the
largest E2612 effect). The other two registers are reported and not decision-bearing.

## Decision rules, declared before any run

A comparison between two single-seed arms is a difference of two draws, whose standard deviation is √2 σ. A 95 %
two-sided band for such a difference is therefore 1.96·√2·σ = **2.77 σ**. Let *s* be a metric's sample standard
deviation over the five seeds and *s*<sub>hi</sub> = 2.874 *s* the upper end of its 95 % interval for σ.

* **R1 — the band is justified for that metric** if 2.77·*s*<sub>hi</sub> ≤ 0.02 (i.e. *s* ≤ 0.0025). Earlier
  "outside the band" verdicts for that metric stand as the single-seed results they were.
* **R2 — plausible, not proven** if 2.77·*s* ≤ 0.02 < 2.77·*s*<sub>hi</sub> (i.e. 0.0025 < *s* ≤ 0.0072). The band
  neither fails nor passes; the record says so and ADR-0041's next unit adds seeds before spending a data arm on a
  difference under 2.77·*s*<sub>hi</sub>.
* **R3 — the band is inside the noise** if 2.77·*s* > 0.02. The 0.02 convention is retired for that metric and
  replaced by **2.77·*s* rounded up to the next 0.005**, and each earlier verdict that used the band is re-read
  against it in this record's Results: E2611's 0.0966 (decoder against transformer, one seed each), E2612's H1
  (−0.27584 technical) and H2 (+0.03026 prose), and E2613's outcome.

Each primary metric is judged separately; a record that finds R1 for the whole pack and R3 for the prose partition
reports both.

**Determinism check (not a hypothesis).** Seed 7 is compared with E2612 arm T by the study's reproduction
comparison, against E2606's declared tolerances (0.0005 bits per character; identity fields exact). *Reproduced* means this
machine and these packages are deterministic for the protocol, and seed 7 is then also a faithful repeat of
E2612-T. *Deviated* means the run-to-run noise at a **fixed** seed is non-zero; the size of the deviation is
reported and is itself a floor for *s*.

No band, seed, metric or flag is changed after a result is read. A seed that fails (crash, out of memory) is
recorded as failed and is rerun once; a second failure is reported, not dropped.

## Known limits, declared in advance

* The interval assumes approximately normal variation. Five points cannot test that; the record shows all five
  values so a reader can.
* The spread is for this configuration at this truncated budget (every decoder run so far stopped at the step
  ceiling with validation still falling). A converged model's spread may be smaller or larger.
* One split. The spread over *splits* (which works are held out) is a separate, probably larger, source of
  variation that this experiment does not measure; the held-out partitions have 1 to 35 works each.
* Sequential on a shared 18-thread CPU host: wall clock is an observed condition, not a result.

## Cost

Five runs at about 90 minutes each on this host: roughly **7.5 hours** of CPU, plus the evaluation of five
checkpoints. Nothing about it is cheap, which is why the plan was five seeds chosen on arithmetic and not three
chosen on convenience.

## Reproduction

```bash
chakaso experiment study --work-dir data/experiments --experiment E2614 \
  --seeds 7,11,13,17,19 --split-seed 7 --expected-corpus-fingerprint 312571f9fa8351a8 \
  --corpus-snapshot data/corpora/gutenberg-multiregister-v1/gutenberg-multiregister-v1-snapshot.json \
  --protocol configs/experiments/E2614-decoder-seed-spread.toml \
  --reference data/experiments/E2612/mixture/report.json --reference-seed 7 \
  --write-report data/experiments/E2614/study.json
chakaso experiment study ... --collect-only    # rebuild the aggregate from the reports, training nothing
```
