# E2614 — a decoder seed spread: how large is the noise the 0.02-bit band is supposed to clear?

**Status:** measured — preregistered and committed before any seed ran; the Results section below was added after
all five reports were on disk. The declarations above it are unchanged.
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

## Results

All five seeds completed (7, 11, 13, 17, 19; none failed, none rerun) on 2026-11-01, 13:17–22:08 IST, sequentially on
a shared CPU host, into `data/experiments/E2614/seed-*/`. The aggregate below is rebuilt from the reports on disk by
`chakaso experiment study … --collect-only`; that rebuild was run and is identical to the study's own aggregate in
every block compared (aggregate, per-run metrics, protocol, reproduction). Machine-readable:
[`results.json`](results.json). Protocol digest `ec6d7f188bd2153df21f63cff155e36a53644765da5b6a673cac08f3533df7d5`.
Wall clock per seed ranged 4,698–8,903 s on a shared host; it is an observed condition, not a result.

### Held-out bits per character, every seed (lower is better)

| Partition | 7 | 11 | 13 | 17 | 19 | mean | s | 95 % interval for σ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **whole pack** | 2.05545 | 2.05873 | 2.05040 | 2.05627 | 2.05850 | 2.05587 | **0.00337** | 0.00202 – 0.00969 |
| **prose** | 1.83952 | 1.83996 | 1.83988 | 1.84314 | 1.83678 | 1.83986 | **0.00226** | 0.00135 – 0.00648 |
| **mathematics** | 2.57777 | 2.59992 | 2.57570 | 2.59410 | 2.60175 | 2.58985 | **0.01232** | 0.00738 – 0.03541 |
| science | 1.94862 | 1.94676 | 1.94944 | 1.95120 | 1.94237 | 1.94768 | 0.00337 | 0.00202 – 0.00968 |
| technology | 1.90273 | 1.90215 | 1.90263 | 1.90328 | 1.90487 | 1.90313 | 0.00105 | 0.00063 – 0.00302 |
| medicine | 1.83814 | 1.82899 | 1.83133 | 1.83624 | 1.83212 | 1.83336 | 0.00374 | 0.00224 – 0.01075 |

Every run stopped at the 2,400-step ceiling (`best_step` 2,400 in all five). Baselines are identical across seeds, as
they must be: uniform 5.71607, unigram 3.50295.

### Decision rules applied, per primary metric

| Metric | 2.77·s | 2.77·s<sub>hi</sub> | Rule | Reading |
| --- | ---: | ---: | --- | --- |
| prose | 0.00625 | 0.01795 | **R1** | the 0.02 band is justified for prose: even the upper end of the interval stays under it |
| whole pack | 0.00934 | 0.02683 | **R2** | plausible, not proven: 0.02 clears the central estimate and not the upper end |
| mathematics | 0.03413 | 0.09808 | **R3** | the band is inside the noise: replaced for this metric by **0.035** (2.77·s = 0.0341 rounded up to the next 0.005) |

The three registers that are not decision-bearing, reported and not interpreted: science R2, technology R1,
medicine R2.

**Determinism check: reproduced, exactly.** Seed 7 against E2612 arm T, by the study's reproduction comparison and
E2606's tolerances: every identity field matches (corpus, split, vocabulary, parameter count, best step), and the
difference is **0.0 on every metric** (2.05544739454246 against 2.05544739454246), not merely inside the 0.0005
tolerance. This machine and these packages are deterministic for the protocol at a fixed seed, so the spread above has
no fixed-seed jitter in it: it is entirely the effect of the seed's two parts (initial weights and data order). Seed 7
is also a faithful repeat of E2612-T, and a typical draw: second-lowest of five on the whole pack, 0.0004 under the
five-seed mean on prose.

### Earlier verdicts re-read against the measured spread

R3 requires this for the metric it retires (mathematics). It is done here for every earlier verdict that used the
band, with the measured s of the matching partition. **A transfer is assumed in all of them:** the spread was
measured for E2612 arm T's configuration, split and vocabulary, and is applied to differences measured at the same
split and vocabulary but for other arms. E2611 was run on a different corpus and split.

| Earlier verdict | Difference | Against the measured noise | Reading |
| --- | ---: | --- | --- |
| E2612 H2, prose cost of the mixture (T − P) | +0.03026 | 1.69 × 2.77·s<sub>hi</sub>, 4.84 × 2.77·s | **stands**: outside noise even at the interval's upper end |
| E2612 H1, technical loss falls (T − P) | science −0.11005, technology −0.28374, medicine −0.15708, mathematics −0.46584 | 4.1×, 33.9×, 5.3×, 4.8× the matching 2.77·s<sub>hi</sub> | **stands**, per partition. The weighted technical aggregate's own spread was not computed, so its −0.27584 is judged through its parts |
| E2613, prose-only H against the mixture T | −0.03331 | 1.86 × 2.77·s<sub>hi</sub>, 5.33 × 2.77·s | **stands**: H8-b is outside the noise of the arm it is compared with |
| E2613, H against the full prose control P | −0.00305 | 0.17 × 2.77·s<sub>hi</sub> | **stands**: indistinguishable from no difference |
| E2613, H against P on mathematics (reported there, not interpreted) | +0.06499 | 1.90 × 2.77·s, 0.66 × 2.77·s<sub>hi</sub> | **cannot be separated from noise** on the replacement band; E2613's "mathematics carries it" reading is withdrawn |
| E2611, decoder against transformer, whole pack | 0.0966 | 3.60 × 2.77·s<sub>hi</sub>, 10.3 × 2.77·s | would stand if the spread transferred to E2611's corpus and split; **that is not measured** |

Two cautions the table does not remove. The spread was measured on arm T, the mixture; the prose-only arms (P, H) were
run once each and their own spread is unmeasured, so the E2613 comparisons assume theirs resembles T's. And a verdict
that survives is a verdict about the *training path's* noise at one split: the spread over **which works are held
out** is a separate and probably larger source (the partitions have 1 to 35 works), and this experiment did not
measure it.

### What this changes

* **The 0.02 convention is no longer a guess for prose, and is too small for mathematics.** For the prose partition,
  where the H8/H9 data questions are decided, it is conservative. For mathematics, a one-seed difference under 0.035
  is not evidence. For the whole pack, a one-seed difference under 0.0268 (2.77·s<sub>hi</sub>) is not yet
  separable from the upper end of the interval, and rule R2 applies: ADR-0041's next unit adds seeds before spending a
  data arm on a difference that small.
* **The replacement band is applied to mathematics only,** as R3 declares. The whole-pack figures above (0.0093
  central, 0.0268 upper) are what R2 means numerically; they are not a new convention.
* **H8's verdict is unchanged,** and so is its exposure caveat: noise is not what limits it.
* **H16 (an optimiser change)** is gated on "more than the seed noise E2614 measures", and that noise now exists: a
  single-seed whole-pack gain would have to exceed 0.0268 to be separable at the interval's upper end.

### What this does not establish

* Not that any model is better or worse: this is the variability of one configuration's training, and K-001 stands.
* Five values cannot test the normality the interval assumes; all five are in the table for a reader to judge. The
  mathematics spread (range 0.026 over five seeds) is large next to the others, and that partition is small.
* The spread is for a **truncated** budget. A converged model's spread may be smaller or larger.
* Not the spread over splits, tokenizers or corpora, and not the spread with the data order held fixed: initialisation
  and order moved together.

### Artifacts

| Artifact | Checksum recorded in its report (`transformer.artifact.checksum`) | SHA-256 of the file |
| --- | --- | --- |
| Checkpoint, seed 7 (`E2614-decoder.pt`) | `fd464c4241669d6d4b1fd6b3a1ab39988f3427180b74c98c91eeec4ff7f0fc9f` | `4b7135aeb8f26a8cbaf67817f6b9e04300e166281bdb685d5d304a9b1d4f305e` |
| Checkpoint, seed 11 | `5afda47cc00dc80b1227491ab5e3dab488ed386b84ba22d56b2c75c331c4d44b` | `029f32591d59876e9b45e2468b67aaff2770677c2640b158262f4feef20da878` |
| Checkpoint, seed 13 | `acf81fe204c8bcb5b72a82a203d20779da5798e4fc3d916479f8c70d81e02804` | `955e6e242d14890f0d29dec4eabcf6caf3d22f0187517e56f9af71d8dc1e8a98` |
| Checkpoint, seed 17 | `38b1448176b963d7245921558de62c1a6d9d4f3f828106c17d3c9eff834f54b2` | `cf66c719ca8e491993f495f5406b33289222e288e6f5ef4a1d5986d9693b5118` |
| Checkpoint, seed 19 | `be5bfdc945ac9c29ac455c3e2b001e7d5f9195a4ad4a35b7a5f1e71dbd3e2e0e` | `146fa95de1e6cb899c1a9cf91fb36d5dacb54864661850654f24f0ac5b62d06a` |
| Aggregate, committed [`results.json`](results.json) | rebuilt from the five reports | study report digest `3e1f0473d15a786bdc74a1842f10413b8399043c2ffe209c8e40acab05546ffb` |

The first column is the checksum the report records for the saved artifact; the second is the SHA-256 of the file on
disk, computed after the run. They are different quantities and are not expected to be equal. As K-028 records, a
report file's digest embeds timings and does not reproduce; the protocol digest does.
