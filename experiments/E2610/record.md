# E2610 — is the released checkpoint's repetition in the model or in greedy decoding? A calibrated, preregistered decoding sweep

**Status:** completed 2026-10-29 — preregistered in `132839b` before the sweep ran; results below
**Type:** Research experiment over **existing artifacts** (no training)
**Code:** `chakaso.training.quality_eval` at the commit that carries this file; `chakaso model quality`
**Depends on:** E2608 (the checkpoints), ADR-0039 (the release gate), the diagnosis recorded in
`docs/research/research-log.md` (2026-10-29)

## Motivation

`chakaso-base@0.1.0` is withheld at Level 3 on one number: mean *character*-bigram repetition 0.5538 over five
greedy continuations. A diagnosis done before this record found that number cannot be read alone — the same
statistic on real held-out prose of the same length is 0.3355 — and that the model's unconstrained samples
(temperature 1.0) match human word diversity almost exactly while its greedy continuations do not. That
points at **greedy decoding of a high-entropy model**, not at a defect in data, packing, tokenization,
attention masking, positions or the key/value cache, each of which was checked (see the research log). It is a
diagnosis from one exploratory run, however, and this record exists to make it a measured, gated claim — or to
fail.

## What I had already seen when this was written (disclosure)

An exploratory script (not committed; `.scratch/diag_quality.py`) and the first baseline run of the suite had
already been read before this record was written. Specifically, I had seen, on 24 prompts drawn from held-out
documents with `prompt_offset 0`: greedy fails the diversity gates; `greedy_rp1.2` is over-diverse; T = 1.0
passes. **The gate values (bands) were therefore chosen knowing the model's rough behaviour on those 24
prompts**, and they are not "blind". What is blind: the **confirmation set** below uses held-out documents
(`prompt_offset 24`) that no run, exploratory or recorded, has generated from; and every sweep configuration
other than the five in the default set has never been run.

## Hypotheses, stated before the sweep

- **H1 (the repetition is a decoding property).** At least one *sampled* configuration of the declared sweep
  passes **every** gate on the tuning set and, selected by the rule below, again passes **every** gate on the
  confirmation set.
- **H2 (a repetition penalty disguises rather than repairs).** Every `greedy_rp*` configuration either still
  fails the periodic/diversity gates, or passes only by exceeding the human diversity band (`over_diverse`).
  A penalty that makes diagnostics pass by overshooting is reported as such, not as a fix.
- **H3 (architecture).** Under identical prompts and identical greedy decoding, the E2608 decoder arm has
  higher mean word distinct-1 than the E2608 transformer arm. (E2608 left this as an unmeasured impression.)

## Data, prompts and split

The model's own held-out documents: the **test side** of `gutenberg-english-prose-v2` (corpus fingerprint
`e9fc2e92f59d88ea`, 138 documents, split seed 7, the pinned snapshot, verified by `read_snapshot`). Prompts are
drawn by `chakaso-heldout-prompts/1`: one per document, a ~240-character passage cut at word boundaries, plus
the 64 tokens that really followed it as the human reference. Licence and provenance: as E2608 (public domain
in the United States, Project Gutenberg; see the data card).

| Set | `--prompt-offset` | Prompts | Used for |
| --- | --- | --- | --- |
| tuning | 0 | 24 | choosing a configuration (documents already seen in the baseline run) |
| confirmation | 24 | 24 | judging the chosen configuration (documents never generated from) |

The two sets share no document (guaranteed by construction and tested).

## Procedure and metrics

Every configuration of `SWEEP_CONFIGS` (19: greedy; greedy with repeat penalty 1.05/1.1/1.2; temperature
0.5/0.7/0.8/0.9/1.0 each with no nucleus, top-p 0.9 and top-p 0.95) over each set, 64 new tokens per sample,
sampled configurations over seeds 11, 23 and 37, greedy once. The **gates** are those in
`chakaso.training.quality_eval.GATES`, fixed in code before this run: mean word distinct-1/2/3 within
±0.05/±0.03/±0.02 of the human reference on the same prompts (two-sided, so *over*-diverse text also fails),
tail-loop fraction 0.00, periodic-run (≥3 repeats) fraction ≤ 0.05, zero U+FFFD characters. Loops are measured
on whitespace-collapsed text, a correction made after the first baseline run flagged a trailing `\n     ` as a
loop; it applies equally to the human reference.

**Selection rule (fixed now).** From the tuning set, take the configurations that pass all gates; of those,
select the one with the **lowest temperature**, breaking ties by the presence of a nucleus (top-p) and then
by name. Rationale: the least-flattened distribution that still behaves like prose. If none passes, nothing
is selected and H1 fails on the tuning set.

## Success, inconclusive and failure

- **H1 success:** the selected configuration also passes all gates on the confirmation set. **Inconclusive:**
  it passes on tuning only (a tuning-set effect; reported as such). **Failure:** none passes tuning.
- **H2 success:** as stated. **Failure:** a penalty configuration passes every gate with every distinct-n inside
  its human band — which would be reported as a genuine repair, with raw samples.
- **H3 success:** decoder arm's greedy distinct-1 exceeds the transformer arm's on the same 24 tuning prompts.
  **Failure:** it does not. The transformer arm runs only the default configuration set; its context is 128, so
  prompts longer than that are right-clipped by the boundary, stated in the results.
- No result here is evidence of coherence, truth or usefulness. The gates are **necessary, not sufficient**: a
  configuration can pass all of them and still write fluent nonsense. Level 3 stays withheld unless a human or a
  semantic judge — neither exists here — says otherwise.

## Failure and missing-run policy

A run that raises is reported with its error and is not aggregated. Reports are written atomically and refuse
to replace each other without `--overwrite`. The sweep is not repeated with different gates after seeing the
results; a different gate is a new record.

## Known limitations and threats to validity

1. **24 prompts per set.** For a sampled configuration that is 72 samples (3 seeds × 24 prompts) that are not
   independent (same 24 prompts). Distinct-n means have a standard deviation in the report; no confidence
   interval is claimed. A pass is a pass on this sample, not a guarantee.
2. **Gate bands were set knowing the first baseline** (see disclosure). The confirmation set bounds that
   optimism; it does not remove it.
3. **Surface statistics only.** A configuration that passes may still be incoherent. The suite has no
   semantic judge.
4. **One checkpoint, one seed, one corpus.** Nothing here speaks to other training runs.
5. **Short outputs.** 64 tokens is ≈ 140 characters; loops that need longer to develop are not seen.
6. **Whitespace collapsing** can hide a genuine loop of newlines or indentation (e.g. the E2608 pilot's 24
   space characters). That case is caught by the diversity gate and by `bound_fraction` but not by the loop
   gates.

## Commands

```bash
# baseline, default set, tuning prompts (already run once to establish the baseline file)
chakaso model quality data/experiments/E2608/release \
  --snapshot data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json --split-seed 7 \
  --config-set sweep --prompts 24 --prompt-offset 0 --out data/experiments/E2610/sweep-tuning.json
chakaso model quality data/experiments/E2608/release \
  --snapshot data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json --split-seed 7 \
  --config-set sweep --prompts 24 --prompt-offset 24 --out data/experiments/E2610/sweep-confirmation.json
# H3: the transformer arm, same prompts, default set
chakaso model quality data/experiments/E2608/transformer/E2608-transformer.pt \
  --snapshot data/experiments/E2608/gutenberg-english-prose-v2-snapshot.json --split-seed 7 \
  --prompts 24 --prompt-offset 0 --out data/experiments/E2610/transformer-default.json
```

Reports (`data/`) are git-ignored; their digests and the figures cited from them are recorded below.

## Results

All runs: Windows 11, Python 3.13.5, torch 2.13.0+cpu, CPU only, the release directory
`data/experiments/E2608/release` (checkpoint SHA-256 `9e832fadf277e848…`), snapshot checksum as in E2608.
Per-configuration aggregates and gate verdicts for every report are committed in
[`results.json`](results.json), with each source report's SHA-256. Source reports: tuning `485c205e076f04ad…`,
confirmation `114a62aa396cb53a…`, transformer arm `5e1fc96ad12199df…`, baseline `cfa4d3382b6b0229…`.

**The human reference** (the real following 64 tokens of the same passages): word distinct-1/2/3 =
0.853 / 0.969 / 0.982 on the tuning prompts (digest `c3614be7182e`) and 0.877 / 0.992 / 1.000 on the confirmation
prompts (digest `269ff2a2d034`). The two sets differ by 0.024 in distinct-1 for *human* text alone, which is the
size of the noise a 24-prompt set carries and the reason the bands are only 0.05 wide.

**Decoder, greedy** (distinct-1 / -2 / -3; tail-loop fraction; periodic fraction):
tuning 0.505 / 0.688 / 0.794; 0.00; 0.08 — confirmation 0.525 / 0.714 / 0.805; 0.04; 0.04. **Fails the three
diversity gates on both sets**, by 0.35 in distinct-1.

**Sampled, selected rule applied.** On tuning, the passing configurations were `t0.8`, `t0.8_p0.95`, `t0.9`,
`t0.9_p0.9`, `t0.9_p0.95`, `t1.0`, `t1.0_p0.9`, `t1.0_p0.95`. The lowest temperature is 0.8; the tie between `t0.8` and
`t0.8_p0.95` goes to the nucleus, so the **selected configuration is `t0.8_p0.95`**.
On confirmation it **fails** on distinct-1: 0.818 against a human 0.877, gap −0.059 (band 0.05); distinct-2/3,
loops, periodic and U+FFFD pass. On confirmation the passing sampled configurations were `t0.8`, `t0.9`,
`t0.9_p0.9`, `t0.9_p0.95`, `t1.0`, `t1.0_p0.9`, `t1.0_p0.95`; `t0.9` and above passed on **both** sets,
`t0.8` passed both, `t0.8_p0.95` and `t0.8_p0.9` did not.

**Repetition penalty, greedy.** `greedy_rp1.05`: fails both sets (tuning: one tail loop, fraction 0.04;
confirmation: distinct-1/2 and periodic). `greedy_rp1.1`: **fails tuning** (periodic fraction 0.17) and **passes
every gate on confirmation** (distinct 0.906 / 0.995 / 0.997 against 0.877 / 0.992 / 1.000, periodic 0.04).
`greedy_rp1.2`: distinct-1 0.972 / 0.970 — over-diverse on both, and distinct-2 = distinct-3 = 1.000 (it never
repeats a bigram), failing the distinct-1 band on both sets and distinct-2 on tuning.

**Transformer arm, same 24 tuning prompts, default set** (digest `c3614be7182e`, the same prompts the decoder
was run on): greedy distinct-1 / -2 / -3 = **0.358 / 0.506 / 0.635**, tail-loop 0.04, periodic **0.29**; the
decoder arm's greedy on those prompts is 0.505 / 0.688 / 0.794, tail-loop 0.00, periodic 0.08. Its unconstrained
T = 1.0 sampling has distinct-1 0.895 (decoder 0.890) and 2 U+FFFD characters (decoder 0).

**Raw samples, because the ratios are not the finding.** One confirmation prompt (index 3), the same prompt
in every row, continuing *"…and to be righted when injury is"* (`pg3207`, Hobbes's *Leviathan*), the human text being *"done them; yet
because no man that considers, can make question of it…"*:

| Configuration | First 200 characters |
| --- | --- |
| greedy | ` to be required to the\nSoveraign Power. For if it be not to be the same to be in the same\nplace, is to be the same to be in ` |
| greedy_rp1.1 (passes every gate here) | ` to be required, that\nthey are, by the same.\n\nTheirsiteth, is,--\n\nTheiring,--\n\nAndyssethly:\n\nAndy-eatingly;\n  Forcepted;\n  ` |
| t0.8_p0.95 (selected on tuning) | ` of a Christian Manners of\nthe Lords; but that which is the cause of a Law an end, is to be\nuncrept to the Word of the present,` |
| t0.9 (passes both sets) | ` known. The Meshiciety of\nReligion; that is therefore a man who doth all that, hath no Law but\nKittings. For as if a man that comm` |
| t1.0 (passes both sets) | ` Lawnies a saving, and finish\nPuhaberlap, in the same account, for other partstayons.\n\nBut in Actour, Dictionary\n\nIts are well the room,` |

A reader will see at once what the gates cannot: the rows that pass them contain invented words
(`Meshiciety`, `Puhaberlap`, `Theirsiteth`) and no argument.

## Interpretation — which hypothesis held

- **H1 — inconclusive, by the rule written for it.** The configuration the preregistered rule selected passed
  tuning and failed confirmation by 0.009 outside a band. The weaker, unpreregistered observation — that
  every configuration at T ≥ 0.9 and `t0.8` itself passed both sets — supports the same direction but
  is not the test that was declared, and is reported as an observation, not a success. What the sweep does
  establish is a pattern across *all nineteen* configurations: distinct-1 climbs monotonically with temperature
  in the un-nucleused rows (0.70, 0.77, 0.83, 0.85, 0.89 on tuning at T = 0.5–1.0) and reaches the human value
  near T ≈ 0.8–1.0, i.e. **the model's own distribution, undistorted, has human-level
  word diversity.** That is evidence against "the model's distribution is degenerate" and for "greedy decoding
  of this model is".
- **H2 — failed as worded; the substance survives.** A penalty (`greedy_rp1.1`) did pass every gate on one set
  with distinct-n inside the band, so the statement "every penalty fails or overshoots" is false. But it failed
  tuning, and its passing sample is `Theirsiteth, is,--  Theiring,--`. The repetition penalty trades the
  *repetition* gates for incoherent text, which these gates do not penalise. This is exactly the disguise the
  roadmap warned about, now with a sample attached; it is not claimed as a hypothesis result.
- **H3 — supported on diversity.** On identical prompts the decoder arm's greedy output is markedly less
  repetitive than the transformer arm's (distinct-1 0.505 against 0.358, periodic fraction 0.08 against 0.29),
  consistent with E2608's loss ordering. One seed per arm, one checkpoint each, 24 prompts: a direction, not a
  magnitude. The transformer arm has context 128 and its prompts were clipped to it by the boundary.
- **The headline.** The released model's repetition is **a property of greedy decoding of a distribution that
  is itself human-diverse, not a defect of the pipeline** (the earlier diagnosis found none), and *fixing the
  surface statistics is easy and does not fix the text.* No decoding setting turned this 3.4M-parameter model
  into a coherent writer, and no gate in this suite measures coherence. Level 3 stays **not met**, now on a
  stronger and more honest statement than "0.5538 character-bigram repetition" — which, calibrated, was
  0.55 against a human 0.34 on a statistic that mostly measures how few distinct letters English has.

## Deviations from the preregistration

1. **The zero-U+FFFD gate penalises a bound-truncated character.** One transformer-arm T = 1.0 sample ends in a
   character cut by the 64-token bound (the decoder correctly flushes it as one replacement character); a
   second contains a replacement mid-text, which is the model emitting a byte sequence that is not valid UTF-8.
   The gate does not distinguish them. It was **not** changed after the run; a gate that separates "flushed at
   the bound" from "emitted mid-text" is a follow-up and a new record.
2. **The first baseline run, whose report is `cfa4d3382b6b0229`, was re-run after a metric correction.** The
   first run flagged a sampled continuation ending in `\n     ` as a tail loop; loop detection now collapses
   whitespace. The disclosure above names this; the baseline file cited here is the corrected run. The earlier
   run's gate verdicts were otherwise identical.
3. **The confirmation-set statement "documents never generated from" is exact for documents, not for the
   corpus.** The model trained on other documents of the same authors and books (the split is by document, not
   by work), so held-out here means *unseen document*, not *unseen work*.
