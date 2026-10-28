# E2610 — is the released checkpoint's repetition in the model or in greedy decoding? A calibrated, preregistered decoding sweep

**Status:** preregistered 2026-10-29, before the sweep ran — results are appended under *Results*
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

*(to be appended after the run)*
