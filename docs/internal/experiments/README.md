# Experiments

**Ten experiment records are committed: E2601, E2602, E2603, E2604, E2605, E2606, E2607, E2608, E2610 and
E2611. Every one was written and committed before its runs began, and every one reports a measured result.**
E2601
([`../../../experiments/E2601/record.md`](../../../experiments/E2601/record.md)) is the project's first
measured result — a from-scratch character model whose held-out next-token loss beats uniform and unigram
baselines; E2602
([`../../../experiments/E2602/record.md`](../../../experiments/E2602/record.md)) reproduces it across five
seeds under a proper three-way-split early-stopping protocol (mean test loss 1.80 ± 0.18 nats, below both
baselines on every seed); E2603
([`../../../experiments/E2603/record.md`](../../../experiments/E2603/record.md)) is the first
**grounded-answer evaluation** baseline, run over the self-authored P27 workload (`chakaso benchmark
grounded`) with a reference oracle — measuring the harness and retrieval, and honestly reporting real
retrieval failures (forbidden hits, false retrievals), not any model capability; E2604
([`../../../experiments/E2604/record.md`](../../../experiments/E2604/record.md)) evaluates the P28
evidence-sufficiency decision (ADR-0032) against the E2603 baseline — 100% accurate on the workload (all
unanswerable cases caught, zero answerable cases wrongly abstained) with a documented narrow,
corpus-specific threshold band; E2605
([`../../../experiments/E2605/record.md`](../../../experiments/E2605/record.md)) measures the subword
Transformer against the character architecture on 71 pinned documents of this repository's own prose —
1.8657 bits per character against 3.1343, 3.3224 unigram and 4.8010 uniform, one seed, which E2606 then
showed was the best of five draws; E2606
([`../../../experiments/E2606/record.md`](../../../experiments/E2606/record.md)) runs the identical protocol
over five pre-declared seeds with the split held fixed — **1.8794 ± 0.0077** bits per character against
**3.0937 ± 0.0267**, every seed under every baseline, the seed-7 artifact reproducing E2605 tensor for tensor
from a corpus recovered out of git history; E2607
([`../../../experiments/E2607/record.md`](../../../experiments/E2607/record.md)) repeats it on 105 documents of
externally licensed Project Gutenberg prose — **2.0176 ± 0.0118** against **3.2218 ± 0.0153**, with the caveat
that no run early-stopped and all three selected the step ceiling, making that level a floor; and E2608
([`../../../experiments/E2608/record.md`](../../../experiments/E2608/record.md)) compares the two
Transformer-family architectures at a matched token budget on 919 licensed documents — the 3,413,760-parameter
rotary decoder at **1.6865** bits per character against the 1,037,056-parameter transformer at **1.9043**, both
truncated, both arms' generations measured for repetition (zero tail loops, zero periodic runs of three or more,
mean bigram repetition 0.5538) — and it is the record whose arm A became `chakaso-base@0.1.0`; E2610
([`../../../experiments/E2610/record.md`](../../../experiments/E2610/record.md)) calibrates that checkpoint's
repetition against human continuations and finds it is greedy decoding of a human-diverse distribution, not a
pipeline defect, while no decoding setting produces coherent text; and E2611
([`../../../experiments/E2611/record.md`](../../../experiments/E2611/record.md)) is the capacity- and
context-matched comparison E2608 could not make — a 1,018,240-parameter decoder reaches **1.8077** bits per
character against the 1,037,056-parameter transformer's **1.9043**, a 0.0966-bit bundle effect, both arms still
at the 2,400-step ceiling.

No *research hypothesis* (H1–H7 in [`../../research/hypotheses.md`](../../research/hypotheses.md)) has been
tested by any of them: E2608's H1/H2/H3 are that experiment's own preregistered predictions about schedule,
architecture and degeneration, not the project's hypotheses about grounding and correction. None of the ten
establishes a research claim about grounding, citation or correction on real language; E2605–E2608 establish
next-token results on pinned text, not capability. The project also has development benchmarks (retrieval and
correction) and a reproducible experiment layer (`chakaso.experiments`) — those are regression instruments over
synthetic fixtures, distinct from a committed experiment record.

Preregistration is what the word "record" means here: seeds, primary metric, budget, aggregation method,
failure policy and success/failure interpretations are fixed in the file before a run, so a result cannot be
sorted into a story afterwards. Where a record got a number wrong against its own aggregate, the correction is
a dated table inside the record that keeps the replaced wording — see E2606's and E2607's *Corrections applied
after the run* sections, found by the 2026-10-22 documentation audit and committed 2026-10-24.

This directory defines the record format so that an experiment is recorded properly rather than being
described in a commit message.

## Where records live

An experiment record is a small text file, committed, at:

```text
experiments/<experiment-id>/record.md
```

with any generated configuration alongside it. Large outputs — checkpoints, logs,
indexes — are never committed ([artifact policy](../engineering/artifact-policy.md)).

Experiment IDs are `E<year><sequence>`, for example `E2601`. They are never
reused, and an experiment that is re-run with a change is a new experiment, not an
edited one.

## Record format

```markdown
# E2601 — <title>

- Status: planned | running | completed | abandoned
- Date: YYYY-MM-DD
- Hypothesis: link to docs/research/hypotheses.md#H1
- Owner:
- Git commit:
- Config hash:

## Motivation

Why this experiment, and what decision it informs. An experiment that would not
change any decision should not be run.

## Baseline

What the result will be compared against, measured on the same data and version.
"Compared to nothing" is not a baseline.

## Configuration

The full resolved configuration, or a hash plus a path to the committed file.
Anything not recorded here cannot be reconstructed later.

## Data

Dataset name, version, source, licence, and the exact split used. Contamination
checks performed.

## Model

Architecture, parameter count, initialization (random or from a checkpoint with its
identifier), and tokenizer version.

## Environment

Hardware, operating system, Python version, dependency versions, and whether
accelerators were used.

## Reproducibility

Seed(s), and the exact command. The test is whether someone else could obtain the
same numbers.

## Metrics

What is measured, how it is computed, and the code that computes it. A metric with
no implementation is a number someone typed.

## Results

The measured values. If a number was not measured, it does not appear in this
section.

## Interpretation

What the results are taken to mean. Separated from the results themselves, because
this is where the reasoning is most likely to be wrong.

## Limitations

What the experiment does not establish: sample size, single seed, one dataset, one
hardware configuration, an evaluation set the author also wrote.

## Conclusion

What is now believed, and at what strength.

## Follow-up

The next experiment, or the decision this settles.
```

## Rules

- A record is written **before** the run, with status `running`. A record written
  afterwards is a summary, and summaries omit the things that went wrong.
- A record whose result is measured on tracked text names the **revision** its corpus came from, and the run
  trains from the snapshot recovered out of it ([ADR-0036](../../decisions/ADR-0036-corpus-snapshot-recovery-from-history.md)).
  A fingerprint alone identifies a corpus; it does not reproduce its bytes, and a documentation edit after the
  pin makes the recorded text unreachable from the working tree.
- An experiment that is re-run with a change is a **new experiment id**. A reproduction is published as its
  own result with its own deviations, never as an edit to the number it is reproducing.
- Abandoned experiments stay in the repository. An approach that failed is a result
  and saves the next person from repeating it.
- A result is never reported in documentation without a record to point at.
- If a hypothesis is not being tested, the experiment is exploration. That is
  allowed; it should say so in `Motivation`.
