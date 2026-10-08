# Experiments

**No research experiment has been run.** The project has development benchmarks
(retrieval and correction) that produce deterministic metrics over small synthetic
fixtures, and a reproducible experiment layer (`chakaso.experiments`) — but those are
regression instruments, not results: there is still no language model, no real dataset and
no tokenizer, and no experiment testing a research hypothesis has a record committed here.

This directory defines the record format so that the first real experiment is
recorded properly rather than being described in a commit message.

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
- Abandoned experiments stay in the repository. An approach that failed is a result
  and saves the next person from repeating it.
- A result is never reported in documentation without a record to point at.
- If a hypothesis is not being tested, the experiment is exploration. That is
  allowed; it should say so in `Motivation`.
