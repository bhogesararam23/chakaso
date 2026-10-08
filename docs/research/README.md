# Research

This directory holds the parts of Chakaso that are questions rather than
specifications.

| Document | Contents |
| --- | --- |
| [`research-log.md`](research-log.md) | Dated entries: observations, decisions, and eventually experiments and results |
| [`literature.md`](literature.md) | Sources that informed the initial design, and what each one supports |
| [`hypotheses.md`](hypotheses.md) | Statements the project intends to test, with what would falsify them |
| [`open-questions.md`](open-questions.md) | Questions with no committed answer |

## How entries are written

A research note distinguishes these, and does not blur them:

- **Observation** — something seen, from a run, a reading, or the repository.
- **Hypothesis** — a claim that could be false.
- **Experiment** — a procedure that could distinguish between hypotheses.
- **Result** — what the procedure produced.
- **Interpretation** — what the result is taken to mean.
- **Limitation** — what the result does not establish.
- **Conclusion** — what is now believed, and at what strength.

The reason for the ceremony is that a research log which mixes a hunch with a
measurement is worse than no log, because the hunch acquires the authority of the
measurement over time.

## Standard of evidence

- No result is recorded without the configuration, versions and seed that produced
  it.
- No comparison is recorded without a baseline on the same version.
- No claim of improvement is recorded without the numbers.
- A negative result is a result and is recorded as one.
- If something was not measured, the log says it was not measured.

## Current state

**No experiment testing a hypothesis has been run.** There is still no model, no tokenizer and
no dataset. A local lexical retrieval and a small, synthetic *development* benchmark now exist
(`chakaso.benchmark`, run with `chakaso benchmark`), and they report retrieval numbers over
hand-built fixtures — a development instrument, not a measurement of a hypothesis or of model
quality. Everything else in this directory today is design reasoning and open questions, and it
is labelled as such.
