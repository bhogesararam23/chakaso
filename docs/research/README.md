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

**No experiment testing any of H1–H7 has been run.** One experiment has been executed —
E-001 in [`research-log.md`](research-log.md) compares lexical, dense and hybrid retrieval over the
synthetic development fixtures — but it exercises the *mechanism and the measuring instrument*, not a
hypothesis: the dense component is an explicitly non-semantic fixture embedding, so E-001 neither
supports nor refutes H7 (it showed no gain and a small MRR regression). There is still no model, no
tokenizer, no dataset and no real embedding. The retrieval, correction and strategy-comparison numbers
are development instruments over hand-built fixtures (`chakaso.benchmark`, run with
`chakaso benchmark`), not measurements of a hypothesis or of model quality. Everything else in this
directory today is design reasoning and open questions, and it is labelled as such.
