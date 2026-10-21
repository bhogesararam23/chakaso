# Research

This directory holds the parts of Chakaso that are questions rather than
specifications.

| Document | Contents |
| --- | --- |
| [`research-log.md`](research-log.md) | Dated entries: observations, decisions, and eventually experiments and results |
| [`literature.md`](literature.md) | Sources that informed the initial design, and what each one supports |
| [`hypotheses.md`](hypotheses.md) | Statements the project intends to test, with what would falsify them |
| [`model-feasibility.md`](model-feasibility.md) | An evidence-bounded assessment of putting a real model behind the boundary (a runnable local model vs a from-scratch training experiment); no experiment run |
| [`next-model-experiment.md`](next-model-experiment.md) | The measured re-inspection of machine, dependencies, compute cost and usable data behind the first model-centred experiment, and why the subword Transformer on repository text was chosen; it holds no result, but where it estimated it now shows what E2605 measured |
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

**No experiment testing any of H1–H7 has been run.** Six procedures have been executed, and none is a
hypothesis test: **E-001** compares lexical, dense and hybrid retrieval over the synthetic development
fixtures (it exercises the mechanism and the instrument, not a hypothesis — the dense component is a
non-semantic fixture embedding, so it neither supports nor refutes H7); **E2601**
([`../../experiments/E2601/record.md`](../../experiments/E2601/record.md)) trained a character model from
random initialization and measured its held-out next-token loss beating uniform/unigram baselines; **E2602**
([`../../experiments/E2602/record.md`](../../experiments/E2602/record.md)) reproduced that across five seeds
under a three-way-split early-stopping protocol (mean test loss 1.80 ± 0.18 nats, below both baselines on
every seed); **E2603** and **E2604** measured retrieval and evidence-sufficiency over a self-authored
grounded-answer workload; and **E2605**
([`../../experiments/E2605/record.md`](../../experiments/E2605/record.md)) trained a 643,840-parameter
subword Transformer from random initialization on the repository's own pinned text and measured **1.8657 bits
per character** on held-out documents against **3.1343** for the character architecture and 3.3224 / 4.8010
for the unigram and uniform predictors of its own identifier space (one seed, one author's prose).

So: there is a trained model, a learned byte-pair subword tokenizer, a document-level corpus manifest that
pins content digests, mini-batch training with validation-selected early stopping, a checksummed
self-describing checkpoint and a Python entry point that loads it. There is still no real embedding, no
external corpus, no measured result from any *useful* language model, and no error bar on E2605 — the
committed generations loop, and the record says so. The retrieval, correction and strategy-comparison numbers
are development instruments over hand-built fixtures (`chakaso.benchmark`); the training numbers are
single-corpus, single-seed. Everything else in this directory is design reasoning and open questions, and it
is labelled as such.
