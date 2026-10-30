# Research

This directory holds the parts of Chakaso that are questions rather than
specifications.

| Document | Contents |
| --- | --- |
| [`research-log.md`](research-log.md) | Dated entries: observations, decisions, and eventually experiments and results |
| [`open-model-landscape.md`](open-model-landscape.md) | A sourced comparison of DeepSeek, Qwen, OLMo, SmolLM and TinyLlama, the scaling and data literature, and what transfers to a CPU-scale model |
| [`chakaso-next-training-design.md`](chakaso-next-training-design.md) | The recommended next model and training strategy: data and tokens before architecture, with the measured evidence and the alternatives rejected |
| [`chakaso-data-curriculum.md`](chakaso-data-curriculum.md) | The staged corpus plan: public-domain prose, technical/scientific text, mathematics, code, licensing, deduplication, mixture weights and contamination controls |
| [`chakaso-capability-benchmarks.md`](chakaso-capability-benchmarks.md) | What each capability claim would require: workloads, scorers, baselines, splits, contamination controls and the necessary-versus-sufficient rule |
| [`chakaso-compute-and-scaling-plan.md`](chakaso-compute-and-scaling-plan.md) | The measured throughput, memory and storage arithmetic for model sizes from 3.4M to 100M parameters, with measured and estimated rows separated |
| [`chakaso-integration-contract.md`](chakaso-integration-contract.md) | The stable Python and serving contract a future consumer may rely on, with the known gaps named |
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

**No experiment testing any of H1–H7 as stated here has been run.** Ten procedures have been executed, and
none is a test of one of these hypotheses: **E-001** compares lexical, dense and hybrid retrieval over the
synthetic development fixtures (it exercises the mechanism and the instrument, not H7 — the dense component is
a non-semantic fixture embedding); **E2601**
([`../../experiments/E2601/record.md`](../../experiments/E2601/record.md)) trained a character model from
random initialization and measured its held-out next-token loss beating uniform/unigram baselines; **E2602**
([`../../experiments/E2602/record.md`](../../experiments/E2602/record.md)) reproduced that across five seeds
under a three-way-split early-stopping protocol (mean test loss 1.80 ± 0.18 nats, below both baselines on
every seed); **E2603** and **E2604** measured retrieval and evidence-sufficiency over a self-authored
grounded-answer workload; **E2605**
([`../../experiments/E2605/record.md`](../../experiments/E2605/record.md)) trained a 643,840-parameter subword
Transformer from random initialization on the repository's own pinned text and measured **1.8657 bits per
character** on held-out documents against **3.1343** for the character architecture and 3.3224 / 4.8010 for the
unigram and uniform predictors of its own identifier space; **E2606**
([`../../experiments/E2606/record.md`](../../experiments/E2606/record.md)) re-ran that protocol across five
pre-declared seeds and put the spread on it — **1.8794 ± 0.0077** against **3.0937 ± 0.0267**, seed 7
reproducing E2605's artifact tensor for tensor; **E2607**
([`../../experiments/E2607/record.md`](../../experiments/E2607/record.md)) moved to externally licensed prose
and measured **2.0176 ± 0.0118** against **3.2218 ± 0.0153** on three Project Gutenberg texts; and **E2608**
([`../../experiments/E2608/record.md`](../../experiments/E2608/record.md)) compared the two Transformer-family
architectures at a matched token budget on a 919-document licensed corpus (**1.6865** for the rotary decoder
against **1.9043** for the transformer, both truncated at the step ceiling, the arms not capacity-matched — a
confound the record states); **E2610**
([`../../experiments/E2610/record.md`](../../experiments/E2610/record.md)) calibrated the released checkpoint's
repetition against human continuations and found it is greedy decoding of a human-diverse distribution, not a
pipeline defect, while no setting produced coherent text; and **E2611**
([`../../experiments/E2611/record.md`](../../experiments/E2611/record.md)) is the capacity- and context-matched
decoder comparison, preregistered before its arm ran.

So: there is a trained model, a learned byte-pair subword tokenizer, a document-level corpus manifest that
pins content digests, four pinned corpora, mini-batch training with validation-selected early stopping, a
checksummed self-describing checkpoint, a Python entry point that loads it, a release contract that one
artifact satisfies at Level 2 of [ADR-0039](../decisions/ADR-0039-three-level-model-release-gate.md), and a
multi-seed spread on the headline comparison. What is still missing is the thing H1–H7 are about: no real
semantic embedding, no second subword algorithm to compare segmentation with (H6 needs that, and the project
has BPE and a character vocabulary, not two subword rules), no usefulness judgement and no retrieval-only
baseline to answer H3, and no measured result from any *useful* language model — the released artifact's
generations repeat phrases with mean bigram repetition 0.5538 and every sample hit its token bound, and both
the card and the record say so. The retrieval, correction and strategy-comparison numbers are development
instruments over hand-built fixtures (`chakaso.benchmark`); the training numbers are model-quality numbers over
five seeds on one corpus, three on a second, and one seed per arm on the third. Everything else in this
directory is design reasoning and open questions, and it is labelled as such.
