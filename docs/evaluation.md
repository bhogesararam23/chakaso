# Evaluation

Chakaso is evaluated as a system, not only as a language model. A good
next-token loss says nothing about whether retrieval found the right evidence,
whether citations hold, whether follow-ups resolve, or whether the system revises
an answer when it should.

This document defines the layers, the metrics, the gates, and the standard the
project holds itself to when reporting a result.

## Layers

| Layer | What it measures | Examples |
| --- | --- | --- |
| Language model | Whether the model learned language | Loss, perplexity, next-token accuracy |
| Knowledge | Whether it knows facts | Standard QA and factuality tasks |
| Retrieval | Whether the right evidence was found | Recall@k, precision@k, MRR, NDCG |
| Grounding | Whether claims are supported by cited evidence | Grounded claim support rate |
| Citation | Whether references are correct and complete | Citation precision, citation recall |
| Conversation | Whether follow-ups resolve | Follow-up resolution rate |
| Correction | Whether revision happens when it should | Correction success rate, unjustified persistence rate |
| Uncertainty | Whether abstention is appropriate | Appropriate abstention rate, calibration |
| System | Whether it is usable | Latency, memory, failure rate, reproducibility |
| Regression | Whether anything got worse | Protected suites must not degrade |

## Metrics

Each metric is defined so it can be computed by a program, not judged by
impression.

| Metric | Definition |
| --- | --- |
| Grounded Claim Support Rate | Fraction of answer claims supported by the cited evidence, on cases where evidence was required |
| Citation Precision | Fraction of cited sources that actually support the associated claim |
| Citation Recall | Fraction of externally verifiable claims that carry appropriate source support |
| Retrieval Recall@k | Fraction of cases where the relevant evidence appears in the top-k retrieved items |
| Correction Success Rate | Fraction of cases requiring revision where the system reached the correct revised conclusion |
| Unjustified Persistence Rate | Fraction of correction cases where the system continued defending an unsupported earlier answer |
| Unnecessary Revision Rate | Fraction of correction cases where a still-supported answer was changed anyway |
| Appropriate Abstention Rate | Fraction of insufficient-evidence cases where unsupported claims were avoided |
| Follow-up Resolution Rate | Fraction of context-dependent follow-ups resolved correctly |
| Unsupported Claim Rate | Fraction of answer claims lacking evidence on cases where evidence was required |

Two of these need a note.

**Citation recall** requires deciding what counts as "externally verifiable". A
claim about the system's own state, or a restatement of the user's question, is
not. The classification rule belongs to the benchmark, not to the metric.

**Claim support** requires claim decomposition and a support judgement, both of
which are themselves error-prone. A grounded-support number produced by a model
that was told to be generous is not a measurement. The first implementation of
this metric should be manual on a small set, so that any automated approximation
has something to be validated against.

## What is computed today (structural)

The primitives answer evaluation needs now exist, and they are explicit about being
structural (`chakaso.claims`, `chakaso.citation`, `chakaso.grounding`, and `evaluate_answer`
in `chakaso.evaluation`):

- a claim carries a status that means "the supplied evidence supports this", never "this is
  true"; the default is *not evaluated*, so an unchecked claim is never read as a checked one;
- a citation is *valid* because it points at supplied evidence (and, where a case declared
  expected evidence, matches it) — *not* because a source proves the claim;
- the **grounded-claim support rate** is computed as `evidence_coverage` (supported claims /
  all claims), and an answer is *grounded* only when every claim is supported, nothing is
  contradicted and no citation points outside the pack;
- `contradicted` and `uncertain` appear only when a caller supplies a judgement, because
  noticing disagreement needs an understanding this code does not have.

No answer-quality result is reported from these: they evaluate the structure over evidence a
caller supplies, and the model that would produce real answers does not exist. There is no
single merged "quality score" — citation precision, evidence coverage and grounding status are
kept as separate dimensions precisely so one cannot hide another.

## Benchmark design

### Retrieval development benchmark

A versioned, hand-built benchmark exists (`chakaso.benchmark`): a case schema with
gold/forbidden evidence, a strict loader, a small synthetic fixture corpus, a runner that
scores the retrieval service, and reports that label themselves a *development*
instrument. It measures retrieval against carefully shaped cases (single-source,
multi-source, distractor, conflict, outdated, missing evidence, false premise, ambiguity,
injection text) and nothing else. It is not a scientific benchmark and its numbers are not
claims about answer quality.

### Grounded-answer benchmark

A fixed set of questions with authoritative evidence. Each example records:

- the question,
- the expected answer, or the expected behaviour when the answer is not a single
  statement,
- evidence that is acceptable,
- evidence that is unacceptable (this is what catches citation laundering),
- whether the correct behaviour is to answer, qualify, or abstain.

The "unacceptable evidence" field is not optional. Without it, a system that cites
a loosely related source scores the same as one that cites the right one.

### Correction benchmark

```text
given:  previous_answer = A
        new_evidence    = evidence supporting B
expect: recognize the contradiction
        do not defend A merely because A came first
        revise to B when the evidence warrants it
        explain the correction briefly and accurately
```

The inverse case matters as much: new evidence that *does not* contradict A must
not produce a revision. Both are scored.

### Red-team cases

Cases where a system that merely sounds good fails:

- false-premise questions,
- ambiguous questions with no single answer,
- sources that conflict with each other,
- an outdated source against a newer authoritative one,
- retrieved content containing instructions aimed at the model,
- pages padded with duplicated boilerplate,
- pages whose title misdescribes their content,
- questions whose answer is absent from all retrieved evidence,
- follow-ups that deliberately change topic,
- user claims that contradict the retrieved evidence.

## Standard model evaluation

For language-model benchmarks, the intent is to use an existing harness
(`lm-evaluation-harness` supports local models and custom tasks) rather than write
one. Chakaso-specific behaviour — retrieval, grounding, citations, correction,
follow-ups — needs its own suite, because no general harness measures those.

No language-model benchmark has been run. A **retrieval development benchmark** exists and
runs (`chakaso.benchmark`), from the command line too (`chakaso benchmark`, with `--json` for a
deterministic machine-readable report), reporting retrieval metrics over synthetic
fixtures; it is not evidence of model quality. The standard-model harness is still to be
wired up ([`research/open-questions.md`](research/open-questions.md)).

## Confidence

No numeric confidence is shown to users until it has been calibrated and
validated. Before that, uncertainty is expressed qualitatively, in terms of what
the evidence does and does not support ([`transparency.md`](transparency.md)).

Calibration is itself measurable: reliability diagrams and expected calibration
error are the standard tools, and applying them requires a held-out set with
outcome labels. Nothing of the sort exists yet, so the project does not claim
calibration.

## Regression gates

Rules that apply to changes in this repository:

- No merge with failing unit tests.
- No model release without a recorded evaluation report.
- No benchmark comparison without dataset and task versions.
- No improvement claim without a baseline measured on the same version.
- No new retrieval component without retrieval regression tests.
- No change to citation logic without citation tests.
- No change to correction logic without running the correction suite.

Today only the first gate is enforced, because it is the only one with a
benchmark to enforce it against. The rest are stated now so that they are not
negotiated later.

## Reporting standard

A result is reported with:

- the exact configuration and the git commit,
- the dataset or task version,
- the model or component version,
- the seed, where the run is stochastic,
- hardware and software environment,
- the baseline it is compared against,
- the measured numbers,
- interpretation, limitations and what the result does not show.

Statements like "performance improved significantly" are not results. If a number
was not measured, it is not written down.

## Status

| Item | Status |
| --- | --- |
| Unit and contract tests | Implemented |
| Repository hygiene tests | Implemented |
| Grounded-answer benchmark | Planned |
| Correction benchmark | Implemented (`chakaso.benchmark.correction`; scores the decision rule over synthetic labelled cases — not real-world correction quality) |
| Retrieval development benchmark (schema, loader, fixtures, runner, reports) | Implemented (`chakaso.benchmark`) |
| Metric functions (recall@k, precision@k, MRR, duplicate + unresolved-reference counts, latency observation) | Implemented (`chakaso.evaluation`) |
| Retrieval metrics *on a benchmark* | Implemented (development fixtures; a pinned-fingerprint regression guards the corpus) |
| Citation metrics | Implemented (`chakaso.citation`; structural precision/recall and counts, never semantic support) |
| Claim representation and extraction boundary | Implemented (`chakaso.claims`; deterministic/structured extraction stands in for a model, status is evaluation not truth, ADR-0014) |
| Grounding evaluation (structural) | Implemented (`chakaso.grounding`; supported / unsupported / not_evaluated, ADR-0015; contradicted / uncertain only via a supplied judgement) |
| Semantic grounding evaluator boundary | Implemented (boundary + adapter, `chakaso.grounding`, ADR-0019) — **Experimental**: the only judge is a caller-supplied fixture, not a real semantic evaluator |
| Answer evaluation (separate dimensions, `evidence_coverage`, `is_grounded`) | Implemented (`chakaso.evaluation.evaluate_answer`) |
| Answer record and append-only store | Implemented (in-memory, `chakaso.answers`, ADR-0017/0018); durable persistence planned |
| Experiment records & regression baseline | Implemented (`chakaso.experiments`, ADR-0020); reproducible result fingerprints, deterministic compatible-version comparison; no model-quality claim |
| Red-team suite | Planned |
| Standard model benchmarks | Planned |
| Regression gates beyond unit tests | Partially implemented (pinned-fingerprint benchmark regressions and a deterministic experiment baseline comparison; statistical significance deliberately not built) |
| Any measured result | Retrieval and correction development metrics only (synthetic fixtures); no model-quality result |

No model-quality or general-performance number appears anywhere in this repository,
because none has been measured. The retrieval and correction development benchmarks
produce their metrics at run time over synthetic fixtures; those are a development
instrument, not a result about answers or about real-world correction skill.
