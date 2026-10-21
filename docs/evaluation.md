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

**The turn's structural outcome is now a durable fact, not just a live view.** Alongside the
recomputable `AnswerEvaluation`, every recorded answer persists an immutable
`evaluation_provenance` snapshot (ADR-0028): the citation-structure counts, the per-claim
structural-grounding statuses, the evaluator name and version that decided them, and the evidence
references in the answer text that did not resolve. This is history, not a re-judgement: a later
evaluator recomputes the live view without rewriting what the original turn recorded, and a
reassessment is a separate event. Because the unresolved/malformed references are now persisted on
each answer, the **rate at which a model invents an evidence reference** — the measurement K-006
said nothing kept — becomes computable from the durable history once a durable store is selected
(ADR-0009: the reference is recorded, not fatal).

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

### Semantic-evaluation development instrument

A versioned, hand-labelled synthetic set of claim/evidence cases exists (`chakaso.evaluation.semantic_benchmark`,
`chakaso benchmark semantic`). Each case names a linguistic situation a judge must distinguish — direct
support, contradiction, partial support, irrelevance, negation, numbers/units, entity mismatch,
conflicting evidence, ambiguity, multi-sentence evidence, citation/evidence mismatch — and carries a
human-authored expected relation. The deterministic scorer reports accuracy, per-class
precision/recall, a confusion matrix, and how unresolved and evaluator-error cases are handled.

It measures the **scoring harness**, never semantic understanding. The only judge available is a
caller-supplied fixture, which is handed the expected relations and therefore scores perfectly — a
fact about the fixture, not a capability. Two named probes (support-all, always-uncertain) exist only
to show the scorer tells an over-claiming judge from an abstaining one. No real semantic evaluator
ships (ADR-0019/ADR-0027), so no number here is presented as general language-understanding
performance, and the dataset is a fixed seam for measuring one when it exists.

### Grounded-answer benchmark

A versioned, self-authored workload now implements the shape described below
(`chakaso.benchmark.grounded_*`; `chakaso benchmark grounded`; ADR-0031). Each example records:

- the question,
- the expected disposition (answer / qualify / abstain), the required facts an acceptable answer
  must carry, and the prohibited claims it must never make,
- evidence that is acceptable (gold passages, named by label and resolved to the content-derived
  `ChunkId`s retrieval actually produces),
- evidence that is unacceptable (this is what catches citation laundering),
- a provenance and licence record per source, and an evaluation rationale.

The "unacceptable evidence" and "provenance" fields are mandatory in the schema: a case with no
documented source licence or an evidence label that names no passage is refused at load, and a
passage that grounds a development case may not also ground a held-out one (split leakage). Sources
are **self-authored** (Apache-2.0, the project owns the text) rather than scraped, because publicly
accessible content is not automatically licensed for redistribution and the runtime is offline.

It is a **development instrument**, not a scientific benchmark and not a claim about answer quality:
the answer side is scored through an injected provider — a reference *oracle* that validates the
harness, and a non-answering provider that reflects the truth that no model answers today. The
measured baseline (E2603) reports real retrieval and structure metrics over the held-out and full
splits and keeps its negative findings visible (forbidden hits, false retrievals on abstention cases)
rather than merging them into one score.

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

## Model evaluation

A language model is measured against held-out text, and the metric that makes two
tokenizations comparable is **bits per character**: the total description length of a
scored stream in bits, divided by the characters those tokens decode to. Per-token
cross-entropy and perplexity are reported beside it and labelled as tokenizer-bound — a
1,024-entry subword vocabulary and a 90-character one do not mean the same thing by
`ln(V)`, so neither number may be used to compare the two. The denominator is the decoded
whole stream rather than a per-token character count, because a multi-byte character split
across byte tokens contributes its characters at the completion, not at each byte.

| Quantity | Definition | Where it is computed |
| --- | --- | --- |
| `loss_nats` | mean next-token cross-entropy over every scored position | `chakaso.training.model_metrics.metrics_from_scores` |
| `perplexity` | `exp(loss_nats)`, one identifier space only | same |
| `bits_per_character` | total bits / characters the scored tokens decode to | same |
| `token_accuracy` | argmax match rate over scored positions | same |
| `positions_scored`, `characters_scored`, `separator_targets` | the denominators, stated so a number can be re-derived | same |
| uniform baseline | `ln(V)` over the model's own identifier space | `chakaso.training.baselines` |
| unigram baseline | Laplace-smoothed training-token frequencies scored on held-out targets | same |
| held-out evaluation of the character model | `chakaso.training.evaluation` (aggregate) and `per_position_cross_entropies` (the series the shared arithmetic consumes) |
| held-out evaluation of the Transformer | `chakaso.training.nn_evaluation`, which scores an artifact with the tokenizer stored inside it |

Two properties are deliberate. Training loss and held-out loss are separate recorded
numbers — a step's training figure is the batch it updated, and a held-out figure is a
full pass over the evaluation side, recombined by position count so a short final batch
cannot tilt it. And the same metric function scores both architectures and both baselines,
so a comparison in an experiment record is one measurement applied four times rather than
four measurements that happen to be printed together.

**What model evaluation is not.** The retrieval and correction benchmarks, the semantic
scorer, the grounded-answer evaluation workload and the evidence-sufficiency policy measure
the harness and the retriever. A reference oracle in the grounded evaluation scores the
plumbing; it is not a model, and no number from those instruments is evidence of language
capability. That distinction is stated in the reports as well as here.

No external language-model benchmark has been run. The intent to use an existing harness
(`lm-evaluation-harness` supports local models and custom tasks) for general capability
stands, and Chakaso-specific behaviour — retrieval, grounding, citations, correction,
follow-ups — will still need its own suite because no general harness measures those.
[ADR-0033](decisions/ADR-0033-model-centred-product-contract.md) rule 7 keeps a borrowed
benchmark from becoming a substitute for a corpus and a task the project can defend.

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
| Grounded-answer benchmark | Implemented (`chakaso.benchmark.grounded_*`; `chakaso benchmark grounded`) — a versioned, **self-authored, Apache-2.0, held-out** workload with provenance-enforcing schema, a runner over the real retrieval + `evaluate_answer` boundaries, separate retrieval/answer metrics and deterministic reports. A development instrument: it evaluates the pipeline and retrieval, **not** answer capability (no model answers — K-001). Measured baseline: E2603 |
| Correction benchmark | Implemented (`chakaso.benchmark.correction`; scores the decision rule over synthetic labelled cases — not real-world correction quality) |
| Retrieval development benchmark (schema, loader, fixtures, runner, reports) | Implemented (`chakaso.benchmark`) |
| Metric functions (recall@k, precision@k, MRR, duplicate + unresolved-reference counts, latency observation) | Implemented (`chakaso.evaluation`) |
| Retrieval metrics *on a benchmark* | Implemented (development fixtures; a pinned-fingerprint regression guards the corpus) |
| Citation metrics | Implemented (`chakaso.citation`; structural precision/recall and counts, never semantic support) |
| Claim representation and extraction boundary | Implemented (`chakaso.claims`; deterministic/structured extraction stands in for a model, status is evaluation not truth, ADR-0014) |
| Grounding evaluation (structural) | Implemented (`chakaso.grounding`; supported / unsupported / not_evaluated, ADR-0015; contradicted / uncertain only via a supplied judgement) |
| Semantic grounding evaluator boundary | Implemented (boundary + adapter, `chakaso.grounding`, ADR-0019/ADR-0027) — **Experimental**: the only judge is a caller-supplied fixture, not a real semantic evaluator; the contract carries evaluator name+version and an explicit error outcome, and `describe_semantic_capability()` reports no real evaluator |
| Semantic-evaluation dataset & scorer | Implemented (`chakaso.evaluation.semantic_benchmark`; `chakaso benchmark semantic`) — a development instrument measuring the harness over synthetic labels; the fixture's perfect score is structure-replay, not semantic capability |
| Answer evaluation (separate dimensions, `evidence_coverage`, `is_grounded`) | Implemented (`chakaso.evaluation.evaluate_answer`) |
| Answer record and append-only store | Implemented (in-memory **and** a durable SQLite backend, `chakaso.answers`, ADR-0017/0018/0021); the durable store is proven to survive a process restart |
| Persisted historical evaluation provenance (citation/grounding/reference outcomes + evaluator identity the turn reached; live view stays recomputed) | Implemented (`chakaso.answers.provenance`, ADR-0028); persisted inside the answer document, backward-compatible decode, corrupt provenance rejected, dimensions kept separate, structural `is_grounded` never a truth verdict |
| Experiment records & regression baseline | Implemented (`chakaso.experiments`, ADR-0020); reproducible result fingerprints, deterministic compatible-version comparison; no model-quality claim |
| Retrieval-strategy comparison (lexical vs dense vs hybrid over the fixtures) | Implemented (`chakaso.experiments.compare_retrieval_strategies`); measured deltas only, labelled development — dense/hybrid run over the non-semantic fixture embedding (E-001 showed no gain, a small MRR regression) |
| Red-team suite | Planned |
| Model metrics: held-out loss, perplexity, bits per character, token accuracy, uniform + unigram baselines | Implemented (`chakaso.training.model_metrics`, `chakaso.training.nn_evaluation`, `chakaso.training.evaluation`); dependency-free arithmetic, tested against hand-computable cases and against the training loop's own position-weighted loss |
| Measured model result on a real text corpus | E2605: a subword Transformer versus the character model versus both baselines, in bits per character, on the project's own pinned documentation corpus. A single author, one domain, CPU scale — not general capability, and not an external benchmark |
| Standard external model benchmarks | Planned |
| Regression gates beyond unit tests | Partially implemented (pinned-fingerprint benchmark regressions and a deterministic experiment baseline comparison; statistical significance deliberately not built) |
| Any measured result | Retrieval, correction and retrieval-strategy development metrics over synthetic fixtures (E-001); the first trained-model results (E2601, E2602) and the first subword Transformer result (E2605), all on small self-authored or synthetic text; the grounded-answer evaluation and evidence-sufficiency baselines over the self-authored workload (E2603, E2604). None of these is a real-workload result, and none tests a semantic-retrieval or answer-capability hypothesis |

No general-performance number appears anywhere in this repository. Model-quality numbers do
exist — held-out loss, perplexity and bits per character for a character MLP and for a small
subword Transformer, measured on the project's own text — and each is stated with the corpus,
split, seed, parameter count and baselines that produced it, because a model number without its
data and protocol is not a result. The retrieval and correction development benchmarks
produce their metrics at run time over synthetic fixtures; those are a development
instrument, not a result about answers or about real-world correction skill.
