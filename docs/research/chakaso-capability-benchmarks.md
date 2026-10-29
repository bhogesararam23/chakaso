# Chakaso capability benchmarks: what each claim would require, and what exists now

**Status:** Research and specification. Most instruments below are **Planned**. The only measured model
instruments are held-out language modelling (E2605–E2608) and the calibrated generation suite (E2610); the
grounded-answer workload (E2603/E2604) measures the harness with an oracle, not a model. Nothing here is a
capability result.
**Date:** 2026-10-29.
**Rule:** a capability claim needs a workload, a model that produces the output, a scorer independent of the
model, held-out data, a baseline, and reported failures. A file, an oracle score or a repetition statistic is
not a capability.

## The claim ladder

Every capability below is assessed on the same ladder. A rung is not skipped because the one below it looks
good.

| Rung | Question | Evidence that closes it |
| --- | --- | --- |
| 0. Mechanism | Does the code path run end to end? | A test or an experiment that executes it |
| 1. Learning | Does the model beat a trivial predictor on held-out data? | Loss or task metric below uniform/unigram or a trivial-answer baseline, on whole held-out works |
| 2. Task | Does the model produce the required output on a defined workload? | A measured run where the model, not an oracle, emits the answer |
| 3. Quality | Is the output good enough for the claimed use? | A scorer independent of the model, a declared threshold, and a failure analysis |
| 4. Robustness | Does it hold across data, seeds and paraphrases? | Replication across seeds and at least one held-out distribution shift |
| 5. Capability claim | May the project say it can do this? | Rungs 1–4, with the limitations published beside the claim |

At the current scale (3.4M parameters, 19.8M training positions, a 6.72M-token prose corpus), the project is
at rung 1 for language modelling and rung 0 for everything else. That is the honest baseline this document
starts from.

## Cross-cutting rules

1. **Whole-work splits.** A held-out document is not a held-out work. E2610's deviation 3 records that the
   current split can put another document of the same book on the training side. Every benchmark below
   splits by work and author, and the split is recorded in the report.
2. **Contamination checks.** Exact and n-gram overlap between training and evaluation text; canary strings
   for synthetic sets; a declared result. A benchmark item that appears in training is not an evaluation
   item.
3. **No merged quality score.** Dimensions stay separate (as `evaluate_answer` already does); a single
   "score" hides which failure produced it.
4. **Raw samples retained.** Every scored run keeps the generations, prompts, seeds, configuration hash and
   artifact hash, so a reader can inspect what the metric scored.
5. **No gate changes after results.** A gate that proves inadequate is recorded as a limitation and a new
   experiment declares a new gate; E2610's deviation 1 is the precedent.
6. **Baselines are named.** A benchmark without a baseline is a demo. Uniform/unigram for language modelling;
   trivial-answer, retrieval-only and oracle baselines for the task benchmarks; an empty-function baseline
   for code.
7. **Negative cases and abstention.** A benchmark must contain cases whose correct answer is "not supported"
   or "insufficient evidence", and must score abstention separately from answering.

## 1. Language modelling

**What exists:** E2605–E2608, `chakaso.training.model_metrics`, `nn_evaluation`.

**Workload.** Held-out documents of a pinned corpus, scored once after validation selection; bits per
character as the tokenizer-independent unit, with nats/token, perplexity and token accuracy reported beside
it and labelled tokenizer-bound.

**Baselines.** Uniform over the identifier space; unigram fitted on the training side; the character
architecture where applicable.

**Splits.** Document-level today; the next version must be whole-work, with the per-side identifiers in the
report.

**Necessary:** lower held-out bits per character than uniform and unigram, on every seed, with the curve and
selected step reported. **Sufficient for a capability claim:** nothing. A lower loss is evidence of learning
statistical structure, not of answering, following instructions or being correct. E2610 is the proof: the
released model's loss is the best measured here and its samples are not text a reader would keep.

## 2. Coherent continuation

**What exists:** `chakaso.training.quality_eval` (E2610) measures word distinct-n, loops, periodic runs and
replacement characters against a human reference on the same prompts. Its own `interpretation_limit` says
these are surface statistics and a fluent-nonsense generator can pass. K-020 records the gap.

**Workload.** 100 held-out prompts, each with a human continuation; 64–128 generated tokens per prompt;
greedy and the declared sampled settings, multiple seeds.

**Scoring, two independent routes:**

- **Human rubric.** 0–4 on grammar, local coherence, global coherence, and factual consistency; three
  raters, inter-rater agreement reported; a sample of at least 30 prompts; the rubric and rater instructions
  fixed before the run. This is expensive and is the honest route to a coherence claim.
- **Cross-model conditional likelihood (proposed E2612).** Score the human continuation and the model's
  continuation under an independent judge model, on the same prompt; report per-token NLL, the paired gap and
  the human-win rate. This is a proxy, not coherence: a judge trained on similar data can prefer its own
  distribution, and a low NLL does not mean a claim is true. The judge must be identified, and a self-judge
  control (the judge's own continuation scored under itself) must be reported.

**Necessary:** passes the repetition gates on the declared prompts and seeds; not bound-truncated on every
sample. **Sufficient:** a human rubric above a declared threshold with reported agreement, or a cross-model
gap in the declared direction together with a human spot-check. A repetition gate alone is never sufficient
(K-020).

## 3. Instruction following

**What exists:** a versioned role-labelled prompt template and the model boundary's message shape. No
instruction-tuned checkpoint, no instruction benchmark, no SFT stage.

**Workload.** 100 self-authored, verifiable instructions with programmatic checks: exact format (JSON, a
specific heading), length bounds, required and forbidden phrases, ordering, a simple transformation, a
refusal where the request is unsupported. Modelled on IFEval's verifiable-constraint idea, but self-authored
so it can be released under the project's licence.

**Baselines.** The base model without instruction tuning; the deterministic development double (which cannot
follow instructions) is not a baseline for quality, but it proves the harness rejects non-answers.

**Necessary:** the base model's instruction accuracy is measured before SFT; an SFT checkpoint improves it
without a measured regression on held-out language modelling. **Sufficient:** accuracy above a declared
threshold on held-out instructions, with raw samples and a failure taxonomy (format, content, refusal,
length). Instruction following is separate from next-token prediction and must never be inferred from loss.

## 4. Factuality and grounded answering

**What exists:** the E2603/E2604 grounded-answer workload and runner, with an oracle and a non-answering
provider. The oracle measures retrieval and evaluation; no model answers. K-001.

**Workload.** The existing 13-passage/14-case workload plus a larger self-authored held-out set. Each case:
a question, acceptable and unacceptable evidence, required facts, prohibited claims, an expected disposition
(including abstain), and a rationale. The model receives retrieved evidence with identifiers and must cite
them.

**Scoring.** Retrieval recall@k, MRR, forbidden hits, false retrievals; citation precision/recall, evidence
coverage, unsupported-claim rate, appropriate-abstention rate, grounded-answer rate. Dimensions separate; no
merged score. An answer is not "true" because a source was cited; grounding is structural.

**Necessary:** retrieval finds the gold evidence on the held-out split; citations resolve only to supplied
identifiers; abstention is possible. **Sufficient:** a model, not an oracle, produces answers that meet a
declared grounded-answer rate on held-out cases with a reported failure taxonomy. Until then the measurement
is of the harness, and is labelled so.

## 5. Mathematics

**What exists:** nothing. No math corpus, tokenizer decision, benchmark or model.

**Workload.** 100 self-authored problems across arithmetic, algebra, geometry and word problems, each with
an exact answer and a programmatic checker; a held-out set generated from different templates and wordings.
Optionally a small set of derivation problems with a human rubric for the steps, because a final-answer check
does not measure reasoning.

**Baselines.** Always-answer-zero; a retrieval-only lookup; a calculator is not a language-model baseline and
must not be compared as one. For a model, report exact-match accuracy and, separately, valid-format rate.

**Necessary:** the tokenizer can represent digits and operators without pathological fragmentation; the
model beats the trivial baseline. **Sufficient:** held-out exact-match above a declared threshold with
per-category failures and a contamination check. GSM8K/MATH scores alone are not sufficient: they are
contaminated in most web-trained models and are not the project's own data.

## 6. Code

**What exists:** nothing. No code corpus, execution sandbox or code benchmark.

**Workload.** 50 function-synthesis problems with unit tests, self-authored or permissively licensed with
provenance, executed in a sandbox with a timeout, no network, no filesystem beyond a temp directory. Report
pass@1, syntax-valid rate, and the failure category (parse, runtime, wrong output, timeout).

**Baselines.** An empty function (fails all), a memorised-solution check (exact and near-duplicate match
against training code), and a trivial return of the first test's expected value where applicable.

**Necessary:** generated code parses and runs in the sandbox at a non-zero rate; the benchmark's problems are
not in the training corpus. **Sufficient:** pass@1 above a declared threshold on held-out problems, with
failures reported and no claim of software engineering capability from a 50-problem set.

## 7. Physics

**What exists:** nothing.

**Workload.** 50 text-based physics problems (mechanics, thermodynamics, electromagnetism, optics) with
numeric answers, units and tolerances; a subset with symbolic answers. A human rubric for the explanation
when the answer is numeric, because a correct number can come from a wrong method.

**Scoring.** Exact numeric match within tolerance; unit correctness; explanation rubric; refusal when the
problem is under-specified. Baselines: always-zero, dimensional-analysis-only, retrieval-only.

**Necessary:** the model produces a parseable final answer; **sufficient:** held-out accuracy above the
declared threshold with a failure taxonomy and a contamination check. Do not infer physics reasoning from
mathematics scores or from prose about physics.

## 8. Research synthesis

**What exists:** nothing. The retrieval and citation primitives exist; the task does not.

**Workload.** 20 tasks, each with 5–10 supplied documents (public domain or self-authored) and a question
that requires combining them. Required facts, prohibited claims, and a citation for every required fact; a
held-out set with different documents.

**Scoring.** Retrieval recall; claim coverage; citation correctness; unsupported-claim rate; a human rubric
for organization and whether the synthesis adds a correct comparison rather than concatenating passages. No
semantic judge exists, so a human (or a declared, validated proxy) is required.

**Necessary:** retrieval finds the required passages; citations resolve. **Sufficient:** a human-rated
synthesis above threshold with every required fact supported by a cited passage and no prohibited claim.
A summary that is fluent but uncited fails.

## 9. Text-based 3D and geometry

**What exists:** nothing. A text-only model does not see images, meshes or scenes; this benchmark is
explicitly **text-based reasoning**, not multimodal understanding.

**Workload.** 50 text-described spatial tasks: coordinate geometry, rotations and reflections, projections
described in words, mesh topology questions (faces/edges/vertices), and simple CAD-like constraints, each with
an exact answer. A held-out set with different objects and wordings.

**Baselines.** Random choice among valid answers; a symbolic solver where the problem is formalisable (to show
what a non-language method achieves); a retrieval-only lookup.

**Necessary:** the model produces a parseable answer and beats random. **Sufficient:** held-out accuracy above
threshold with a failure taxonomy; separately, a human-rated explanation. No claim about images, meshes or 3D
scenes is permitted from a text-only result.

## What would be necessary and sufficient for the project's headline claim

The long-term claim is "a model that learns, explains and assists with language, mathematics, science and
engineering". It is not one benchmark. A defensible version requires, at minimum:

1. held-out language modelling better than baselines on more than one distribution;
2. coherent continuation judged by humans or a validated proxy, not by repetition ratios;
3. instruction following measured separately from pretraining loss;
4. grounded answering with citations and abstention, produced by the model, not an oracle;
5. at least one domain task (mathematics, code, physics or synthesis) with a checkable scorer and a held-out
   set;
6. replication across seeds and a distribution shift;
7. published failures and a statement of what remains unknown.

At 3.4M parameters and 6.72M tokens, none of 1–7 is expected to pass. The purpose of defining them now is to
make the next model's claims falsifiable before it is trained, not to decorate a release.

## Contamination and benchmark gaming controls

- Keep whole works, benchmark items and their paraphrases out of training; check exact and n-gram overlap.
- Record the check and its result in every report; a missing check is a limitation, not a pass.
- Do not select a checkpoint on the test split; validation selects, test reports once.
- Do not tune a gate after seeing the result; a changed gate is a new record.
- Report raw samples and the worst cases, not only the aggregate.
- Do not compare to a frontier model's published score on a different harness; a number is only comparable
  when the workload, split and scorer are the same.
- Treat a benchmark's own contamination as a known risk; the Data Provenance Initiative and the GSM1k-style
  work exist because public benchmarks leak into web data.

## Status table

| Capability | Workload | Model produces output? | Scorer | Measured? | Status |
| --- | --- | --- | --- | --- | --- |
| Language modelling | held-out documents, whole-work next | yes | bits/character vs baselines | E2605–E2608 | **Measured**, one corpus/register |
| Coherent continuation | 100 prompts + human references | yes | repetition suite; human rubric planned | E2610 (surface only) | **Partial**; K-020 |
| Instruction following | 100 verifiable instructions | no SFT checkpoint | programmatic checks | no | **Planned** |
| Factuality / grounded QA | E2603 workload + held-out | no model answers | structural citation/grounding | E2603/E2604 oracle only | **Harness measured**; K-001 |
| Mathematics | 100 exact-answer problems | no | exact match + rubric | no | **Planned** |
| Code | 50 unit-tested functions | no | sandbox execution | no | **Planned** |
| Physics | 50 numeric/symbolic problems | no | tolerance + rubric | no | **Planned** |
| Research synthesis | 20 multi-document tasks | no | claim/citation + human rubric | no | **Planned** |
| Text-based 3D/geometry | 50 spatial tasks | no | exact answer + rubric | no | **Planned** |

## References

1. E2603/E2604. [`../../experiments/E2603/record.md`](../../experiments/E2603/record.md),
   [`../../experiments/E2604/record.md`](../../experiments/E2604/record.md).
2. E2605–E2608. [`../../experiments/`](../../experiments/).
3. E2610. [`../../experiments/E2610/record.md`](../../experiments/E2610/record.md) and K-020.
4. Zhou, J. et al. *Instruction-Following Evaluation for Large Language Models.* arXiv:2311.07911, 2023.
   <https://arxiv.org/abs/2311.07911>
5. Cobbe, K. et al. *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168, 2021.
   <https://arxiv.org/abs/2110.14168>
6. Hendrycks, D. et al. *Measuring Mathematical Problem Solving With the MATH Dataset.* arXiv:2103.03874,
   2021. <https://arxiv.org/abs/2103.03874>
7. Chen, M. et al. *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374, 2021.
   <https://arxiv.org/abs/2107.03374>
8. Rein, D. et al. *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* arXiv:2311.12022, 2023.
   <https://arxiv.org/abs/2311.12022>
9. Lin, S. et al. *TruthfulQA.* arXiv:2109.07958, 2021. <https://arxiv.org/abs/2109.07958>
10. Wei, J. et al. *Emergent Abilities of Large Language Models.* arXiv:2206.07682, 2022.
    <https://arxiv.org/abs/2206.07682>
11. Longpre, S. et al. *The Data Provenance Initiative.* arXiv:2310.16787, 2023.
    <https://arxiv.org/abs/2310.16787>
12. *GSM1k* — a contamination-focused re-evaluation of GSM8K-style math data (arXiv:2410.05229, 2024),
    cited as the kind of check this project must run before trusting a public math score.
    <https://arxiv.org/abs/2410.05229>
