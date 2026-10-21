# Research log

Dated entries in the order they happened. Entry types follow
[`README.md`](README.md): observation, hypothesis, experiment, result,
interpretation, limitation, conclusion.

---

## 2026-10-07 — Log opened

**Observation.** The repository began from an empty tree. There is no model, no
tokenizer, no dataset, no retrieval implementation and no benchmark. Nothing in
this log can be a result yet.

**Observation.** The founding specification fixes three constraints that determine
what can be measured: no commercial inference or search dependency at runtime, a
model boundary that application code depends on rather than a model, and evidence
identifiers owned by retrieval rather than generated
([ADR-0001](../decisions/ADR-0001-local-first-runtime.md),
[ADR-0002](../decisions/ADR-0002-language-model-boundary.md),
[ADR-0003](../decisions/ADR-0003-evidence-identifier-ownership.md)).

**Interpretation.** The third constraint is what makes the project's headline
metrics computable at all. Citation precision and unsupported claim rate are
fractions over a set of claims and a set of evidence. If the generator produces
source identity, neither set is well defined, and the metrics degrade into judging
whether an answer sounds sourced.

**Limitation.** This is a design argument. It predicts that a system with
retrieval-owned identifiers will be more measurable than one without, and that
prediction has not been tested. It is also not the same claim as "such a system
produces better answers", which is a separate and unproven proposition.

**Decision recorded.** Build the system before scaling the model
([roadmap](../roadmap.md)). The reasoning: evaluation infrastructure, evidence
handling and correction behaviour must exist before a trained model can be judged,
and a model judged by nothing cannot be improved deliberately.

**Open at this point.** Whether a small local model with strong retrieval produces
useful answers ([H1](hypotheses.md), [H3](hypotheses.md)); whether a dedicated
reassessment path beats single-pass reconsideration ([H2](hypotheses.md)); how the
first web fetch mechanism should work
([open questions](open-questions.md#how-should-the-first-web-fetch-mechanism-work)).

---

## 2026-10-07 — Foundation scope fixed

**Observation.** The repository has to distinguish carefully between two things
that are easy to conflate: *the model layer is replaceable* and *the model exists*.
The first is now true by construction. The second is false, and nothing in the
documentation should read as though it were true.

**Observation.** The first foundation unit that is genuinely useful rather than
decorative is the set of types that everything else exchanges: evidence
identifiers, source and chunk records, conversation state, and the model boundary.
These are the pieces whose shape is expensive to change later, because they are
referenced from everywhere.

**Interpretation.** Writing these types before retrieval exists is not speculative
scaffolding. The identifier contract in ADR-0003 is only enforceable if the types
that carry identifiers exist first; retrofitting immutability and provenance onto
types designed without them is the expensive version of the same work.

**Limitation.** The types may still be wrong. Their correctness will be tested by
whether retrieval and correction can be built on them without modification, which
has not happened.

---

## 2026-10-07 — No experiments run

**Conclusion (strong, and the only one available).** The project has made design
decisions and has not produced evidence about language models. Every quantitative
claim in this repository would be fabricated. There are none.

---

## 2026-10-08 — Retrieval substrate built (local)

**Observation.** The blocking question — what identifies a source with no URL — was
not a naming problem but a conflation: a canonical URL was serving as both identity and
fetch permission. Separating them (ADR-0011) is what let ingestion proceed, and it kept
every web identifier byte-identical because the web reference *is* the canonical URL. The
formula of ADR-0007 never moved; only its input widened.

**Observation.** Building the smallest retrieval step first — lexical BM25 over a local
corpus, no embeddings, no network — was a deliberate choice of sequence, not a limitation
of ambition. A lexical baseline is deterministic and cheap enough to evaluate, so it can
establish what retrieval actually needs from the decisions that would otherwise be guesses:
which index, whether to rerank, what a dense retriever must improve on.

**Interpretation.** Normalization sits before chunking on purpose, because normalization
feeds the content hash and the chunk identifiers. Two representations of one document
(CRLF and LF, composed and decomposed accents) must produce one set of identifiers, or an
evaluation would compare formatting rather than content. The consequence recorded as a
limitation: reproducibility now depends on pinning the interpreter, because NFC depends on
the Unicode tables of the running Python.

**Decision recorded.** The web fetcher is an optional, explicitly-bounded adapter
(ADR-0012), off every default path. Making the transport injectable is what keeps the
runtime offline (ADR-0001) while still letting the whole fetch policy — scheme, size,
redirects, timeout, destination — be tested deterministically.

**Limitation (stated, not hidden).** The destination screen blocks private *address
literals* but cannot stop a hostname that resolves to a private address; that needs
IP-pinned resolution in the transport, which is not implemented. It is recorded as K-008
rather than claimed as a solved SSRF defence.

**Open at this point.** Whether lexical retrieval is good enough to be worth evaluating
against a real corpus, and where it fails — which is the question the metric functions and
a first hand-built benchmark would answer. Neither the benchmark nor any measured result
exists yet, so nothing here is a claim about retrieval quality.

---

## 2026-10-08 — Evaluation and correction foundations built

**Observation.** The benchmark made the metric functions runnable against something defined:
cases with gold (acceptable) and forbidden (unacceptable) evidence, a strict loader, a synthetic
fixture corpus and a runner. Making *forbidden evidence* a first-class field is what lets the
instrument catch citation laundering — without it, a system that cites a loosely-related source
scores the same as one that cites the right one.

**Observation.** The load-bearing distinction throughout is structural versus semantic. Citation
validation and the default grounding evaluator decide only presence and set membership — valid /
unknown / irrelevant; supported / unsupported — and never that a source proves a claim.
`contradicted` and `uncertain` enter only through a judgement a caller supplies, and the engine
names no winner in a conflict and applies no precedence automatically (ADR-0015).

**Decision recorded.** The correction decision is a written rule, not a heuristic (ADR-0016): a
claim is corrected only when the supplied evidence contradicts it — never because it was produced
first or has aged — losing prior support *qualifies* rather than deletes, and an unresolvable
conflict or an unsettled required claim yields `needs_review` / `abstain` instead of a guess.
Correction here decides and records; it produces no revised prose, because there is no model to
write one.

**Interpretation.** Keeping an answer's quality dimensions separate — citation precision, evidence
coverage, grounding status, contradictions — rather than merging them into one score is what makes
a partially-supported answer legible as exactly that. A single "quality score" would hide an
uncited claim behind high precision over the claims that *were* cited.

**Limitation.** Every number here is over a small, hand-built, synthetic corpus and scores
*retrieval* only — no model answers, so nothing downstream of retrieval is measured. The
development benchmark's output is a development instrument, not evidence about general
performance, and no experiment testing a hypothesis has been run. K-001 (nothing measured against
a real workload) is narrowed for retrieval but not closed; for grounding, citation and correction
the machinery now exists while the semantic capability and any real measurement do not.

**Open at this point.** Whether a semantic evaluator — the named future implementation of the
`GroundingEvaluator` boundary — can set `contradicted`/`uncertain` reliably enough to replace a
supplied judgement, and whether the correction rule holds against real conflicts once a model
produces the answers being corrected.

---

## 2026-10-08 — Persistence, reassessment, a semantic boundary and reproducible experiments

**Observation.** Answer identity turned out to be a different kind of decision from evidence
identity. Everything else in the system derives identifiers from content (ADR-0007), but an answer
is a runtime event: a corrected answer that happens to repeat the original wording must stay
distinguishable from the one it supersedes, so a content hash is the wrong tool and a per-event
random identifier is the right one (ADR-0017). The reproducibility the content hash would have
given is instead carried by the transcript and the deterministic benchmark version, not the id.

**Observation.** Making persistence a boundary (`AnswerStore`) before building any backend let the
correction history, the conversation manager and the experiment layer all be written against the
append-only invariants — no overwrite, no dangling or cross-conversation correction — once, in the
store, rather than at every call site. The in-memory implementation is enough to test those
invariants now and leaves a durable one as a drop-in (ADR-0018), so no storage schema is frozen
before a real workload asks for one.

**Decision recorded.** The semantic grounding capability is specified as a *boundary* whose only
implementation is a caller-supplied fixture, deliberately not a keyword-overlap heuristic dressed
up as entailment (ADR-0019). The line ADR-0015 drew — structure is not understanding — is held by
naming the fixture in every result, so a report cannot mistake a replayed relation for judgement.

**Observation.** Separating an experiment's reproducibility fingerprint from its environment
metadata was the point of the experiment layer: the same run on the same code reproduces the same
`result_id` seconds later on a different host, while a change to configuration, benchmark version,
evaluator or the metrics moves it (ADR-0020). Folding the clock or host into the identity would
have made every run "different" and destroyed exactly the reproducibility being checked.

**Interpretation.** The correction development benchmark is the first place the decision rule is
exercised end to end against labelled expectations, and it can fail: a test that corrupts a case's
expectation flips the run red, which is the pre-condition for a green run meaning anything. This
narrowed K-001 for the correction path the way the retrieval benchmark did for retrieval — a
labelled synthetic set, not a real workload.

**Limitation.** No experiment testing a hypothesis about a model has been run. Every benchmark and
experiment number here is computed over small synthetic fixtures and scores a rule or a retriever,
not answer quality; there is still no language model, and the semantic boundary has no real judge.
K-001 stands.

---

## 2026-10-10 — Durable storage through the runtime, and the first strategy comparison

**Observation.** The layers built across this run — a durable SQLite answer store (ADR-0021), a
deterministic query planner (ADR-0022), a dense retrieval boundary over a non-semantic fixture
embedding (ADR-0023), reciprocal-rank-fusion hybrid retrieval (ADR-0024) and a retrieval-aware
conversation runtime (ADR-0025) — each entered as a boundary with one honest implementation, the
same move that kept the model, storage and semantic-judging seams open. The runtime composes them
without a monolith: a grounded turn runs plan → retrieve → generate → validate → record → persist,
while the deterministic double still cites nothing, so the loop executes without any answer being
real.

**Experiment E-001 — does dense or hybrid retrieval change measured retrieval, over the synthetic
development fixtures?** Configuration: dataset `dev-retrieval` v`1.0.0`, content fingerprint
`36e65ab20468d141`; `top_k=5`; baseline `lexical` (BM25); candidates `dense` and `hybrid`, both over
`FixtureEmbeddingModel` (`fixture-hash-embedding-64`, seed `chakaso-fixture-embedding`,
`is_semantic=false`); fusion `rrf_k=60`, weights 1.0/1.0. Offline and deterministic; reproduced by
`chakaso benchmark strategies`.

**Result (measured, not assumed).** recall@k: lexical 1.000, dense 1.000, hybrid 1.000. precision@k:
0.257 for all three. hit_rate: 1.000 for all three. forbidden_hits: 2 for all three; false_retrievals:
3 for all three; duplicates: 0 for all three. **MRR: lexical 1.000, dense 0.929, hybrid 0.929** — a
−0.071 *regression* against the baseline for both non-lexical strategies.

**Interpretation.** Over a non-semantic hashed bag-of-tokens embedding, dense and hybrid retrieval
produce no measurable gain on recall, precision or hit rate, and reorder the top of the ranking
slightly worse (lower MRR) than BM25. That is the expected signature of a fixture double: it
re-derives token overlap in vector form, so fusion mostly reshuffles near-ties rather than surfacing
semantically-relevant-but-lexantically-absent evidence.

**Limitation.** This is a neutral-to-negative result about the *mechanism over synthetic fixtures*,
not about dense or hybrid retrieval in general. The embedding is explicitly non-semantic, the corpus is
a hand-built toy, and no model answers. It neither supports nor refutes H7 (a *real* dense retriever
exceeding lexical on paraphrase), which needs a genuine embedding model and a paraphrase-rich gold
corpus — neither exists. It is recorded so a future semantic embedding has a concrete number to beat:
to count as progress on this fixture, it must exceed a −0.071 MRR delta, not merely differ.

**Conclusion.** The strategy-comparison instrument works and reports honestly, including a regression;
no strategy is claimed superior. K-001 stands.

---

## 2026-10-15 — Reassessment becomes reachable, and a model-feasibility check

**Observation.** P21 closed the last deterministic gap on the correction side without inventing an
autonomous one: a follow-up that *explicitly names* a stored answer (a valid `ans_…` reference) now
composes into a reassessment through the runtime, while the intent trigger stays separate from execution
and an ambiguous or unknown target is refused rather than guessed (ADR-0029). No revised prose is
produced — there is still no model — and the original answer and its provenance are never rewritten.

**Observation.** The machine available for model work was inspected directly, and its binding constraints
are environmental, not architectural: **no CUDA GPU (Intel Arc only), ~15.4 GB total RAM with ~1.4 GB
free at inspection, and no ML dependency installed in a deliberately dependency-free environment.** This
is an observation about one host at one moment, not a measurement of any model.

**Interpretation.** Both a runnable pretrained model and a from-scratch training experiment fit the
`LanguageModel` boundary cleanly; the gate on either is admitting a numerical dependency as an *optional,
non-default* extra (its own ADR) — not a change to any interface. Under the observed constraints a
1–3B pretrained run is memory-bound and its external weights strain reproducibility and licence,
whereas a tiny from-scratch run is self-owned, memory-comfortable, and would produce the project's first
*genuine measured number* (a held-out curve) — the thing K-001 says is still absent.

**Limitation.** No experiment was run and no model was trained, downloaded or installed for this note; it
contains no measured result. Every non-inspected figure there is labelled an estimate or an assumption.

**Conclusion (weak, pending the dependency ADR).** Defer the pretrained-model adapter until a real
grounded-answer evaluation exists to consume it; the smallest honest next step is a scoped, out-of-CI
from-scratch training experiment recording a held-out metric, gated behind a dependency-feasibility ADR.
The full analysis is in [`model-feasibility.md`](model-feasibility.md). K-001 stands.

---

## 2026-10-19 — The dependency decision is made (ADR-0030): numpy, optional, hand-verified gradients

**Observation.** Package metadata was read from PyPI's authoritative JSON API. The newest NumPy
(`2.5.4`) declares `Requires-Python >=3.12`, so it *drops* the project's 3.11 floor; the newest NumPy
release still shipping a `cp311` wheel is `2.4.6`. NumPy's licence resolves to BSD-3-Clause plus bundled
permissive components (0BSD/MIT/Zlib/CC0); PyTorch's is a much larger mixed set and its CPU wheel is
orders of magnitude heavier. This is a packaging-fact observation, **not a measured model result**.

**Decision recorded.** [ADR-0030](../decisions/ADR-0030-minimal-optional-training-dependency.md) admits
**NumPy as a single, optional `[train]` extra** (`numpy>=1.26,<2.5`, the cap chosen specifically to keep
Python 3.11 installable) with the model's gradients **derived by hand and verified by finite differences**
— not a framework's autograd. The base install stays dependency-free and the core CI matrix is unchanged;
training tests `importorskip("numpy")` and are exercised remotely only by a separate optional CI job.
Training code is placed in `chakaso.training`, the module [ADR-0005](../decisions/ADR-0005-src-layout.md)
reserved for it. No dependency was installed and no code was written for this decision; it is the gate the
roadmap requires before model code.

**Interpretation.** The choice trades a framework's convenience for *verifiability*: for a few-thousand-
parameter char model, a hand-derived gradient checked against finite differences makes "the loss actually
connects to the trainable parameters" a tested property of this project's own code rather than something
inherited on trust. That is exactly the instrument the first measured result needs to be believable. The
decision narrows the [Which local model should the shell use?](open-questions.md#which-local-model-should-the-shell-use)
open question further — it fixes the numerical substrate for the from-scratch path — without selecting a
model, and it keeps a heavyweight framework as a documented *revisit* option, not a dismissed one.

**Limitation.** This decided a dependency from published metadata and prior inspection; it ran nothing and
therefore measured nothing about training speed, memory or output quality. The bound and the wheel
availability are packaging facts verified on this date; performance and memory claims still belong only to
an experiment that actually runs. K-001 stands: the next unit is that experiment.

---

## 2026-10-19 — E2601: the first model actually trained and measured (a from-scratch character LM)

**Experiment E2601 — can a randomly initialised character model, trained by a real optimizer, lower its
held-out next-character cross-entropy below a trivial baseline?** Configuration (pinned in
`chakaso.training.experiment.DEFAULT_CONFIG`): context 4, embed 8, hidden 12, Adam, learning rate 0.02,
seed 7, 300 full-batch steps, evaluated every 25; corpus `chakaso-toy-char-v1` v1.0.0 (self-authored,
Apache-2.0), leak-free split with the vocabulary fit on training text only; 344 train / 78 held-out
positions over a 28-symbol vocabulary; 984 parameters. Environment: Windows 11, CPython 3.13.5, numpy
2.4.6, CPU-only (no CUDA). Reproducible from the seed and config. Full record:
[`../../experiments/E2601/record.md`](../../experiments/E2601/record.md).

**Result (measured, not assumed).** Baselines on the held-out tokens: uniform 3.3322 nats, unigram
3.0531 nats. Held-out cross-entropy: 3.3118 at random initialization (step 0), falling to a minimum of
**1.6650 nats at step 50**, then rising back to **2.4946 nats at step 300** (final bits/char 3.5989,
perplexity 12.12, next-character accuracy 0.6026). The trained model beats **both** baselines at its best
and at the final step. The full training-loss / held-out-loss curve is recorded in the experiment file.

**Interpretation.** This is the project's first genuinely measured learning result: the training machinery
(from-random init, hand-derived-and-finite-difference-verified gradients, a real Adam update) actually
reduces held-out error below the frequency-only floor, so the model learns character-order structure, not
just token statistics. The held-out curve is non-monotonic — it reaches its minimum at step 50 and rises
while training loss keeps falling: textbook overfitting of a 344-token corpus by a 984-parameter model.
That is a measured finding, reported as one.

**Limitation.** One seed, one tiny synthetic corpus, 78 held-out positions — a high-variance point
estimate, not a general result. No wall-clock/peak-memory figure is claimed (none was measured to stated
precision). The trained artifact is an *experiment*, not a language model in the product: it is not wired
behind the `LanguageModel` boundary, generates nothing usable, and has nothing to do with grounding,
citation or correction. The remote CI does not execute the training tests (numpy is an optional extra and
the CI workflow cannot be modified with the current credentials — see K-009); they are gated and run
locally/where the extra is installed.

**Conclusion (moderate).** Chakaso can now train and honestly measure a model from random initialization.
K-001 is narrowed for the *training loop* — there is now a real, measured model — but the headline gap
stands in its stronger form: no real *workload* and no useful language model have been demonstrated. The
next experiment the evidence justifies is a controlled held-out comparison (early stopping / capacity,
multiple seeds) to turn this point result into one with an error bar.

---

## 2026-10-19 — E2602: the training loop holds up across seeds under a proper early-stopping protocol

**Experiment E2602 — under early stopping selected on validation, what is the held-out (test) loss across
seeds, and does E2601's overfitting reproduce?** Protocol (fixed before the result, in
`chakaso.training.controlled`): three-way train/validation/test split (`split_three_way`, units
disjoint and preserved, vocabulary fit on training text only); the same 984-parameter character model and
Adam as E2601; 300 full-batch steps evaluated every 10; **stop at the first step achieving a new minimum
*validation* loss and report the test loss there**; seeds pre-declared as (7, 21, 1337, 2024, 42);
validation/test fractions 0.2 each. Environment as E2601 (Windows 11, CPython 3.13.5, numpy 2.4.6, CPU).
Record: [`../../experiments/E2602/record.md`](../../experiments/E2602/record.md).

**Result (measured, not assumed).** Mean held-out **test** cross-entropy at the validation-selected step:
**1.7978 nats, sample stdev 0.1767** across the five seeds (per-seed 1.5784–2.0655). Mean test loss at the
un-early-stopped step 300: **2.7804** — worse on every seed than the early-stopped value. Mean unigram
baseline (test): **2.9808**; uniform ~3.26–3.33. The validation-selected step is early on every seed (30–50).

**Interpretation.** E2601's single-seed result was not a lucky partition: under a protocol that selects
only on validation and reports on an untouched test set, every seed lands below both baselines, with a
spread (±0.18) much smaller than the margin over the unigram baseline (~1.18 nats). The overfitting E2601
showed reproduces as a *systematic* effect — a fixed 300-step budget overshoots the held-out optimum on
every seed — which is the evidence that early stopping, not more training, is the correct reading here.

**Limitation.** Still one tiny synthetic corpus (~250-line training partitions, 77–93 test positions);
five seeds reduce sampling noise but not the data's unreality, and with a 20% validation set the
selection itself is noisy. No wall-clock/memory measured. This is a language-modelling toy result with an
error bar; it says nothing about grounding, citation, correction or any real workload.

**Conclusion (moderate, stronger than E2601).** The from-scratch training loop genuinely reduces held-out
loss below trivial baselines, reproducibly across seeds. The project now has a measured, multi-seed
training result behind the boundary — and the evidence points away from repeating it and toward *breadth*:
a generation-quality measurement through the `trained_char` adapter, or the human-inspectable
grounded-answer evaluation corpus H1–H7 actually need. K-001 stands.

## 2026-10-19 — P27/E2603: a grounded-answer evaluation workload exists, and the first run finds retrieval over-returning

**Observation.** Until P27 every measurement scored machinery or a toy model; nothing scored a *grounded
answer* against a question with acceptable and unacceptable evidence. [ADR-0031](../decisions/ADR-0031-grounded-answer-evaluation-workload.md)
built the first such workload: a self-authored, provenance-bearing, held-out set of grounded-answer cases
(`ga-kestrel-library`, 13 passages / 14 cases) run by `chakaso benchmark grounded`, which composes the real
retrieval and `evaluate_answer` boundaries and reports retrieval and answer dimensions separately.

**Experiment E2603 — does the harness run the pipeline, and what does the first baseline reveal?** Protocol:
lexical retrieval, `top_k=5`; two answer providers over the identical workload — a **reference oracle** (an
oracle, not a model) that replays each case's required facts and cites only surfaced gold, and a **non-answering**
provider that abstains. Windows, CPython 3.13.5, offline, numpy-free. Record:
[`../../experiments/E2603/record.md`](../../experiments/E2603/record.md).

**Result (measured).** Reference oracle over all 14 cases: retrieval recall@k **0.8182**, MRR 0.8636, hit
rate 0.9091, **3 forbidden hits**, **3 false retrievals**; answer mean evidence coverage 0.8182, citation
validity 1.0, grounded-answer rate 0.8182, appropriate-abstention rate 1.0; zero stage failures. Held-out
split recall@k 1.0. Non-answering provider: retrieval identical, answered_cases 0, appropriate-abstention
rate 0.2143 (it over-abstains every answerable case).

**Interpretation.** The oracle arm validates the harness — a correct, well-cited, correctly-abstaining
answer scores as expected, and the metric arithmetic is pinned by independent tests. The informative part
is the negative result the separate dimensions refuse to hide: **lexical retrieval returns its full top-k
even on the three questions that have no answer in the corpus**, so a retrieval-aware runtime is handed
evidence it would have to abstain from, and it surfaces forbidden passages on three cases. A merged score
would have buried this.

**Limitation.** The workload is self-authored, tiny and author-written — easier than real text, and it
cannot support any claim about answer capability. No model participates: the answer side is an oracle and a
deliberate abstainer, so no number here is model quality (K-001, in its strong form, stands). No confidence
interval (the run is deterministic).

**Conclusion (narrow but real).** Chakaso now has a working grounded-answer evaluation capability and a
first measured baseline. The most valuable outcome is a concrete, evidence-driven next step rather than a
headline: retrieval lacks an evidence-sufficiency signal (recorded as
[K-010](../agent/KNOWN_ISSUES.md)), which is now the active unit (P28). The frontier of the project — a
real workload met by a model that answers — is unmoved.

## 2026-10-19 — P28/E2604: an evidence-sufficiency decision, measured against the E2603 baseline

**Observation.** E2603's failure analysis, case by case, showed the false retrievals were not a score
mystery: unanswerable questions retrieved passages that shared only function words (`a`, `to`, `on`, `and`,
`library`), while every answerable case's real evidence carried a rare content term. The first hypothesis —
a document-frequency threshold alone — was tested and **failed**: on a 13-document corpus even `to`/`on`/`and`
have df=1, so a pure df rule flagged the abstention cases as sufficient. The correction (require a
non-function-word term) is what made the signal mean what it claimed.

**Experiment E2604 — can a deterministic sufficiency policy separate "no evidence" from "k candidates"
without costing recall?** Protocol (ADR-0032): `DiscriminatingTermSufficiencyPolicy` (non-function-word,
low-document-frequency term overlap; `max_document_fraction=0.25`) added as an *additive seam* — retrieval
still returns top-k; only a decision is layered on. Baseline is E2603 with the policy off (byte-identical
code path). Reference oracle answers; no model. Record:
[`../../experiments/E2604/record.md`](../../experiments/E2604/record.md).

**Result (measured).** Split=all: sufficiency **accuracy 1.0** (14 assessed), **false abstentions 0**,
**missed abstentions 0** — all three E2603 unanswerable cases now judged insufficient, no answerable case
wrongly abstained. Retrieval recall@k (0.8182), MRR (0.8636), forbidden hits (3) and false retrievals (3)
are **unchanged**, because the policy does not touch retrieval. Held-out: sufficiency accuracy 1.0, recall@k
1.0. A parameter sweep found the policy is correct only for `max_document_fraction ≈ 0.23–0.31`: below it
the df=3 answerable `book` case is wrongly abstained; above it the df=4 noise term `library` leaks the
abstentions.

**Interpretation.** The decision layer works and is honest about its own basis (it names the terms behind
each verdict), and it is a genuine, measured improvement to whether the system can tell it should abstain —
*without* the cheap win of abstaining on everything (zero false abstentions is the required other half of
the trade-off). The additive design means retrieval still over-returns, so this is a decision Chakaso can now
make, not yet behaviour it acts on (P32).

**Limitation.** The narrow, corpus-specific working band is the central caveat: a threshold that only holds
for one 13-document vocabulary is **not a calibrated confidence**, and there is no evidence a band exists for
a larger or differently-worded corpus. A single-document corpus breaks it outright (asserted by test).
No model; no real corpus; K-001 stands. No statistical claim (deterministic, tiny).

**Conclusion (narrow but real).** P28 delivers a deterministic, inspectable evidence-sufficiency signal that
measures perfectly on the workload it was designed against, and honestly flags that "measures perfectly here"
is a weak claim. The next unit (P29) is to test whether it survives calibration and a harder, extended
workload — or to record, truthfully, that this corpus is too small to calibrate it at all.

---

## 2026-10-21 — The repository's shape is measured, and it is not the project's stated shape

**Observation.** Counted rather than remembered: 131 modules and 20,697 lines under `src/chakaso/`, 93
test modules, 32 decision records of which exactly one concerns training, 11 top-level CLI commands, and
two implementations behind `LanguageModel` — a development double and an adapter over a measured
984-parameter character model. Two of the four committed experiments are about a model.
([`../internal/engineering/model-centred-audit.md`](../internal/engineering/model-centred-audit.md))

**Interpretation.** The founding goal is to train Chakaso's own model, so this is a statement about
priorities rather than a bug report: the centre of mass was in the layers that serve a model which does
not exist. Two consequences made the drift load-bearing rather than cosmetic — application milestones
P4/P5/P10 cannot be completed without a model that generates, and every model decision kept being sized by
what the application left over.

**Conclusion.** The product contract is written down as
[ADR-0033](../decisions/ADR-0033-model-centred-product-contract.md): the model is the product, the Python
library is the primary interface, the CLI is a secondary utility whose handlers stay thin, and model
evaluation is kept separate from harness evaluation. It defers P29 and P32 by decision, not by drift.

---

## 2026-10-21 — The environment P22 assessed is not this environment

**Observation.** Re-inspected on 2026-10-21 (`[`next-model-experiment.md`](next-model-experiment.md)` has
the commands): 18 logical CPUs, 16.57 GB RAM with 3.70 GB free, no CUDA device, `torch` 2.13.0+cpu already
importable with 14 intra-op threads, `numpy` 2.4.0, ~168 GB free disk. P22's record of ~1.4 GB free and no
ML library installed was accurate on 2026-10-15 and is superseded as a *current* observation, not corrected.

**Result (measured).** A decoder-only Transformer at batch 32 × context 128 costs 0.444 s per optimizer step
with 774,656 parameters and 1.252 s with 3,190,784 — 9,230 and 3,271 tokens per second, CPU only. The
existing full-batch character loop costs 0.230 s per step at a 20,000-example batch and 1.868 s at 100,000,
and materializes a `(batch × vocabulary)` probability array: ~1.04 GB float64 at 650,000 positions.

**Result (measured, and it argues against the intuition).** Same shape, same dtype: 20 ×
(8192×256 @ 256×256) took 0.159 s in NumPy float32 and 0.137 s in torch float32. A framework buys 1.16×
on array multiplication on this machine. Cold import: `numpy` 0.25 s, `torch` 2.96 s.

**Limitation.** Peak process memory could not be measured: `K32GetProcessMemoryInfo`,
`GetProcessWorkingSetSize` and `tasklist` PID filtering all returned nothing from this session's shell, so
no RSS figure appears anywhere in this repository, and the claim is restricted to "the run completed on this
machine" rather than "it fit in N bytes".

---

## 2026-10-21 — The dependency decision is re-made on its own reopen conditions

**Observation.** [ADR-0030](../decisions/ADR-0030-minimal-optional-training-dependency.md) rejected
PyTorch "for now" and listed when to revisit it, including *a concrete workload a larger model would
improve* and *hand-derived gradients becoming a correctness risk*. Both now hold, on measurements: the
corpus is 69 tracked documents and 754,789 characters (counted here), which the full-batch loop cannot
train on; and verifying every scalar of a Transformer backward pass by central differences extrapolates to
~45 hours at the measured forward cost (984 scalars took ~1.13 s on the character model), against ~42 s for
a 200-scalar sample.

**Conclusion.** [ADR-0034](../decisions/ADR-0034-optional-pytorch-model-extra.md) admits `torch` as a
separate optional `[nn]` extra, leaves `[train]` carrying exactly numpy, confines torch imports to a named
module list enforced by test, pins checkpoint loading to `weights_only=True`, and scopes reproducibility to
this interpreter and CPU build rather than claiming cross-version bit-identity. The honest bookkeeping
matters as much as the decision: the speed argument did **not** survive measurement, and the accepted
verification guarantee is weaker than the character model's — sampled finite differences instead of every
scalar — which is written into the record and into the ADR rather than glossed.

---

## 2026-10-21 — The data layer is dependency-free on purpose, and the corpus is pinned by content

**Observation.** The model path's first three layers — `chakaso.training.manifest` (per-document SHA-256
over normalized text, licence, source, drift refusal), `split_text_units` (whole documents, deduplicated by
content *before* selection) and `chakaso.training.subword` (byte-level BPE with a named pre-tokenizer,
deterministic tie-breaking and a self-describing checksum-verified record) — import neither numpy nor
torch, so all 98 of their tests run in the dependency-free CI matrix instead of inheriting K-009's gap.

**Result (measured).** Preparation of the real corpus: 69 documents, 754,789 characters, manifest
fingerprint `0f9f7d2f57644a4d`, a 49/10/10 document split with no duplicate content, and a 1,024-entry
vocabulary trained in 20.3 s over 178,252 pieces that compresses the training side to 234,960 tokens at
2.12 characters per token, with 82,663 validation and 38,982 test tokens.

**Interpretation.** Two properties are what make a later model number mean anything: the corpus cannot move
silently (a changed, deleted or newly-matching file is a typed refusal), and the held-out sides cannot leak
(a duplicate document is dropped before selection because otherwise it both trains twice and straddles the
boundary; a leakage test with per-document unique markers runs over 24 seeds and counts violations).

**Limitation.** The corpus is one author's technical prose about one project, so every number measured on it
describes that distribution — including the house style, which a model can memorize and still be useless.
