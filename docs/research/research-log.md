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
