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
