# Hypotheses

Each hypothesis is written so that it could be shown to be false, and each names
what would falsify it. A hypothesis that cannot fail is a preference.

None of these has been tested. None has supporting data. They are listed because
they determine what gets built: a hypothesis that nothing in the repository is
designed to test is a hypothesis that will never be tested.

---

## H1 — Explicit evidence tracking reduces unsupported claims

**Hypothesis.** Tracking which evidence supports which claim, as structured state
rather than as prose the model produces, reduces the rate of unsupported claims
compared with asking the same model to be careful in a prompt.

**Falsified if.** On a fixed question set with a fixed model and retrieval
configuration, unsupported claim rate is not lower with structured evidence
tracking than with prompt-level instruction alone.

**Why it matters.** This is the project's central engineering bet. If prompt
instructions achieve the same result, the architecture is unnecessary overhead.

**Depends on.** Claim decomposition, claim-level support judgement, and a question
set with evidence where support is known. None exist.

---

## H2 — A dedicated reassessment path reduces unjustified persistence

**Hypothesis.** Separating reassessment from generation — taking the previous
answer and new evidence as explicit inputs and classifying claims against the
evidence — reduces unjustified persistence below what a single model pass achieves
when asked to reconsider.

**Falsified if.** Unjustified persistence rate is not lower with the dedicated path
than with single-pass reconsideration, at equal rates of unnecessary revision.

**Why it matters.** Correction is the project's second central claim. A model that
agrees with every challenge would score well on persistence and badly on
unnecessary revision, which is why both metrics are required.

**Depends on.** A correction benchmark with both contradiction and
non-contradiction cases. Not built.

---

## H3 — Small models with strong retrieval can produce useful grounded answers

**Hypothesis.** A small local model can produce answers that a user finds useful
when retrieval quality is high, even though its parametric knowledge is weak.

**Falsified if.** On the grounded-answer benchmark, answer usefulness does not
exceed a retrieval-only baseline (returning ranked evidence with no generated
answer) by a margin a reader would consider meaningful.

**Why it matters.** It determines whether the project should spend its effort on
the model or on retrieval. If retrieval-only is nearly as good, that is a finding
worth publishing rather than a failure.

**Depends on.** A retrieval implementation, a usefulness judgement procedure, and
a baseline. None exist.

---

## H4 — Retrieval does not remove the need for uncertainty behaviour

**Hypothesis.** Adding retrieval to a model reduces, but does not eliminate,
unsupported claims; appropriate abstention remains a distinct behaviour that must
be built and measured separately.

**Falsified if.** With retrieval in place, unsupported claim rate on
insufficient-evidence cases falls to a level where abstention training adds no
measurable improvement.

**Why it matters.** It is the reason uncertainty is treated as its own evaluation
layer rather than as a property of retrieval. The hallucination survey cited in
[`literature.md`](literature.md) reports that retrieval mitigates but does not
solve the problem; that is background, not evidence about this system.

**Depends on.** A retrieval implementation and an abstention benchmark.

---

## H5 — Interaction logs can be used as training data without teaching self-imitation

**Hypothesis.** Logged interactions, filtered by measured outcome (supported
claims, successful corrections, appropriate abstentions), can be used as training
data without the model learning to reproduce the mistakes that remain in the log.

**Falsified if.** A model trained on unfiltered logs performs no worse — or
performs better — on unsupported claim rate than one trained on outcome-filtered
logs.

**Why it matters.** It decides whether the feedback loop in the roadmap is
realistic or whether it quietly degrades the model while appearing to improve it.

**Depends on.** An evaluation harness that can label outcomes, a trained model to
train further, and enough logged interactions to matter. None exist.

---

## H6 — Tokenizer choice measurably affects small-model quality

**Hypothesis.** Vocabulary size and segmentation rule (BPE versus Unigram) change
the measured quality of a small model on a fixed corpus and evaluation, by more
than run-to-run variation at a fixed seed.

**Falsified if.** Quality differences between tokenizer configurations fall within
seed-to-seed variance.

**Why it matters.** It determines whether tokenizer work is a research variable or
a settled engineering default. If it is not a research variable, the project
should stop treating it as one.

**Depends on.** A tokenizer, a model, a corpus, and enough compute to run
configurations more than once. None exist.

---

## H7 — A real dense (semantic) retriever exceeds lexical retrieval on paraphrased evidence

**Hypothesis.** A genuine embedding model, dense retrieval over it, retrieves paraphrased gold
evidence that lexical BM25 misses, raising recall@k above the lexical baseline on a corpus whose
gold chunks match the query in meaning but not in surface tokens.

**Falsified if.** On such a corpus, dense recall@k is not above lexical's — or any gain disappears
once gold chunks share no surface token with the query.

**Why it matters.** It is the research reason to build dense and hybrid retrieval at all. If a real
embedding does not beat lexical on paraphrase, the added machinery earns nothing and the project
should stop treating dense retrieval as a path to better grounding.

**Depends on.** A real (semantic) embedding model and a paraphrase-rich gold corpus. Neither exists.
The fixture embedding built in [ADR-0023](../decisions/ADR-0023-dense-retrieval-boundary.md) is
explicitly non-semantic, so the retrieval-strategy comparison run today
([research-log E-001](research-log.md)) does **not** test this hypothesis — over the fixture it
showed no gain and a small MRR regression. H7 remains untested.
