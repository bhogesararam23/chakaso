# Open questions

Questions with no committed answer. Some of these block work; the blocking ones say
so. This file is not a wish list — an entry appears because a decision is needed
and cannot currently be made well.

---

## Blocking: retrieval

### How should the first web fetch mechanism work?

The constraint is that the fetch path must not require a commercial search API
([ADR-0001](../decisions/ADR-0001-local-first-runtime.md)). Candidate mechanisms:

- a public search endpoint that does not require an account,
- a curated seed list of domains plus site-local search,
- sitemap or feed discovery from a set of trusted domains,
- a manually assembled corpus for the first evaluation, deferring live fetching.

**Why it is undecided.** Each option trades coverage against operational risk, and
the choice affects what the first benchmark can measure. Building a crawler before
there is a question it must answer would be guessing.

**What would resolve it.** A concrete first evaluation question. If the first
evaluation set is hand-built, a curated corpus is sufficient and live fetching can
wait.

### What is the fetch policy?

Rate limits, robots handling, caching, user-agent identification, maximum response
size, redirect depth. These are policy values, not research questions, but they
must be written down before any fetch happens, and they do not exist yet.

---

## Blocking: generation contract

### Inline citations or a structured side channel?

Should the model write evidence references inside its answer text, or return the
answer and its references as separate structured fields?

Inline is simpler to prompt and harder to validate; a model can produce a
plausible-looking reference that is subtly wrong. A side channel is more
machine-checkable but imposes a structure on the model's output that small models
handle poorly.

**Why it is undecided.** The answer probably depends on measuring both against the
grounded-answer benchmark, which does not exist. Deciding it from intuition risks
encoding an assumption into every prompt template.

**What would resolve it.** A small hand-built comparison once the benchmark exists.

### How should claim-level support be established?

Checking whether cited evidence supports a claim requires decomposing an answer
into claims and judging each against the evidence. A model can do this, but a
model judging its own output is a weak instrument. The alternatives are human
judgement (accurate, slow), a separate model (fast, unvalidated), or a
retrieval-style overlap measure (fast, crude, and easy to fool).

**Why it is undecided.** Whichever is chosen becomes the definition of the
project's headline metric, so it should be chosen against a human-labelled sample
rather than on convenience.

---

## Blocking: first local model

### Which local model should the shell use?

Requirements: runnable on CPU, permissively licensed, small enough to iterate
against, with a tokenizer whose behaviour is understood.

**Why it is undecided.** It depends on the first evaluation question, and on
whether the shell needs to be *useful* or merely *present*. A model chosen for
usefulness and a model chosen for iteration speed are different models, and picking
one to serve both would serve neither.

---

## Not blocking: design

### Where should conversation state be persisted?

In memory for the first shell, then what? The type exists without a store. This is
deferred because the shape of the state will change once retrieval and correction
attach to it, and a store designed now would be designed against a guess.

### What is the artifact policy for weights and datasets?

Recorded as an open question in
[`../internal/engineering/artifact-policy.md`](../internal/engineering/artifact-policy.md).
Weights are too large for Git; the storage mechanism is undecided. Not blocking
until a checkpoint exists.

### Does the correction path need its own model call, or a rule-based comparison?

Rules are auditable and cheap; a model call is flexible and harder to trust. Both
may be needed, with rules for the structural cases and a model for the rest. Not
decided.

---

## Research

### Which domain eventually justifies specialization?

Deliberately deferred
([roadmap](../roadmap.md#domain-selection-gate)). Scored against explicit criteria
when the decision is made, not before. Nothing in the current architecture assumes
an answer.

### Is preference optimization worth its complexity here?

DPO is a plausible later stage. Whether it earns its place depends on whether
supervised behaviour plateaus in a way that preference pairs can address, which
cannot be known before there is a supervised model and an evaluation of it.

### Can calibration be done honestly at this project's scale?

Calibration needs held-out data with outcome labels. Whether a small project can
produce enough of it for a trustworthy reliability diagram is unknown. Until it is
known, no numeric confidence is displayed anywhere
([transparency](../transparency.md#what-is-withheld)).
