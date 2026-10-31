# Open questions

Questions with no committed answer. Some of these block work; the blocking ones say
so. This file is not a wish list — an entry appears because a decision is needed
and cannot currently be made well.

---

## Blocking: retrieval

### What is the identity of a document that has no URL?

**Answered. See [ADR-0011](../decisions/ADR-0011-typed-source-references.md).**

A source is identified by a typed `SourceReference`, not only by a canonical URL. A
web reference keeps `canonicalize_url` as its canonical form, so web identifiers are
unchanged; a `file` reference is a `file` URI built from the resolved path; a `text`
reference marks content with no location. The fetch permission stays inside
`canonicalize_url` and is used only for web references, so widening identity did not
widen what the future fetcher may retrieve — which is exactly the distinction this
question was about: a reference is an identity, a fetchable URL is a permission.

The historical framing is kept below because it records what was rejected and why.

Source identity was derived from the canonical URL and a content hash
([ADR-0007](../decisions/ADR-0007-content-derived-identifiers.md)), and
`canonicalize_url` accepts only `http` and `https` because those are the schemes the
fetcher may retrieve. A local file has neither. Its honest identity is its absolute
path plus a content hash.

The candidates, none of them free:

- **A `file://` reference.** Standard, and what the scheme exists for. The cost is
  that `canonicalize_url`'s scheme check currently means "may be fetched", and
  widening it would quietly widen what the future fetcher is allowed to retrieve.
  Keeping identity and fetchability in one list is how a security check becomes a
  normalizer.
- **A separate reference type for local documents.** Honest about the distinction,
  but it means `SourceRecord` holds either a URL or a filesystem path, and every
  consumer learns to handle both.
- **Refuse local documents and require a URL.** No new decision, but it makes
  ingestion depend on fetching, so the retrieval layer cannot be built or tested
  before the fetch policy exists.

**What would resolve it.** Decided in
[ADR-0011](../decisions/ADR-0011-typed-source-references.md): a source reference and a
fetchable URL are separate concepts — one is an identity, the other a permission.
Local ingestion is no longer blocked on this question.

**Why it is recorded instead of guessed.** ADR-0007 makes the canonicalization rule
part of the identifier contract, so changing it changes every future identifier. A
decision that broad should not be made as a side effect of a file-reading module.

### How should the first web fetch mechanism work?

The constraint is that the fetch path must not require a commercial search API
([ADR-0001](../decisions/ADR-0001-local-first-runtime.md)). Candidate mechanisms:

- a public search endpoint that does not require an account,
- a curated seed list of domains plus site-local search,
- sitemap or feed discovery from a set of trusted domains,
- a manually assembled corpus for the first evaluation, deferring live fetching.

**Why it is undecided.** Each option trades coverage against operational risk, and
the choice affects what the first benchmark can measure. Building a crawler before
there is a question it must answer would be guessing. **Note:** this question is now
only about *discovery* — which URLs to fetch. The *transport* for a URL that has been
chosen exists as a bounded, opt-in fetcher (ADR-0012); deciding *whether* and *what* to
fetch is query planning, still open.

**What would resolve it.** A concrete first evaluation question. If the first
evaluation set is hand-built, a curated corpus is sufficient and live fetching can
wait.

### What is the fetch policy?

**Answered. See [ADR-0012](../decisions/ADR-0012-optional-bounded-fetcher.md).**
`FetchPolicy` declares and validates the scheme allowlist, response-size cap, redirect
depth, timeout and a header-safe user-agent, and `HttpFetcher` enforces them, tested
offline. What is still open: caching, robots handling, per-host rate limits, and a
DNS-rebinding-safe transport (connecting to a pre-validated IP). These are recorded as
K-008 rather than claimed as solved.

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

**Narrowed, not answered, by** [`model-feasibility.md`](model-feasibility.md): under the
inspected constraints (no CUDA, ~15.4 GB RAM with little free, no ML dependency in a
dependency-free environment) the binding gate is not architecture but the decision to admit
an *optional* numerical dependency (its own ADR) and a memory budget. The assessment defers the
pretrained adapter (Option A) until a real grounded-answer evaluation exists, and points to a
scoped from-scratch training experiment (Option B) as the smallest honest next step.

**Narrowed further by** [ADR-0030](../decisions/ADR-0030-minimal-optional-training-dependency.md):
that gate is now decided — a from-scratch experiment will use **NumPy as an optional `[train]`
extra**, with hand-derived, finite-difference-verified gradients, keeping the base install
dependency-free and the core CI matrix unchanged. This fixes the *numerical substrate* of the
first training path; it does **not** select a model, a size, or a framework for anything larger.
The specific model (char-level toy now; a real or pretrained model later) remains open.

---

## Not blocking: design

### Where should conversation state be persisted?

**Answered for now, and deliberately so.** The conversation manager keeps state in
memory and returns a new immutable `Conversation` on each successful turn. Nothing
is written to disk.

**Why that is enough for now.** A store designed before retrieval and correction
attach to the state would be designed against a guess about its shape. The state
type is immutable and append-only, so adding persistence later means writing a
store, not changing what is stored.

**What would reopen it.** A conversation that has to survive a process restart, or
an experiment that needs transcripts as data. Both are real, neither is current.
The consequence to keep in view is that a session is lost on exit, and the CLI
should say so rather than implying otherwise.

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

## Questions the 2026-11-01 campaign could not close

**Does a 0.02-bit materiality band mean anything without a noise floor?**
The band is inherited from E2611's declaration, not calibrated. E2612's H1 cleared it 13.8×, which no plausible
seed noise explains; its H2 missed it by 1.5×, which a single draw easily could. The project has a five-seed
spread for the *character* model (E2602: mean test loss 1.80 ± 0.18 nats) and **none for the decoder at any
size** — E2609 was never run. Until a seed spread exists at the reference configuration, every "inside the band"
in this repository is an absence of evidence about a threshold that has not been measured.
*What would answer it:* three seeds of the E2612-T configuration, ~4.5 hours, reporting the standard deviation of
test bits per character.

**Does the mixture damage prose, or did arm T simply read less prose?**
Declared as a confound before E2612 ran, then measured: the training pool is 50.51 % prose by characters
(52,513,592 of 103,964,830 over 6,515 documents), so at a shared 20,000,000-identifier cap the mixture arm saw
roughly 10.1M prose identifiers against the control's 20.0M. The two explanations of the +0.03026 are
indistinguishable from inside this design, and the honest answer is H8's third arm rather than more argument.

**What does Chakaso's text look like on its own domain?**
Unanswerable from the current artifacts: every generation sample in E2612 continued this repository's section
headings while the models trained on Gutenberg (K-026), so the only qualitative evidence about two external-corpus
models describes a corpus they never saw. The mechanism to fix it is implemented and seeded
(`--sample-prompts held_out`); the measurement does not exist.

**Which records are re-trainable without a git-ignored cache?**
The campaign established the boundary's *existence* — E2608's vocabulary cannot be re-fitted by the code that
shipped between `6f3274f` and `cfdcd1a`, and E2612's cannot be re-fitted by the code after it — but not which
records survive a clean clone. [K-017](../agent/KNOWN_ISSUES.md) already notes that half the measured records
carry no machine-readable evidence in the repository. Whether a vocabulary that a measured result cites is an
artifact or a build input is an unresolved policy question under
[artifact-policy.md](../internal/engineering/artifact-policy.md), not a bug.

**How much does an exact byte-pair fit actually cost at the sizes the plan needs?**
Answered for the sizes measured, still open beyond them. 1,002,483 characters of the multi-register corpus fit
4,096 identifiers in 6.6 s, 4,014,828 in 494 s, and the 12,663,402-character E2612 sample in **2,116.4 s
(35.3 minutes)** — against the 29.2 s the fast-but-wrong loop measured at 4.0M and the 145.55 s E2612 reported for
*all* of its preparation. The curve is superlinear, so anything larger than 12.7M is unknown and probably worse.
Every preparation estimate in this repository's records was made against the wrong loop, which is the same class
of error as the smoke that was 11–14 % optimistic on throughput.

**Can the model's distribution be judged by anything other than its loss?**
The campaign strengthened the question rather than answering it. E2612's better arm produced *more* repetitive
text (`looping_fraction` 0.4 → 0.6; mean bigram repetition 0.675527 → 0.712437), and E2610 already found that no
decoding setting writes coherent text from the released checkpoint. So a held-out loss improvement and a
generation-quality readout are not merely weakly correlated here — they moved in opposite directions. What
instrument could detect a real improvement in *what the model has learned*, given that both current instruments
disagree, is the open methodological problem, not the open engineering one.

**Is archaic English the binding constraint, or 20M tokens?**
Both records that could separate them are silent. E2612 says a result on this corpus "does not transfer to modern
web text without a new corpus and a new record", and the data review found that Gutenberg cannot date its own
register by construction (`Issued` is the release date; 60,360 of 61,900 English works show ≥ 2000). But no
partition in any run is modern, so the corpus's *age* has never been measured — only assumed. The cheapest
disambiguation is a matched-token arm on the licence-clean newspaper corpus, scored on a held-out-era partition;
that experiment would also say whether 20M tokens is enough to see a register effect at all.

**Should a re-run that cannot reproduce a digest be called a reproduction?**
The artifacts say report digests are file hashes over bodies that include wall clock, throughput, peak RSS and
thread count, so a faithful re-run reproduces every metric and no digest ([K-028](../agent/KNOWN_ISSUES.md)).
`chakaso experiment model`'s own wording invites the hashes to be compared. Either the tool must emit a
content-only digest over the metric-bearing fields, or the project must state plainly that reproduction means the
numbers agree and treat digests as identifiers. Right now a reader can be misled in either direction, which is
worse than an inconvenient hash.
