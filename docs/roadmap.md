# Roadmap

The roadmap's rule: build the system before scaling the model, and require each
phase to produce a working artifact plus evidence that the next phase is
justified. Phases are not dates. A phase is done when its exit criteria are met,
and the criteria are written so that a phase cannot be declared done by assertion.

## Phases

| Phase | Milestone | Exit criteria | Status |
| --- | --- | --- | --- |
| P0 | Specification | Architecture, interfaces, constraints and evaluation plan written down | Done |
| P1 | Repository foundation | Package builds, tests and static checks pass in CI, decisions recorded | Done |
| P2 | Local conversational shell | Model interface with a local implementation; conversation state; CLI that holds a conversation | Done (over the deterministic double — no real model) |
| P3 | Local retrieval | Documents can be ingested, indexed and retrieved as evidence with stable identifiers | Done (lexical; dense/hybrid are experimental fixtures) |
| P4 | Web-grounded answers | Permitted public retrieval produces source-linked answers with verified citations | Partial — bounded opt-in fetcher and narrow HTML ingestion exist; no default path fetches, and no model turns retrieved evidence into a cited answer |
| P5 | Correction loop | Challenge → reassessment → correction works on fixed tests | Partial — the decision rule, reassessment over a stored answer, an explicit identifier trigger and a correction development benchmark pass on synthetic tests; no revised prose (no model) and no autonomous loop |
| P6 | Evaluation harness | Automated grounding, citation, retrieval and conversation benchmarks | Partial — retrieval, correction, strategy-comparison and semantic-evaluation development benchmarks plus structural claim/citation/grounding/answer evaluation exist over synthetic fixtures; no real-workload or model-quality benchmark |
| P7 | Training pipeline | Tokenizer, data, configuration and checkpoint pipeline runs end to end | Partial — E2601 ran a character tokenizer, a leak-free split, a validated config, an optimizer, a checkpoint format and a training loop end to end on a 344-token synthetic corpus; a real/subword tokenizer and scale are not built |
| P8 | Chakaso-0 | Tiny model trains from random initialisation and learns a controlled corpus | Achieved as a measured experiment — E2601 trained a 984-parameter character model from random init that lowers held-out loss below uniform/unigram baselines; it is a learning-loop demonstration, not a useful model, and the single-seed high-variance result is flagged as such |
| P9 | Chakaso Base | Small model reaches useful language-model baselines | Not started |
| P10 | Chakaso Grounded | Instruction, grounding and correction behaviour improve on internal benchmarks | Not started |
| P11 | Domain selection | A defensible niche is chosen using explicit criteria | Not started |
| P12 | Domain adaptation | Specialized dataset and benchmark established | Not started |
| P13 | Research release | Model, code, evaluation and limitations published together | Not started |

P0 and P1 are deliberately boring and deliberately first. Most of what makes a
research project unreproducible is decided in the first week.

### Milestone reconciliation (assessed against exit criteria, not file presence)

A phase is "Done" only when its exit criteria hold on evidence, not when a related
file exists. The status column above is deliberately conservative; this records the
reasoning, because a multi-capability milestone can be half-met and must not be read
as fully met.

- **P1 — Done.** Every P1 detail below this section holds: the package installs from
  a clean checkout, formatting/lint/types/tests run in CI on 3.11–3.13 without secrets
  or network, configuration is externalized/typed/validated, the `LanguageModel`
  boundary has a contract and contract tests with no business logic importing a
  concrete adapter, source/evidence records carry stable identifiers and content
  hashes, and the agent context matches reality.
- **P2 — Done, with the caveat that defines the whole project.** A conversation is
  held end to end through the retrieval-aware runtime and CLI (`chakaso chat`), state
  is immutable and append-only, and the model boundary has a local implementation.
  That implementation is the deterministic development double: the *plumbing* of a
  conversational shell exists; a model that answers does not. P2 is met as a shell and
  must not be read as a working language model.
- **P3 — Done for lexical retrieval.** Normalization, section-bounded chunking, local
  ingestion, an in-memory corpus and a deterministic BM25 retriever rank into an
  evidence pack with stable content-derived identifiers, reachable from `chakaso
  retrieve`. Dense and hybrid retrievers exist behind the same boundary but over a
  declared non-semantic fixture embedding, so they satisfy the *interface* of P3, not
  a semantic claim; they are recorded as Experimental.
- **P4 — Partial, and the gap is the model.** The exit criterion needs *source-linked
  answers with verified citations* from *permitted public retrieval*. The transport
  (bounded, opt-in fetcher) and the narrow HTML reader exist, and citation *validation*
  is structural; but no default path fetches, there is no durable source index, and
  with no model nothing generates a cited answer. The blocking absence is generation,
  not fetching.
- **P5 — Partial: it decides and records, it does not rewrite.** The correction
  decision rule, application-level reassessment over a stored answer, the explicit
  `ans_…` trigger composed into the runtime, and a correction development benchmark all
  pass on synthetic tests — so "challenge → reassessment → decision" holds on fixed
  tests. What the exit criteria do not yet cover is a corrected *answer*: there is no
  model to write revised prose, and reassessment is an explicit, identifier-triggered
  operation, not an autonomous loop. That is a deliberate boundary, not a missing flag.
- **P6 — Partial: instruments exist, a real workload does not.** Retrieval, correction,
  strategy-comparison and semantic-evaluation development benchmarks and the structural
  evaluation primitives all run and are reproducible (`chakaso.experiments`,
  `chakaso benchmark`). They measure mechanisms over hand-built fixtures; the exit
  criterion's "conversation benchmark" against a real model and real evidence is not met
  because neither exists (K-001).
- **P7 — Not started, correctly.** No tokenizer, data pipeline, model definition or
  checkpoint pipeline exists. This is the milestone the current active task (P23, the
  dependency decision) gates, and it must not be marked started before that ADR is
  accepted.

## P1 detail: what "foundation" means

The current phase is complete when all of the following hold:

- The package installs from a clean checkout and the tests pass.
- Formatting, linting, type checking and tests all run in CI without secrets and
  without network access to a model provider.
- Configuration is externalized, typed and validated, with clear failures on bad
  input.
- The language-model boundary exists with a documented contract and contract
  tests, and no business logic imports a concrete implementation.
- Source and evidence records exist with stable identifiers and content hashes.
- Documentation states what is implemented, what is planned and what does not
  exist, and the agent context matches reality.
- Every decision of consequence has a record explaining it.

## P2 detail: the first thing that behaves like a system

P2 is the first phase where Chakaso does something. Its scope is deliberately a
shell: a conversation can be held, state persists across turns, and a local model
answers through the boundary. It does not retrieve, does not cite, and does not
correct.

The reason to build the shell before retrieval is that the shell defines the state
that retrieval and correction will attach to. Building retrieval first would mean
retrofitting conversation state around evidence handling, which is backwards.

## First implementation sprint

The founding work list, now assessed against what actually shipped:

1. Configuration system with typed, validated configuration. — **Done.**
2. Core primitives: identifiers, content hashing, errors. — **Done.**
3. `LanguageModel` interface, capability declaration, registry. — **Done** (interface and
   registry; the only implementation is the development double).
4. One local model adapter, chosen for being runnable on CPU. — **Done in the honest
   sense only:** a CPU-runnable adapter exists behind the boundary, but it is the
   deterministic double, not a language model. A *trained* adapter is P7/P8.
5. Conversation state and the Conversation Manager. — **Done** (transactional, append-only).
6. Source and evidence records with URL normalization. — **Done** (typed references, ADR-0011).
7. Citation resolution with rejection of unknown identifiers. — **Done** (ADR-0009).
8. Local document ingestion and a lexical retrieval baseline. — **Done** (`chakaso retrieve`).
9. A retrieval interface for public web access that does not require a commercial search
   API. — **Partial:** a bounded, opt-in fetcher exists (ADR-0012) and never runs on a
   default path; deciding *which* URLs to fetch (query planning for web discovery) is open.
10. Answer generation with source-identifier citations. — **Not done:** the plumbing
    records citations, but no model generates an answer to cite into. Gated on P7/P8.
11. Reassessment entry point. — **Done** (application-level reassessment over a stored
    answer; explicit identifier trigger, ADR-0029).
12. Unit and integration tests for all of the above. — **Done** (see `tests/`, 1258 at
    the current head; 1250 in the dependency-free matrix, 8 numpy-gated).
13. A hand-built evaluation set of modest size, with unacceptable-evidence fields. —
    **Done for development fixtures** (synthetic retrieval and correction benchmarks with
    forbidden-evidence fields) and, since P27, for a **grounded-answer evaluation workload**
    (`chakaso benchmark grounded`, ADR-0031): a self-authored, provenance-bearing, held-out set of
    answer cases with required facts, prohibited claims and unacceptable-evidence fields, run and
    measured by E2603. A *real-world* evaluation set (a corpus the project did not write) is P6's
    open half and still stands.
14. A CLI that holds a conversation. — **Done** (`chakaso chat`, grounded through the runtime).

The founding sprint is complete except for the two items that depend on a real model
(9's discovery half, and 10). Those are not CLI gaps; they are the model gap that the
next sprint addresses.

## First model-training sprint

The active milestone. The dependency decision (P23) is recorded in
[ADR-0030](decisions/ADR-0030-minimal-optional-training-dependency.md) — `numpy`, in an optional
`[train]` extra, chosen *before* any model code, per ADR-0001. The sprint steps, now assessed against
what E2601 actually did ([`../experiments/E2601/record.md`](../experiments/E2601/record.md)):

1. Collect a small corpus whose licence permits use. — **Done** (self-authored, Apache-2.0).
2. Build a tokenizer and inspect its segmentations by hand. — **Done** (deterministic character vocab).
3. Implement the tiny model and verify tensor shapes with tests. — **Done** (`chakaso.training.model`).
4. Overfit a tiny dataset on purpose, to prove the training loop can learn at all. — **Done** (training
   loss falls to ~0.25; the loop provably learns — and, as measured, overfits past step 50).
5. Train a small experiment with a held-out split. — **Done** (leak-free split, held-out curve recorded).
6. Log loss, throughput, memory and checkpoints. — **Partial**: loss curve and a fingerprinted checkpoint
   are recorded; throughput/peak-memory were not measured to a stated precision, so none is claimed.
7. Compare against a trivial baseline (a uniform or unigram predictor). — **Done**: held-out loss beats
   both baselines.

Scaling happens after reproducibility, not before. The first experiment demonstrated the learning loop and
produced the project's first genuinely measured number; it is not a step toward a useful chatbot, and the
measured overfitting is itself the evidence that justifies the next, controlled experiment (early
stopping / capacity, multiple seeds).

## Grounded-answer evaluation workload (P27)

The training sprint narrowed K-001's *training-loop* half but left the project's real open question
untouched: is there anything a grounded answer could be measured against? P27 answered that with
[ADR-0031](decisions/ADR-0031-grounded-answer-evaluation-workload.md) and built it — the first
evaluation workload whose units are grounded-answer cases, not retrieval fixtures.

1. A provenance-bearing, human-inspectable case/schema with unacceptable-evidence and prohibited-claim
   fields. — **Done** (`chakaso.benchmark.grounded_cases`).
2. A legally-usable, offline, self-authored source set (no third-party corpus assumed licensed). —
   **Done** (`ga-kestrel-library`, Apache-2.0).
3. Split integrity — no passage grounds both a development and a held-out case. — **Done**, enforced by
   the dataset and the loader, and pinned by tests.
4. An evaluation runner over the real retrieval + answer boundaries, separating retrieval, generation
   and evaluation failures, reporting dimensions separately. — **Done** (`chakaso benchmark grounded`).
5. A first measured baseline, with its negative findings recorded, not hidden. — **Done**
   ([`../experiments/E2603/record.md`](../experiments/E2603/record.md)).

E2603's honest result is that the harness works and that **lexical retrieval over-returns**: it produced
a false retrieval on every abstention case and hit forbidden evidence on three. That measured failure,
not appetite, sets the next sprint — a retrieval evidence-sufficiency decision (P28, tracked as K-010).
The *real-world* evaluation corpus (item 13's open half) and a model that actually answers both remain
out of reach and unclaimed.

## Domain-selection gate

The domain decision is deliberately deferred. When it is made, candidate domains
are scored against explicit criteria:

- availability of usable data,
- difficulty for general-purpose models,
- potential for proprietary or hard-to-replicate data,
- whether outcomes are measurable,
- access to practitioners or domain data,
- commercial value,
- research value,
- competitive landscape,
- whether the work can start cheaply.

A niche is not selected because it sounds impressive. The gate exists so that the
choice can be defended later.

## Long-term target

```text
                     Chakaso System
                           |
          +----------------+----------------+
          |                                 |
    Chakaso Model                  Domain Intelligence
          |                                 |
   Base + Instruct                  Retrieval + Tools
   + Grounding                      + Data + Memory
          |                                 |
          +----------------+----------------+
                           |
                    Evaluation Core
                           |
                     Feedback Loop
                           |
                    New Training Data
```

The loop at the bottom is the part that matters. The system's own measured
failures are the source of the next training data, which is why evaluation is
built before the model is scaled.

## Release discipline

- Every model release has a model card.
- Every dataset release has a data card and a licence note.
- Every benchmark result records configuration, dataset and version information.
- Every correction-behaviour change has regression results.
- Limitations are published rather than omitted.
- Prototype benchmark performance is not presented as general capability.
