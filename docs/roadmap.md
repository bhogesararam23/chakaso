# Roadmap

The rule that ordered this roadmap — build the system before scaling the model,
and require each phase to produce a working artifact plus evidence that the next
phase is justified — was reversed by
[ADR-0033](decisions/ADR-0033-model-centred-product-contract.md): the model is now the
product and the system is one of its consumers, because P4, P5 and P10's remaining exit
criteria all need a model that generates, and scheduling them ahead of it kept deferring
the thing the project exists to build. The rest of the discipline is unchanged: a phase
is done when its exit criteria are met on evidence, never by assertion, and the criteria
are written so a phase cannot declare itself done.

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
| P7 | Training pipeline | Tokenizer, data, configuration and checkpoint pipeline runs end to end | **Done** — E2605 ran the whole pipeline end to end on a real, content-pinned corpus (71 documents): manifest → document split → learned byte-pair subword tokenizer → sequence packing → Transformer → mini-batch AdamW with warmup, clipping, early stopping and bit-identical resume → checksummed checkpoint → held-out metrics → `load_model`. What it does *not* include is scale: the corpus is one author's prose and the run was one seed |
| P8 | Chakaso-0 | Tiny model trains from random initialisation and learns a controlled corpus | Achieved as a measured experiment, twice over — E2601 trained a 984-parameter character model from random init that lowers held-out loss below uniform/unigram baselines, and E2605 trained a 643,840-parameter subword Transformer that reaches 1.8657 bits per character on held-out documents against 3.1343 for the character model. Both are learning-loop demonstrations, not useful models, and both are single-corpus results without an error bar |
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
  criterion's "conversation benchmark" against real evidence is not met, and no model
  produces an answer in any default path to benchmark (K-001). Model evaluation is now a
  separate instrument with its own measured numbers (`chakaso.training.model_metrics`,
  E2605), which is exactly the distinction ADR-0033 rule 5 requires.
- **P7 — Done, and only after its own dependency decision was re-made on measurements.**
  When this line read "Not started, correctly" there was no tokenizer, model definition or
  checkpoint pipeline. There are now two of each, plus a content-pinned corpus manifest and
  a training loop that resumes bit-identically, and E2605 ran all of it end to end on real
  text. The gate it waited on — [ADR-0030](decisions/ADR-0030-minimal-optional-training-dependency.md)'s
  revisit conditions — was met and re-decided as
  [ADR-0034](decisions/ADR-0034-optional-pytorch-model-extra.md) and
  [ADR-0035](decisions/ADR-0035-transformer-checkpoint-format.md) rather than assumed away.

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
   registry, with three adapters behind it: the development double, a character model and a
   trained subword Transformer).
4. One local model adapter, chosen for being runnable on CPU. — **Done.** A CPU-runnable
   adapter exists behind the boundary and it is still the deterministic double on every default
   path; a *trained* adapter now exists too (P31, the continuation of P7/P8 —
   `chakaso.models.load_model` over E2605's checkpoint), it is measured rather than useful,
   and nothing wires it into a turn.
5. Conversation state and the Conversation Manager. — **Done** (transactional, append-only).
6. Source and evidence records with URL normalization. — **Done** (typed references, ADR-0011).
7. Citation resolution with rejection of unknown identifiers. — **Done** (ADR-0009).
8. Local document ingestion and a lexical retrieval baseline. — **Done** (`chakaso retrieve`).
9. A retrieval interface for public web access that does not require a commercial search
   API. — **Partial:** a bounded, opt-in fetcher exists (ADR-0012) and never runs on a
   default path; deciding *which* URLs to fetch (query planning for web discovery) is open.
10. Answer generation with source-identifier citations. — **Not done:** the plumbing
    records citations, and no turn generates an answer to cite into. A trained adapter that
    *could* generate exists behind `load_model`; wiring it into the runtime is P32, which
    ADR-0033 deliberately leaves after the model's own measurement.
11. Reassessment entry point. — **Done** (application-level reassessment over a stored
    answer; explicit identifier trigger, ADR-0029).
12. Unit and integration tests for all of the above. — **Done** (see `tests/`, 1,405 tests
    collected at the current head; 1,389 pass in the dependency-free matrix, 16 skip there,
    and 1,603 pass with `[train,nn]` installed).
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
   Re-measured later with the same honesty: E2605 records per-step curves, wall clock per stage and the
   checkpoint checksum, and peak memory is *still* not measured, because the process-memory APIs return
   nothing from this shell.
7. Compare against a trivial baseline (a uniform or unigram predictor). — **Done**: held-out loss beats
   both baselines.

Scaling happens after reproducibility, not before. The first experiment demonstrated the learning loop and
produced the project's first genuinely measured number; it is not a step toward a useful chatbot, and the
measured overfitting is itself the evidence that justifies the next, controlled experiment (early
stopping / capacity, multiple seeds). That experiment ran: E2602 reproduced the result across five
pre-declared seeds with a proper test split (1.80 ± 0.18 nats, below both baselines on every seed).

## The model path (P30, P31) and the first Transformer (E2605)

[ADR-0033](decisions/ADR-0033-model-centred-product-contract.md) replaced "system before model" with
"the model is the product", and this is the sprint that acted on it rather than restated it.

1. **P30 — the data layer, dependency-free on purpose.** `chakaso.training.manifest` (per-document SHA-256
   over normalized text, licence, source, typed drift refusal), `split_text_units` (whole documents,
   deduplicated by content *before* selection), `chakaso.training.subword` (byte-level BPE, named
   pre-tokenizer, deterministic tie-breaking, self-describing checksum-verified record) and
   `chakaso.training.packing` (fixed-length next-token examples, dropped tail reported). All of their tests
   run in the dependency-free CI matrix, which is why admitting them cost no new tier.
2. **P31 — the model, admitted by measurement.** [ADR-0034](decisions/ADR-0034-optional-pytorch-model-extra.md)
   re-opened ADR-0030 on its own revisit conditions and admitted `torch` as a separate `[nn]` extra,
   recording that the speed argument did not survive measurement (1.16× on a same-shape matmul) and that
   the gradient guarantee became sampled rather than exhaustive.
   `chakaso.training.transformer`, `nn_training`, `nn_checkpoint`
   ([ADR-0035](decisions/ADR-0035-transformer-checkpoint-format.md)) and `model_metrics` followed.
3. **The interface first, not last.** `chakaso.models.load_model(checkpoint)` returns a `LanguageModel`
   from a checkpoint with no terminal, no config file and no application import — the concrete adapter is
   reached through the registry's single selection point, so ADR-0002 held while the model changed under it.
4. **Measured, with the comparison unit chosen before the run.** E2605
   ([`../experiments/E2605/record.md`](../experiments/E2605/record.md)) trained a 643,840-parameter subword
   Transformer from random initialization on the pinned 71-document corpus and reported **1.8657 bits per
   character** on held-out documents against **3.1343** for E2601's character architecture, 3.3224 unigram
   and 4.8010 uniform. Early stopping selected step 1380 of 2400; the un-stopped model was 0.1434 nats worse.
   The committed generations loop.

What follows is fixed by the record, in this order: reproduce E2605 across pre-declared seeds so the gap
carries a spread; obtain an external corpus with a recorded licence, because one author's prose bounds every
number above; only then ask a capacity question. **P29** (calibrating the retrieval sufficiency policy) and
**P32** (runtime consumption of that decision) stay deferred — ADR-0033's ordering puts a measured model
ahead of both, and nothing in E2605 changes their content.

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
not appetite, set the next sprint — a retrieval evidence-sufficiency decision (P28, tracked as K-010),
now delivered.

## Retrieval evidence sufficiency (P28)

E2604 measured the response to E2603's negatives. [ADR-0032](decisions/ADR-0032-retrieval-evidence-sufficiency.md)
adds a deterministic, additive sufficiency decision at the retrieval boundary (`chakaso.retrieval.sufficiency`):
the retriever still returns top-k, and a policy reports whether the pack is adequate, judged by whether a
retrieved chunk shares a rare, non-function-word query term — the signal that actually separated the
failures (junk matched `the`/`to`/`library`; real evidence matched `closed`/`licensed`/`borrower`).

1. Failure analysis, case by case, before choosing a rule. — **Done** (a pure document-frequency threshold
   was tried and *failed* — `to`/`on`/`and` are df=1 at 13 docs — so the design required a content word).
2. Additive policy + result type + boundary, retrieval contract unchanged. — **Done** (`EvidenceSufficiency`,
   `DiscriminatingTermSufficiencyPolicy`), with a test that it never mutates the pack.
3. Measured as a separate evaluation dimension. — **Done**: E2604 sufficiency accuracy **1.0** on the
   workload, **0** false abstentions (recall preserved), **0** missed (all abstentions caught), retrieval's
   own metrics unchanged.
4. Honest limits. — **Done**: a narrow corpus-specific band (~0.23–0.31), a broken single-doc case, no
   runtime effect yet, not a calibrated confidence.

The sprint that follows is set by E2604's caveat, not by appetite: **P29** calibrates the policy under a
strict development/held-out protocol and stress-tests it on a harder, extended self-authored workload —
or records honestly that this corpus is too small to calibrate. Runtime consumption of the decision is P32.

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

## Handoff phases

The foundation release (nodes A–R, 2026-10-22) ends with a verified, reproducible, integrated model that is not
yet useful. These seven phases are the prioritized continuation, written so that the next session can start at
the top without re-deriving what is true. Nothing in this list is a status claim: each phase says what would
*close* it, and the evidence that closes it does not exist yet.

| Phase | Work | Exit criteria | Do not |
| --- | --- | --- | --- |
| **0 — Publish what exists** | The release itself: contract, model card, data cards, three-tier CI, vertical slice, `chakaso model` lifecycle | Level 1 and Level 2 assessed as met in [MODEL_CARD.md](../MODEL_CARD.md) with digests; every doc describing the runtime matches the tree; CI green on all three tiers | Do not describe Level 3 as near, or let the loss number imply a capability |
| **1 — Make the curve turn** | Raise the token budget until validation loss stops improving inside it: more steps, cheaper evaluation (a fixed held-out subset with the full pass kept for the final report), or a larger batch at the same token count | An arm whose report's `completion` is `converged` rather than `budget_ceiling`, with the turning point in its curve and the wall-clock cost of getting there measured | Do not reduce `eval_every` to afford steps and then compare the resulting loss to a curve sampled differently without saying so |
| **2 — Put a spread on the reference model** | **E2609**: arm A across five pre-declared seeds, split fixed, using E2606's method unchanged | mean ± sample SD bits per character for the decoder family, the spread stated beside the headline number, and a documented choice of which checkpoint is the reference | Do not choose seeds by their scores; do not call the sample SD a confidence interval |
| **3 — Answer the architecture question** | A capacity-matched arm (equal parameters by shrinking `d_model`/depth) and a context-matched arm, so H2 is tested without the 3.3× confound E2608 had to accept | a comparison where the two arms share parameters *or* context and the difference is the architecture, with the token budget matched exactly as in E2608 | Do not report E2608's 0.2178 bits as an architecture result; the record already says why it is not one |
| **4 — Attack degeneration with measurement, not cosmetics** | Two independent lines: decoding (top-p and repeat-penalty sweeps reported *against the generation suite*, not against fluency impressions) and data (author/genre/register diversity beyond 19th-century narrative prose) | repetition ratios and periodic-run counts reported per configuration with raw samples retained, and an explicit position on whether looping is fixed, unfixed, or unfixed-but-quiet | Do not use a repetition penalty to make output *look* varied while the diagnostics stay hidden; the suite's interpretation limit exists for exactly this |
| **5 — Put a trained model in front of a workload** | Wire a checkpoint into the runtime behind an explicit opt-in (the default stays the double), then measure grounded answering on the E2603/E2604 workload with the model producing the text | a measured grounded-answer result with a real model in the loop, citation and grounding numbers reported with their failures, and K-001 narrowed on evidence rather than on a demo | Do not make a trained model the default configuration before Level 3 is met; do not let a harness oracle score what the model did not say |
| **6 — Domain selection and adaptation** | The original P11/P12: choose a defensible niche on explicit criteria, build its dataset and benchmark, and adapt | a domain chosen with written reasons, a pinned corpus under a stated licence, and a benchmark that the *domain* work can be measured against | Do not start domain work while Phase 1 has not produced a converged model; adapting a truncated base confounds two unknowns at once |

Phases 1–4 can interleave, and 2 is deliberately cheap next to 1: the seed spread uses E2606's method with no
new machinery. Phase 5 is the one that changes what the project *is*, and it waits for a model worth putting in
front of a workload, which is the discipline ADR-0033 exists to keep.

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
