# ADR-0031: The grounded-answer evaluation workload is self-authored, provenance-bearing data

- Status: Accepted
- Date: 2026-10-19
- Relates to: [ADR-0004](ADR-0004-apache-2.0-license.md), [ADR-0007](ADR-0007-content-derived-identifiers.md), [ADR-0011](ADR-0011-typed-source-references.md), [ADR-0013](ADR-0013-benchmark-identity-and-versioning.md), [ADR-0015](ADR-0015-grounding-boundary.md), [ADR-0027](ADR-0027-semantic-evaluation-contract.md)

## Context

Everything measured so far measures machinery, not answers. The retrieval and correction
development benchmarks score ranking and a decision rule over hand-shaped fixtures; the semantic
instrument scores a harness whose only judge is a caller-supplied fixture; E2601/E2602 measure a
toy character model's next-token loss. None of them says whether Chakaso's actual answer pipeline
produces a useful, evidence-supported, correctly cited answer to a real question — the question
H1–H7 and the strong form of K-001 exist to ask. `docs/evaluation.md` lists "Grounded-answer
benchmark" as **Planned** and gives its required shape: a question, expected answer *or expected
behaviour*, acceptable evidence, unacceptable evidence (the field that catches citation laundering),
and the expected disposition (answer / qualify / abstain).

The roadmap's release discipline warns against growing a full benchmark before there is model
behaviour worth measuring. That tension is real and this record does not dissolve it: the only
model that exists is a development double that emits fixed, non-responsive prose. A workload built
now can be scored against the double, against a caller-supplied answer, and against a future real
model — but scoring the double measures the harness and honestly reveals that the double abstains
from every grounded case. It is not, and must never be reported as, evidence of answer capability.

The harder constraint is legal, not technical. A grounded-answer workload needs *source evidence* a
person can read and audit, and that the repository is permitted to store and redistribute. Publicly
accessible text is not automatically licensed for redistribution, and the project forbids inventing
corpora or committing copyrighted material whose terms do not allow it.

## Problem

What is a grounded-answer evaluation workload made of, where does its source evidence come from,
under what terms may it be stored and redistributed, and where does it live in the code so that it
inherits the reproducibility discipline the project already enforces?

## Options considered

- **Scrape or download an existing public QA/datasets corpus.** Reuses realistic language, but the
  licence of "publicly accessible" content is frequently unclear or non-redistributable; committing
  it risks publishing material the project has no right to ship, and it silently imports
  contamination and provenance questions the project cannot audit. The fetcher is off every default
  path (ADR-0012) and CI is offline, so this is also unbuildable in the required environment.
- **Reuse the existing retrieval fixtures as the answer workload.** Cheap and already versioned, but
  they carry no expected answer, no prohibited claims, no provenance fields and no split; treating a
  retrieval case as an answer case would under-specify exactly the judgements that make an
  answer-evaluation honest.
- **A small, self-authored workload whose passages the project writes itself, with provenance and
  licence recorded per source, validated by a strict schema.** Mirrors the training corpus precedent
  (`chakaso-toy-char-v1`: self-authored, Apache-2.0, provenance stated in code) and keeps every
  source human-inspectable with no third-party terms to track down.

## Decision

Adopt the third option. Concretely:

- A **grounded case** is a distinct schema from a retrieval `BenchmarkCase`: a stable slug
  `CaseId`, a `BenchmarkCategory` and `ExpectedBehavior` reused from `chakaso.benchmark.cases`, a
  question, the expected answer disposition, **required facts**, **prohibited claims**, the **gold**
  and **forbidden** evidence (named by source-passage label and resolved to content-derived
  `ChunkId`/`SourceId` the same way retrieval gold evidence is), an evaluation rationale, and a
  **split** label. Prohibited claims and forbidden evidence are first-class, not optional — without
  them an over-citing system scores like a correct one.
- Every source passage in the workload carries a **provenance record**: authorship method (here,
  authored for Chakaso), a **licence**, and any redistribution restriction. A case whose evidence
  has no documented provenance is **invalid at load time**; the schema refuses missing provenance
  rather than defaulting it.
- The initial workload is **self-authored** and covered by the project's Apache-2.0 licence (ADR-0004),
  exactly like the training corpus. No third-party copyrighted text is fetched or committed. It is
  deliberately **small enough for complete human inspection**, and it is labelled for what it is.
- The workload lives inside `chakaso.benchmark` as a grounded-answer sub-module, reusing `CaseId`,
  `DatasetId`, `content_version` and the benchmark errors (ADR-0013). Its identifier/version rule is
  the same rule the retrieval benchmark already obeys; a new top-level package would fork that
  discipline for no benefit.
- The workload separates **development** from **held-out** cases with a check that no case, and no
  source passage, appears on both sides — split integrity is asserted, not assumed. Configuration or
  prompt selection is done on development; any reported held-out number is computed on cases that
  played no part in selection.
- The evaluation runner composes existing boundaries only: retrieval through `RetrievalService`,
  answers through an injected answer provider, decomposition through `chakaso.claims`, and scoring
  through `chakaso.evaluation.evaluate_answer` over the `chakaso.citation` / `chakaso.grounding`
  primitives. It distinguishes a retrieval failure from a generation failure from an evaluator
  failure, records per-case inspectable outcomes, and never merges the dimensions into one quality
  score.
- Results over this workload — with the double, with a replayed reference answer, or with a future
  model — are labelled by what produced them. The workload and its harness are **Implemented**;
  whether the pipeline produces *good* answers on real language remains **unmeasured** (K-001).

## Reasoning

- Self-authorship is the only source the project can prove it is allowed to ship while offline. It
  makes provenance a fact in the file rather than a claim about someone else's licence.
- Reusing the benchmark identity layer means a grounded result carries the same auditable
  dataset-id / dataset-version / content-fingerprint / per-case-version provenance ADR-0013 demands,
  instead of inventing a parallel and weaker scheme.
- Making prohibited claims, forbidden evidence, provenance and split *mandatory schema fields* is
  what turns "a benchmark" into a *defensible* one: each is the guard against a specific way a
  number can lie.

## Trade-offs

- Self-authored passages are less natural than real text; they are shaped to exhibit a judgement, so
  the workload is easier than a real corpus. Accepted — its job is a trustworthy seam for measuring
  the pipeline, not a leaderboard.
- Reusing `BenchmarkCategory` couples the grounded schema's vocabulary to the retrieval one's.
  Accepted — the categories (direct, multi-source, missing-evidence, conflicting, ambiguous,
  unsupported-premise) are the same failure modes; divergent labels would only invite drift.
- A harness scored against a non-answering double measures the harness. Accepted, provided every
  report says so; the honesty rule (do not describe planned work as done) is the load-bearing one.

## Rejected alternatives

Scraping/downloading public corpora was rejected because its licence and provenance cannot be audited
and it cannot run in an offline CPU-only environment; reusing the retrieval fixtures as-is was
rejected because they omit the answer-level judgements (expected disposition, required facts,
prohibited claims, provenance) that make an answer evaluation meaningful.

## Consequences

- `chakaso.benchmark` grows a grounded-answer sub-module: case + provenance schema, a strict loader
  that refuses missing identifiers/provenance, malformed evidence references, duplicate cases,
  invalid labels and split leakage, a versioned self-authored workload with a pinned content
  fingerprint, a runner, separate retrieval/answer metrics, and deterministic text/JSON reports that
  label themselves development instruments.
- Introducing a real semantic judge or a pretrained model remains gated behind its own feasibility
  decision (ADR-0030's pattern); this workload is the thing such a judge would be evaluated against,
  not a substitute for it.
- `docs/evaluation.md` "Grounded-answer benchmark" moves from Planned to Implemented (a development
  instrument over a self-authored, held-out workload; no answer-capability claim), and K-001 stays
  open because the workload is not a real-world corpus.
