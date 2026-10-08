# Active task

## Current unit: P6 — a first retrieval benchmark and evaluation harness

**Goal.** Turn the retrieval substrate from "built and tested" into "measured." Build the
smallest honest evaluation the project can stand behind: a hand-built query set with
relevance judgements over a fixed local corpus, run through the lexical retriever, scored
with the metric functions that already exist in `chakaso.evaluation`. Then report what the
lexical baseline actually does — recall@k, precision@k, MRR, duplicate rate — as *measured*
numbers with a named, versioned dataset behind them.

**Why this is next.** The retrieval pipeline exists end to end, and the metric functions
exist, but there is no dataset to run them on, so no number about retrieval is currently
true or even computable. The open question the substrate leaves is exactly the one an
evaluation answers: where does lexical retrieval succeed and where does it fail. That
question must be measured before it is answered with a dense retriever, or the dense
retriever is judged against nothing. This follows the project's standing rule — evaluation
before scaling — and it is what would begin to close K-001 (nothing measured against a real
workload).

**Why the smallest thing.** A hand-built set of queries with declared relevant chunks needs
no corpus download, no model and no network. It is reproducible, reviewable, and versioned
in the repository, which is the only kind of benchmark this project will report a number
from.

## What the previous unit completed (P3/P4 — the retrieval substrate)

All of the following is implemented, tested, documented and green on Python 3.11–3.13.
None of it is a claim about retrieval *quality* — there is no measured result yet.

- [x] Reconciliation: stale ADR references and the `domain`/host description corrected
- [x] Generalized source identity — `SourceReference`/`SourceKind`, web/file/text, with web
      identifiers unchanged (ADR-0011)
- [x] Deterministic content normalization (representation only)
- [x] Local document ingestion from explicit paths, with filesystem-safety rules and typed
      errors; never crawls
- [x] In-memory corpus with idempotent add and a provenance guard
- [x] Deterministic lexical (BM25) retriever behind a `Retriever` protocol, with inspectable
      explanations and no fabricated results
- [x] Retrieval orchestration into a validated `EvidencePack` with scores and provenance
      preserved
- [x] End-to-end offline pipeline tests, including a turn through `ConversationManager.send`
- [x] `chakaso retrieve` — explicit-source retrieval from the command line
- [x] Bounded, opt-in web fetcher under a `FetchPolicy`, with a literal-address destination
      screen and typed acquisition errors (ADR-0012)
- [x] Web ingestion: a narrow HTML reader, `ingest_acquired`, and an in-memory
      `AcquisitionCache`/`CachingFetcher`; fetched text is data, not instruction, and never
      assigns identity
- [x] Evaluation metric functions (`chakaso.evaluation`), tested on hand-built fixtures

## What this unit is not

- Not a model, a tokenizer or training. The model boundary and its development double are
  unchanged.
- Not a dense retriever or a vector index. The benchmark's job is to say what the lexical
  baseline is worth before anything replaces it.
- Not a query planner. Retrieval is still invoked explicitly; deciding *whether* and
  *which* to retrieve is a separate unit (P4).
- Not a live-web crawl. No default path touches the network, and the benchmark is built on a
  fixed, committed corpus.
- Not a place for a number without a dataset. Until the query set and judgements exist in
  the repository, nothing in `chakaso.evaluation` is run and no metric is reported.

## Ordering after this unit

1. A first retrieval benchmark and harness (P6) — this unit.
2. Query planning: when to retrieve, and which sources (P4).
3. Dense embeddings and a vector index, measured against the lexical baseline (P3/P4).
4. Claim-level support checking, then reassessment and the correction loop (P5).
5. A local CPU model adapter (P2/P7), then the tokenizer, dataset pipeline and a tiny
   Transformer (P7–P8).

## Known gaps that bound this unit

- K-008: the fetcher's destination screen blocks address literals but not a hostname that
  resolves to a private address; there is no robots or per-host rate limiting. Fetching
  stays opt-in and off every default path. This does not block a local benchmark.
- K-001: nothing has been measured against a real workload. This unit is the first step to
  closing it; it closes it only when a benchmark exists and a number is reported from it.

## What previous units knowingly left undone

Recorded here rather than discovered later:

- An answer's model identity is reported in the reply but is not stored in conversation
  state, so it is not recoverable from a transcript alone. That belongs to the
  `AnswerRecord` the correction unit introduces (K-007).
- Nothing persists how often a model invents an evidence reference across turns; the metric
  function to count it (`unresolved_reference_count`) now exists, but no run records it
  (K-006).
- Nothing resolves references to earlier turns ("that", "the second one"); follow-ups carry
  the previous turns and nothing more specific.
