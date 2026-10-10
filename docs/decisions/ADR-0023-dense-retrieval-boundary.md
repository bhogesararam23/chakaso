# ADR-0023: Dense retrieval is an embedding boundary with an exact index; the only embedding is a non-semantic double

- Status: Accepted
- Date: 2026-10-10
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0002](ADR-0002-language-model-boundary.md), [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0019](ADR-0019-semantic-grounding-boundary.md), [ADR-0022](ADR-0022-query-planner-boundary.md)

## Context

The query planner can name a `dense` mode (ADR-0022) but refused to emit it, because no
retriever backed it. Dense retrieval means ranking by similarity in a vector space, which needs
an embedding model. The project's constraints are hard: ADR-0001 forbids a required commercial
inference or search API, the CPU-first posture forbids assuming a GPU or a large model download,
and the test environment must stay offline and dependency-light. There is no trained Chakaso model
and no embedding model.

The temptation is to either pull in a heavyweight embedding library or an API, or to skip dense
retrieval entirely until a model exists. Both are wrong: the first breaks the dependency and
offline rules; the second leaves the planner's `dense` mode permanently unbackable and the dense
boundary untested when a real model does arrive.

## Problem

How should dense retrieval be built so that the architecture is real, exercised and replaceable
now — an actual retriever behind the `Retriever` boundary and an exact index — without a genuine
embedding model, without a heavy dependency, and without ever letting a stand-in vector be
described as semantic?

## Options considered

- **Bundle a real open-source embedding model.** Rejected: a large dependency, a download, and a
  GPU/heavy-CPU assumption that CI and the local-first rule cannot carry (ADR-0001).
- **Call a commercial embedding API.** Rejected: an external provider the runtime must not require
  (ADR-0001), and it would put a hosted dependency inside retrieval.
- **Define the protocol only, implement nothing.** Rejected: an uninspected abstraction with no
  consumer is exactly the empty scaffolding the project forbids; a boundary earns its place by
  having an implementation and a test behind it.
- **An `EmbeddingModel` boundary, a deterministic non-semantic double, and an exact CPU index.**
  Chosen.

## Decision

`chakaso.retrieval.embeddings` defines the `EmbeddingModel` protocol — `metadata` and a batch
`embed(texts) -> tuple[EmbeddingVector, ...]` (positional output, one fixed-length finite vector
per input) — with `EmbeddingMetadata` carrying `model_id`, `dimension`, `normalized`,
`development_double` and, decisively, `is_semantic`. The one implementation today is
`FixtureEmbeddingModel`, which hashes a text's tokens into a fixed-length bag-of-words vector and
L2-normalizes it. It declares `is_semantic=False` and `development_double=True`: two texts sharing
no surface token get orthogonal vectors, so its cosine similarity is token overlap, not meaning,
and nothing downstream may call it semantic.

`chakaso.retrieval.dense` adds `DenseIndex` (an exact in-memory store of chunk vectors, built once,
validating uniform dimension and rejecting a duplicated chunk) and `DenseRetriever`, a second
implementation of the existing `Retriever` protocol that ranks by similarity and returns the same
`RetrievalResult` shape the lexical retriever returns. The index is exact — it scans every vector —
because that is correct and dependency-free at this scale; an approximate structure (FAISS or
similar) is deferred until a measured corpus is large enough to need one, so it is a later decision
rather than a present guess.

`RetrievalResult` gains a required `strategy` field (`lexical` / `dense` / `hybrid`) so every
result names the mechanism that produced it — the provenance evaluation needs (a dense hit must not
be silently comparable to a lexical one without being labelled). A dense result sets `matched_terms`
to empty rather than inventing term overlaps it did not compute. The `is_semantic=False` flag and the
empty `matched_terms` are the two places this design refuses to overstate what a fixture embedding
can do.

Because a dense retriever now exists, the planner's `dense` mode becomes selectable:
`DeterministicQueryPlanner` accepts `LEXICAL` and `DENSE` and still refuses `HYBRID`, which has no
fusion layer until the hybrid retriever is built. This updates ADR-0022's consequence that `dense`
was unbacked — the boundary and the honesty rule are unchanged; only the set of backed modes grew.

## Reasoning

- **Boundary-first makes a real embedding a drop-in, not a rewrite.** The same reasoning as ADR-0002
  and ADR-0021: define the seam while the only implementation is a double, so the genuine model
  replaces the implementation behind the protocol and the index, retriever and planner are untouched.
- **`is_semantic` makes the honesty structural, not just prose.** Declaring non-semantics in the
  metadata a retriever reports means a caller cannot accidentally treat a fixture vector as an
  understanding; this mirrors the fixture-only `SemanticJudge` of ADR-0019, where the boundary exists
  and the implementation is honestly labelled.
- **Exact scan over approximate index is the smaller commitment.** Approximate indexes earn their
  cost at scale and bring a dependency; a research project's first dense retriever should be
  provably correct and reproducible, which an exact scan is.

## Trade-offs

- The fixture embedding is essentially lexical, so a dense-over-fixture benchmark cannot show dense
  beating lexical on anything but exact-token overlap; results using it must be labelled a
  development fixture and may not be reported as evidence that dense retrieval helps. That is a
  property of the double, not the boundary, and is stated wherever such an experiment is run.
- Hashed bag-of-words has collisions (different tokens can share a bucket) and fixed dimension — fine
  for a double, wrong for real work; `is_semantic=False` is the guard against the confusion.
- An exact index is O(n·dimension) per query — acceptable now, and the boundary is what lets an
  approximate index replace it later without touching callers.

## Consequences

- The planner's `dense` mode is now backed; `hybrid` is the only remaining named-but-refused mode,
  and the hybrid fusion layer (the next unit) is what backs it.
- Every `RetrievalResult` now carries its producing strategy, so retrieval evaluation can attribute
  a hit and hybrid fusion can preserve per-component provenance without inferring origins.
- A real embedding model lands behind `EmbeddingModel` unchanged; when it sets `is_semantic=True`,
  dense retrieval's claims can rise to match — and not before.
- The retrieval corpus and its indexes remain in-memory (nothing here persists vectors to disk);
  durable storage so far covers answer history only (ADR-0021).
