# ADR-0024: Hybrid retrieval fuses lexical and dense results by reciprocal rank fusion

- Status: Accepted
- Date: 2026-10-10
- Relates to: [ADR-0002](ADR-0002-language-model-boundary.md), [ADR-0022](ADR-0022-query-planner-boundary.md), [ADR-0023](ADR-0023-dense-retrieval-boundary.md)

## Context

Two retrievers now sit behind the `Retriever` protocol — lexical BM25 and dense similarity —
and they fail differently. A hybrid retriever that consults both and returns one ranking is
the standard way to get their complementary coverage, and it is the last thing the query
planner's `hybrid` mode was waiting for (ADR-0022 named it, ADR-0023 backed `dense`, `hybrid`
was still refused).

The hard part is that the two components do not share a score scale: a BM25 score and a cosine
similarity are not comparable numbers, so any fusion that adds or interpolates them would be
comparing quantities that mean different things — an unfounded arithmetic the project's honesty
rule forbids. The fusion also has to stay deterministic, transparent, independently testable,
and, above all, auditable: a research reader must be able to see which component put a chunk
where and why the final order came out as it did.

## Problem

How should lexical and dense rankings be combined into a single ordering without inventing a
shared score scale, without a learned model, while staying deterministic and explainable, and
while preserving enough per-component provenance that a fused result can be evaluated honestly?

## Options considered

- **Normalize the two scores and interpolate.** Rejected: it forces a comparability between a
  BM25 score and a similarity that does not exist; the blend would depend on score ranges that
  are artefacts of each method, not meaningful magnitudes.
- **A learned / model-based fusion.** Rejected: no model exists, and a heavy dependency breaks
  the CPU-first, offline, dependency-light posture (ADR-0001).
- **Simple interleaving of the two lists.** Rejected: it is explainable but crude — it cannot
  reward a chunk both components ranked highly, and it loses the per-component reasoning.
- **Reciprocal Rank Fusion over the two ranked lists.** Chosen: it needs only each component's
  *ranks*, not a comparable score; it is deterministic and published; a chunk high in both lists
  naturally rises.

## Decision

`chakaso.retrieval.hybrid.HybridRetriever` implements the existing `Retriever` protocol. It
holds a lexical `Retriever` and a dense `Retriever` plus a `HybridConfig`, and fuses their
results with reciprocal rank fusion. The formula is stated, not hidden:

```
fused(c) = Σ over components that returned c of   weight_component / (rrf_k + rank_component(c))
```

where `rank_component(c)` is the chunk's 1-based rank within that component's own list, and a
component that did not return `c` contributes nothing. The default `rrf_k` is 60 (the published
value); `lexical_weight` and `dense_weight` (both default 1.0, non-negative, not both zero) let
one component be favored; `candidate_k` controls how many results each component retrieves before
fusion (defaulting to the caller's `top_k`). Determinism follows from summing components in a
fixed order and breaking final ties by chunk identifier.

Each fused result is a `RetrievalResult` tagged `strategy=hybrid`, whose `score` is the RRF value
(a ranking artefact, explicitly not a similarity or probability) and which carries a new
`HybridProvenance` on the result: the lexical rank and score (or `None`), the dense rank and score
(or `None`), and the fused score. `matched_terms` comes from the lexical component when it
returned the chunk and is empty otherwise — the fusion never invents term overlaps. A result that
neither component returned is never produced.

`HybridConfig` is a frozen, validated dataclass passed to the retriever — the typed tuning the
hybrid path needs — and it is deliberately *not* a second TOML-loading system (ADR-0006). It is
the library-level input object; surfacing it in the file-backed configuration is the natural next
step for whoever consumes it first (the retrieval-aware runtime or the benchmark), not a field
added here before something reads it.

Because a hybrid retriever now exists, `DeterministicQueryPlanner` accepts `HYBRID`: every
retrieval mode the planner can name is now backed, and only `NONE` remains a non-retrieval mode.
This updates the consequence in ADR-0023 that `hybrid` was the last named-but-refused mode.

## Reasoning

- **Ranks are the only thing two unlike scorers reliably agree on.** RRF sidesteps score
  incomparability by using rank position alone, which is exactly the property this fusion needs
  to be defensible rather than a hidden weighted sum of apples and oranges.
- **Provenance is what makes a fused number research-usable.** A single blended score would be
  the opacity RRF is meant to avoid; recording each component's rank and score lets evaluation
  attribute a hit and lets a later experiment compare fusion against a single strategy honestly.
- **Composition over orchestration.** The hybrid retriever holds two `Retriever`s; it does not
  own generation, grounding or persistence, keeping with the modular runtime the architecture
  requires (ADR-0022).

## Trade-offs

- Rank-only fusion discards score magnitude: a chunk ranked first by both wins, but a very strong
  dense match and a marginal one are indistinguishable to RRF. That is the accepted cost of not
  inventing a shared scale.
- With the fixture embedding (non-semantic, ADR-0023), hybrid over it mostly re-ranks token
  overlap; any hybrid-vs-lexical benchmark result here is a synthetic development measurement and
  may not be reported as dense/hybrid understanding. The boundary is real; the quality claim is
  withheld until a genuine embedding exists.
- `rrf_k` and the weights are published/neutral defaults, not tuned values; nothing has measured
  them against a real corpus yet (K-001).

## Consequences

- The planner's full mode vocabulary (`lexical`, `dense`, `hybrid`) is now backed, so a retrieval
  decision can name any of them and something can execute it once the runtime wires them in.
- `RetrievalResult` carries an optional `provenance` field, the seam a benchmark will use to
  compare strategies and to answer whether fusion introduced duplicates or changed ranks.
- `HybridConfig` is the object to lift into typed file configuration when the runtime or benchmark
  first needs to select fusion parameters per run.
- Hybrid retrieval remains in-memory and offline; nothing here persists vectors or reaches the
  network.
