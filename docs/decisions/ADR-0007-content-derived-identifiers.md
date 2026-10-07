# ADR-0007: Evidence identifiers are derived from content, not assigned at random

- Status: Accepted
- Date: 2026-10-07
- Relates to: [ADR-0003](ADR-0003-evidence-identifier-ownership.md)

## Context

[ADR-0003](ADR-0003-evidence-identifier-ownership.md) establishes that the
retrieval layer owns source and chunk identity, that records are immutable, and
that generated text may only reference identifiers it was given. It does not say
how an identifier is produced.

Two things pull in opposite directions:

- **Immutability and history.** An answer recorded last week cites evidence by
  identifier. If the page has since changed and the identifier stayed the same, the
  old answer now points at content it never saw, and its provenance is a lie.
- **Idempotence and deduplication.** The same page is fetched repeatedly during
  development. If every fetch mints a new identifier, an evidence pack can be
  filled with five copies of one document, and a retrieval evaluation measures
  duplicates instead of retrieval.

## Problem

Should identifiers be assigned at random (or by a counter) when content is
retrieved, or derived deterministically from the content they name?

## Options considered

- **Random identifier** (UUID or similar) per retrieval. Simple, collision-free,
  and each retrieval is trivially distinguishable.
- **Monotonic counter** per store. Simple, but identity depends on insertion order,
  so it is not reproducible across runs or machines, and it makes a store a
  required piece of global state.
- **Derived from content**: hash the canonical URL together with the content hash
  for a source, and the source identifier together with position and text for a
  chunk.
- **Derived from the canonical URL alone.** Stable and deduplicating, but it cannot
  distinguish two versions of a page, which is the case ADR-0003 exists to protect.

## Decision

Identifiers are derived deterministically from the content they name.

```text
source_id = "src_" + sha256(canonical_url + 0x1F + content_hash)[:16]
chunk_id  = "chk_" + sha256(source_id + 0x1F + position + 0x1F + text)[:16]
```

The parts are joined with a unit separator so that concatenation cannot be
ambiguous (`"ab" + "c"` must not hash the same as `"a" + "bc"`).

Consequences of this formula, stated because they are the point:

- The same URL with the same content yields the same `source_id`, so repeated
  fetches deduplicate.
- The same URL with **different** content yields a different `source_id`, so an
  older record is never silently replaced and an older answer still points at the
  bytes it actually used. This is the property ADR-0003 requires; the mechanism is
  content-addressing rather than a retrieval counter.
- Chunk identity changes if the text at that position changes, so re-chunking a
  document produces new chunks rather than silently redefining old ones.

Identifier format is fixed: a four-character type prefix followed by sixteen
hexadecimal characters (64 bits of digest). The prefix makes an identifier
self-describing in a log line, in stored metadata, and in model output, where it
also gives a validator something unambiguous to recognise.

## Reasoning

- **Reproducibility.** A run can be replayed and produce the same identifiers. This
  matters for evaluation: a benchmark that reports which evidence was retrieved is
  only comparable across runs if the identifiers mean the same thing.
- **Deduplication comes for free.** URL normalization plus a content-derived
  identifier is the whole deduplication mechanism. No separate dedup index, and no
  ordering requirement on the store.
- **Immutability is enforced by construction rather than by discipline.** There is
  no way to "update" a record, because a changed record has a different name. A
  bug that would overwrite history instead creates a second record, which is
  visible.
- **A store becomes optional.** Identity does not depend on insertion order or on
  a database having assigned a key, so retrieval can be tested with no store at
  all.

## Trade-offs

- **Sixty-four bits invites a birthday collision.** For the scale this project
  operates at — thousands to low millions of chunks — the probability is
  negligible, but it is not zero, and it is a truncation rather than a full
  digest. The alternative, a full 64-character digest, is unreadable in logs and
  impractical for a model to reproduce. If a collision ever occurs, the correct
  response is to detect it by comparing stored records rather than to widen the
  identifier silently.
- **Identity depends on the canonicalization rule.** If URL normalization changes,
  every identifier derived from a URL changes with it, and old records no longer
  match new ones. Normalization is therefore part of the identifier contract and
  cannot be changed casually.
- **Identity depends on the content hash, so it depends on the extraction.** Two
  runs that extract slightly different text from the same page produce different
  sources. That is correct — the evidence really is different — but it means
  "did we already fetch this page?" is not answerable from the URL alone.
- **Longer identifiers than a counter.** `src_3f2a91c4d5e6b708` costs more tokens
  in a prompt than `3`. This is the price of provenance, and it is why the prefix
  is four characters rather than something descriptive.

## Rejected alternatives

**Random identifiers** were rejected because they make evaluation non-reproducible
and require a store to answer "have I seen this before". The immutability benefit
is real but is obtained more cheaply by content addressing.

**A monotonic counter** was rejected for the same reasons plus one more: it makes
identity depend on a mutable global, which is exactly the kind of hidden state that
makes a research pipeline hard to re-run.

**Canonical URL alone** was rejected because it cannot represent two versions of a
page. A system that overwrites its evidence when a page changes cannot evaluate
correction, because the earlier evidence it should have corrected against no
longer exists.

## Consequences

- `chakaso.core.identifiers` owns derivation and validation. Nothing else
  constructs an identifier string by hand.
- URL normalization is now part of the identifier contract. A change to it is a
  change to every future identifier, and must be recorded.
- Identifier validation is cheap and total: a string either matches the fixed
  pattern or it is not an identifier. Citation validation can therefore reject a
  fabricated reference before any lookup happens.
- Chunking configuration becomes identity-affecting. Two chunkers over the same
  document produce disjoint chunk sets, which is a feature for comparison and a
  reason not to mix chunk sets from different configurations in one evidence pack.
- The store must handle two records for one URL. That is the intended behaviour,
  not an error case.
