# ADR-0003: The retrieval layer owns evidence identifiers; the model only references them

- Status: Accepted
- Date: 2026-10-07

## Context

Chakaso's central claim is that an answer can be traced to the evidence that
supports it. Language models are fluent generators of plausible URLs, titles and
quotations, and a system that asks a model to emit citations is a system that
will eventually emit citations to sources that do not exist or do not say what
was claimed.

The failure is not only embarrassing. It is unmeasurable: if source identity is
produced by the generator, there is no ground truth against which citation
precision or citation recall could ever be computed.

## Problem

Who is authoritative for source identity and citation content?

## Options considered

- **Model-generated citations** — the model writes URLs and titles inline in its
  answer. Simple, and works acceptably often.
- **Model cites opaque identifiers assigned by retrieval** — the retriever
  assigns a stable identifier to every source and chunk; generated text may only
  reference those identifiers; the application resolves identifiers to verified
  metadata when rendering.
- **Post-hoc attribution** — the model answers freely, then a separate step
  searches retrieved evidence for sentences resembling the answer's claims and
  attaches sources afterwards.
- **No citations** — return the evidence alongside the answer and let the user
  do the matching.

## Decision

Source and chunk identity belongs to the retrieval and evidence layer, not to
the model.

- The retrieval layer assigns stable identifiers (`source_id`, `chunk_id`) when
  content is retrieved, and persists the metadata that gives them meaning:
  canonical URL, title, domain, retrieval timestamp, publication timestamp where
  known, and a content hash.
- These records are immutable once written. Re-retrieving the same URL produces a
  distinguishable record rather than mutating history.
- Generated text may only reference identifiers that were supplied in the
  evidence passed to that generation call.
- Any identifier in generated output that was not supplied is treated as an
  error condition, not as a source. It must never be resolved to a URL, and it
  must be recorded, because fabricated identifiers are a metric (unsupported
  claim rate), not a cosmetic defect.
- URLs shown to a user are resolved from stored metadata by the application.

## Reasoning

- It makes citation correctness checkable. "Did the cited chunk support the
  claim?" is answerable when the citation is an identifier into a known set. It
  is not answerable when the citation is prose the model invented.
- It removes an entire class of fabricated output by construction rather than by
  prompt instruction. Prompt instructions reduce a failure; they do not bound
  it.
- Stable identifiers are a prerequisite for every planned evaluation metric in
  the grounded, citation and correction layers.
- Content hashes make the evidence reproducible. A stored answer can be
  re-examined against the exact bytes that supported it, even after the live page
  has changed.

## Trade-offs

- The model must be taught a citation convention (reference by identifier), and
  small models will get this wrong. Handling malformed references is real work
  that a free-text citation scheme avoids.
- Answers become harder to read in raw model output, since identifiers are
  resolved only at the presentation layer.
- Storing source text and metadata alongside hashes costs disk and forces an
  explicit artifact-retention policy.

## Rejected alternatives

**Model-generated citations** was rejected as the primary mechanism. It produces
the exact failure Chakaso exists to avoid, and it makes the project's headline
metrics uncomputable.

**Post-hoc attribution** was rejected because similarity between a claim and an
evidence sentence is not support. Attributing an answer after the fact tends to
launder unsupported claims by attaching loosely related text to them, which is
worse than no citation because it implies verification that did not happen. A
claim-evidence check is still needed, but it must validate references the
generator actually made rather than manufacture them.

**No citations** was rejected because it abandons the project's differentiator
and cannot distinguish a grounded answer from a lucky one.

## Consequences

- Retrieval produces records, not strings. The application passes structured
  evidence to the model, not a blob of scraped text.
- The application carries an identifier-resolution step between generation and
  presentation, and that step is a validation gate rather than a lookup.
- Citation validation is a first-class component with its own tests and metrics,
  because it is where the project's claims are either supported or not.
- Retrieved content is untrusted input. Identifiers and metadata may be trusted
  as far as the retriever validates them; the *text* of a retrieved document must
  never be treated as instructions.
- Prompt templates must be versioned, because the citation convention is part of
  the prompt contract and changes to it change measured behaviour.
