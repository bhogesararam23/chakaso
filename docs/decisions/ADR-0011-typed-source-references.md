# ADR-0011: Source identity is a typed reference, not a URL

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0007](ADR-0007-content-derived-identifiers.md)

## Context

[ADR-0007](ADR-0007-content-derived-identifiers.md) derives a source identifier from
the canonical URL and a content hash. That answers "which web page" cleanly, but the
project needs evidence from sources that are not web pages: a document already on
disk, a passage of text supplied by a caller, a generated fixture, and later an
imported dataset. None of them has a URL.

The retrieval plan was blocked on exactly this, recorded in
[`../research/open-questions.md`](../research/open-questions.md): reading a local
file requires deciding what identifies a source with no URL. Two things resisted a
quick answer.

`canonicalize_url` accepts only `http` and `https`. Its scheme check is not only a
normalization rule — it is also the de facto list of schemes a future fetcher is
allowed to retrieve. Routing a local or pasted reference through it, or widening its
scheme list to admit one, would silently widen what the fetcher may reach. A security
boundary and an identity rule were living in one function.

Meanwhile `SourceRecord` stores `canonical_url`, and every consumer of provenance
reads it as a web URL.

## Problem

How should a source be identified when it is not a web page, without faking a URL,
without changing the identifiers of sources that are, and without turning URL
normalization into a fetch-permission list?

## Options considered

- **`file://` and pasted text through `canonicalize_url`.** Reuses one normalizer.
  Rejected: it makes the identity function the place where fetch permission is
  decided, which is the coupling the open question warned about, and `canonicalize_url`
  is tuned for web semantics (tracking parameters, default ports) that mean nothing
  for a local path.
- **Make every non-web source a fake `http` URL** (for example `http://local/<path>`).
  Rejected outright: provenance would claim every source is fetchable from a web
  server, which is false, and the citation machinery would render a fabricated URL.
- **A parallel record type for local sources.** Honest, but it splits the evidence
  model in two and forces every downstream consumer — chunking, packs, citation
  resolution — to handle two shapes for no benefit.
- **A typed source reference that owns identity and is produced differently per
  kind.** One abstraction, `SourceReference`, with a kind, a canonical string and an
  optional locator. Web references consult `canonicalize_url`; file and text
  references do not. The identifier derivation of ADR-0007 is unchanged; only its
  input is generalized from "canonical URL" to "canonical reference".

## Decision

Introduce `chakaso.evidence.identity` with a `SourceKind` (`web`, `file`, `text`) and
an immutable `SourceReference` carrying `kind`, `canonical` and `locator`.

- A **web** reference's `canonical` is `canonicalize_url(url)`, so it is byte-identical
  to today and every existing identifier is preserved.
- A **file** reference's `canonical` is a `file://` URI built from a resolved absolute
  path via the standard library, and it never passes through `canonicalize_url`.
- A **text** reference's `canonical` is a `text:` reference, optionally labelled, for
  content supplied directly with no persistent location.

`derive_source_id` now takes the canonical reference instead of a canonical URL. The
formula — `"src_" + sha256(canonical + 0x1F + content_hash)[:16]` — and
[ADR-0007](ADR-0007-content-derived-identifiers.md) are unchanged. `SourceRecord`
stores the `kind` and the `canonical` reference (renaming its `canonical_url` field to
`canonical_reference`) and gains `create_file` and `create_text` alongside `create`.

The three canonical forms are mutually unreachable as strings: a canonical web
reference always starts with `http://` or `https://`, a file reference with `file://`,
and a text reference with `text:`. One cannot be confused for another, so folding all
three into one hash input cannot collide across kinds.

## Reasoning

- **Identity and fetchability become different concepts.** A source *is* something;
  whether the fetcher may retrieve it is a separate policy. Only web references touch
  `canonicalize_url`, so widening source identity no longer widens the fetch
  permission. The list of fetchable schemes stays where it belongs and stays narrow.
- **Backwards compatibility is preserved by construction.** Because a web reference's
  canonical string is exactly the canonical URL, ADR-0007's identifiers do not move.
  No re-minting, no migration.
- **No fabricated URLs.** A local file is named by a genuine `file` URI; pasted text
  by a `text` reference. Neither pretends to be a web page, so the citation layer
  never renders a URL that was never fetched.
- **One record type, one evidence model.** Chunking, packs and citation resolution
  keep operating on `SourceRecord` and `EvidenceChunk` unchanged; the generalization is
  confined to how a record is identified and created.

## Trade-offs

- **File identity is machine-local.** A `file://` URI names a path on one filesystem;
  the same document copied to another machine produces a different reference. That is
  correct — it is a different copy — but it means content-based deduplication across
  machines needs a separate content reference, deferred with the dataset pipeline.
- **Relative paths resolve against the working directory**, so a reference built from a
  relative path can differ between runs unless the working directory is fixed.
  Absolute paths give stable identity; the ingestion API takes explicit paths and its
  tests pin the resolved form.
- **`text` is not a registered URI scheme.** It is a local marker for a reference with
  no location, chosen to be distinct from every real scheme. If a standard for opaque
  content references is ever adopted, it supersedes this rather than reusing `text:`.
- **A stored `kind` can disagree with a hand-built canonical string.** Prevented by a
  consistency check in both `SourceReference` and `SourceRecord` that requires the
  canonical string to begin with its kind's marker, so an inconsistent record fails at
  construction rather than in production.

## Rejected alternatives

Rejected options are the fake-URL and `canonicalize_url`-widening paths above, plus a
parallel record type; each was declined for putting fetch permission into identity,
for inventing provenance, or for splitting the evidence model.

## Consequences

- `chakaso.evidence.identity` owns what a source *is*; `chakaso.core.identifiers` still
  owns how an identifier is derived from it (ADR-0007 is untouched).
- `canonicalize_url` is now explicitly a URL normalizer and web-scheme policy, used
  only for web references. It is no longer on the path that identifies a local source.
- The `SourceRecord.canonical_url` field is renamed to `canonical_reference`; tests and
  the record-shape documentation are updated in the same change.
- Local document ingestion and lexical retrieval can now be built, because the
  blocking question in `../research/open-questions.md` is answered.
- A `dataset` or `generated` kind can be added to `SourceKind` with its own reference
  builder without touching the derivation formula or existing kinds.
