# ADR-0047: a pinned corpus declares the digest of the text it ingests, and where its raw bytes live is left to the owner

- Status: Accepted in part (2026-11-02). Part 1 is implemented; part 2 is a recommendation that needs the owner's decision to act on.
- Date: 2026-11-02
- Related: [ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md), [ADR-0044](ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md),
  [artifact policy](../internal/engineering/artifact-policy.md), K-031.

## Context

A dataset catalogue pins each Project Gutenberg source by the SHA-256 of the bytes as downloaded, and `corpus prepare` refuses a
mismatch. On 2026-11-02 one of 430 pinned sources (`science-pg3489`) no longer matched: the upstream file had been re-edited
after the 2026-10-30 retrieval. The refusal was correct, but it could not say what kind of change it was. Whether the change lay
inside the text training reads, or only in the header, licence preamble or trailer that ingestion discards, was learned only by
re-ingesting under a scratch catalogue and checking that the corpus fingerprint came out as pinned. That outcome was luck of where
the change fell, not a property of the process. The original bytes were not recoverable: the URL is not an archive, and the one
archive mirror tried returned nothing from the sandbox.

Two separate gaps are in that story:

1. **No declared identity for the ingested text**, so drift cannot be classified before anything trains.
2. **No durable copy of the raw bytes**, so a drifted source can be classified but not restored.

## Decision

### 1. A catalogue source may declare `ingested_sha256` (implemented)

The digest is taken over what ingestion hands to the corpus: each document's path and SHA-256, in reading order, under the
catalogue's chunking bound (`IngestedSource.documents_digest`). It is not a digest of the extracted body, because the documents
are what training sees and what the corpus fingerprint is made of; two bodies that differ only in the blank lines between
paragraphs chunk to the same documents and are the same training text.

* The field is optional. A catalogue without it parses, validates and serialises exactly as before, so no existing catalogue
  changes meaning. `chakaso corpus declare` now writes it for new catalogues.
* When it is declared, `DatasetCatalog.ingest` refuses text whose digest differs, whatever the raw digest says.
* `verify_raw` still refuses every raw mismatch. It now also says, per source, whether the ingested text is identical, differs, is
  no longer readable as a Project Gutenberg text, or cannot be classified because no digest was declared. `classify_drift`
  returns the same classification as data. **Nothing is accepted by the classifier**: the guard is not weakened, and a
  header-only change still needs an explicit, recorded act (a new catalogue entry for the new raw digest) before a run proceeds.
* `configs/corpora/gutenberg-multiregister-v1.toml` now declares the digest for all 430 sources (430 added lines, none removed).
  They were computed from the bytes on disk on 2026-11-02 and checked against the pinned corpus: every one equals the digest of
  the same source's documents in the snapshot whose corpus fingerprint is `312571f9fa8351a8`, the fingerprint the E2612–E2616
  records cite. For `science-pg3489` this is the only evidence available for the *original* bytes' ingested text, and it is
  indirect: it rests on the fingerprint reproducing exactly.
  `classify_drift` on the current bytes now reports exactly `{"science-pg3489": "ingested-text-identical"}`.
* `gutenberg-english-prose-v2` does not declare it: its raw files are not on this host, and a digest computed from anything else
  would not be evidence.

### 2. Raw bytes (recommendation; needs the owner)

The raw directory of the multiregister corpus is 156 MB on disk and about 59.5 MB gzip-compressed; the 1 MiB tracked-file bound
and the artifact policy keep it out of Git, and that stays. A URL is still not an archive. The recommendation is **one archive of
the verified raw directory, kept in a place the owner controls, cited from the catalogue's data card by its SHA-256**, so that a
future drift can be restored as well as classified. The archive needs no new format: the raw directory already is one file per slug
named by the catalogue, and `verify_raw` already verifies it.

This is a recommendation and not an action because the destination is a hosting decision: a release attachment on this repository,
object storage, or a registry. Each publishes 59.5 MB of third-party text, and Project Gutenberg's own terms say that users
outside the United States must check local law. It is the same open question the artifact policy already holds for released
weights, and is recorded there rather than guessed at.

## Consequences

* The next time a pinned source drifts, the refusal says whether training would have changed.
* The catalogue grows by one 64-character line per source; the corpus fingerprint, the snapshot and every recorded number are
  unchanged.
* Until part 2 is decided, a source whose raw bytes change and whose ingested text is identical can be re-pinned (as E2615 did,
  disclosed in its record), and one whose ingested text differs cannot be re-pinned at all without a new corpus version.
