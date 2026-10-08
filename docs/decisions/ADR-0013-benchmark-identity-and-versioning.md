# ADR-0013: Benchmark cases have a stable identity and a content-derived version

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0007](ADR-0007-content-derived-identifiers.md), [ADR-0011](ADR-0011-typed-source-references.md)

## Context

Retrieval and evaluation exist: `chakaso.evaluation` has metric functions, and the
retrieval pipeline can rank a corpus for a query. But a metric is meaningless without a
defined thing it was computed over. The next milestone is a benchmark — versioned cases, a
runner, and reports — and a benchmark number is only as trustworthy as the ability to say
exactly which set of judgements produced it.

`docs/evaluation.md` already fixes the reporting standard: a result is reported with the
dataset or task version, the component version, and the configuration. That standard cannot
be met unless a benchmark case has an identity that survives edits and a version that
changes when its meaning changes.

The evidence layer solved a related problem by deriving identifiers from content
([ADR-0007](ADR-0007-content-derived-identifiers.md)). A benchmark case looks superficially
similar — it too is content that should be addressable — but the constraints differ.

## Problem

How should a benchmark case and a benchmark dataset be identified and versioned, so that a
reported result can be reproduced against exactly the judgements it claims to measure?

## Options considered

- **One identifier that is a content hash, like evidence.** Then editing a gold-answer
  list silently turns the case into a different case with a different name, and a
  hand-curated case can no longer be referenced by a stable label in prose, a report, or a
  regression test.
- **A content hash plus a random UUID.** Two identifiers, one of them meaningless in a
  text file, and a UUID defeats diff review of a curated dataset.
- **A human-written identifier and a separate content-derived version.** The identifier is
  the stable address; the version is a fingerprint of the judgements, so the address
  survives editing while any change to what is judged is detectable.

## Decision

Split the two roles explicitly, at both levels:

- A **`CaseId`** and a **`DatasetId`** are stable, lowercase slug strings, written by the
  author and unique within their scope. They are how a case or dataset is addressed.
- A **case version** is a deterministic fingerprint (`content_version`) over the case's
  *judgements only* — category, query, gold and forbidden evidence, required and forbidden
  claims, expected behaviour, citation requirement — and explicitly not over its
  identifier. Renaming a case keeps its version; changing what it expects bumps it.
- A **dataset** carries an author-set `version` string (a semantic label) *and* a
  `content_fingerprint` computed over the sorted `(case_id, case_version)` pairs, so an
  edit that the author forgot to declare is still visible, and a reordering of the file is
  not.
- Collections inside a case are canonicalized on construction (deduplicated, sorted), so
  two authorings that mean the same thing compare equal and fingerprint the same.

The version and fingerprint are truncations of SHA-256 over a canonical JSON rendering,
mirroring the evidence identifier choice (ADR-0007) for the same reason: readable in a
report, collision-free at benchmark scale.

## Reasoning

- **A name that survives editing is what makes a case addressable.** Curated cases are few
  and read by people; `direct-deadline` in a report must keep pointing at the same case
  across dataset versions, or a regression claim is untraceable.
- **A version that is content, not a counter, cannot drift from what it labels.** An author
  bumping a number can forget; a fingerprint cannot. The dataset's `content_fingerprint` is
  the backstop for exactly that forgetting.
- **The split is cheap and it is testable.** Identity stability, order-independence, and
  judgement-sensitivity are each asserted by a test rather than assumed.

## Trade-offs

- **Two version notions per dataset** (semantic string, content fingerprint) is more than a
  bare minimum. It earns its place: the string is for humans deciding what a release means,
  the fingerprint is for catching the gap between what a human said and what changed.
- **A renamed case is not a content change but is a history change.** If a case is renamed
  between versions, the fingerprint of the dataset is unchanged while the address moves.
  This is acceptable because renames are rare, deliberate, and visible in the diff; it is
  noted rather than papered over.
- **Fingerprints depend on the canonical JSON rendering.** Adding a new judgement field to
  the schema changes fingerprints for otherwise-identical cases. That is correct — a new
  required judgement is a version change — but the rule means the schema's field set is part
  of the identity contract, like URL normalization is for evidence.

## Rejected alternatives

A pure content identifier was rejected because it destroys the stable address that makes a
case citable across versions; a UUID was rejected as meaningless in a file a human reviews
and diffs. Both were declined for the same reason: a curated benchmark is read and argued
over by people, and its vocabulary must be theirs.

## Consequences

- `chakaso.benchmark.identity` owns `CaseId`, `DatasetId`, `canonical_json` and
  `content_version`; nothing else constructs these by hand.
- A case's version excludes its identifier; a dataset's fingerprint excludes case order.
  Tests pin both.
- The benchmark loader and runner must record, in every result, the dataset identity,
  dataset version, and dataset content fingerprint, so a report names the exact judgements
  behind its numbers.
- Changing the case schema's judgement fields is a benchmark-identity change and belongs
  with a version decision, not a silent edit.
