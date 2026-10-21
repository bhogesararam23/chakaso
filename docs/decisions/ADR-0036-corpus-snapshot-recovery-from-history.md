# ADR-0036: A recorded corpus is recovered from the revision that pinned it, not re-pinned from the tree

- Status: Accepted
- Date: 2026-10-21
- Relates to: [ADR-0004](ADR-0004-apache-2.0-license.md),
  [ADR-0005](ADR-0005-src-layout.md), [ADR-0013](ADR-0013-benchmark-identity-and-versioning.md),
  [ADR-0020](ADR-0020-experiment-artifacts.md),
  [ADR-0030](ADR-0030-minimal-optional-training-dependency.md),
  [ADR-0033](ADR-0033-model-centred-product-contract.md),
  [ADR-0035](ADR-0035-transformer-checkpoint-format.md)
- Follows up on: E2605's recorded blocker — the pinned corpus of that experiment cannot be re-prepared from
  this repository's own working tree, because 13 of its 71 documents were edited after the pin.

## Context

[ADR-0020](ADR-0020-experiment-artifacts.md) keeps generated artifacts out of Git, and the corpus manifest
in `chakaso.training.manifest` pins a corpus by the SHA-256 of every document's normalized text. Those two
rules meet badly here. E2605 trained on this repository's own tracked markdown, pinned it as content
digests, and the documentation work that followed edited that same text. `verify_corpus` now refuses the
working tree, which is correct: a model measured on one corpus must not be re-measured against a different
one. The consequence is that the project's best result cannot be reproduced at all — the training code is
fine, the checkpoint is fine, and the input bytes are unreachable from the files that are on disk.

`write_snapshot` and `read_snapshot` already existed, and nothing called them. That is the whole of the
problem: the bytes that *were* reachable at the moment of measurement stopped being reachable, and no record
of them survived outside the tree that moved.

## Problem

Decide how a recorded experiment's exact corpus bytes are recovered and stored, so that reproduction is
possible from a clean checkout without weakening the drift guard that makes the recorded number mean
something.

## Options considered

- **Re-pin the manifest whenever the tree moves.** One line of code, and it destroys the result: the held-out
  documents would no longer be the documents the model was scored on. Rejected outright; this is the failure
  mode ADR-0013's content fingerprint exists to prevent.
- **Lift or narrow the drift check for historical experiments.** A recorded corpus would then be "whatever
  matches most of the digests", which is not a corpus. Rejected.
- **Commit the corpus snapshot into Git.** 780,223 characters of text that is already in Git in another form:
  a duplicate that no one can edit, that ADR-0020 excludes, and that grows with every experiment. Rejected.
- **Publish the corpus as a release asset or object-store blob with a checksum.** A network dependency in the
  reproduction path, infrastructure the project does not have, and — decisively — a checksum that points at
  bytes nobody has is not reproducible. Rejected; the same objection applies to any "reconstruct it from the
  published digest" idea.
- **Recover the bytes from the version-control history and store them as a snapshot.** The corpus is tracked
  text, so every byte it contributed still exists at the revision that recorded it. Read that revision, put
  it through the existing verification, and write the artifact.

## Decision

A recorded corpus is recovered from a named revision, through `chakaso.training.snapshot`:

1. The **manifest is the trust anchor**; a revision is only a source of bytes. The caller names a manifest and
   a revision; nothing about the revision can change which corpus is expected.
2. The revision's candidate files are materialized into a staging tree that nothing else reads, and
   `verify_corpus` runs over it — the same function, with the same three-way check (changed, missing,
   newly-selected), as a fresh preparation. A revision one documentation edit away from the pin is refused
   exactly as the stale working tree is.
3. The output is a snapshot (`chakaso-corpus-snapshot/2`): every document body with its own digest, the
   manifest record embedded, a checksum over the whole file, and an **acquisition block** naming the revision
   and the commit object it resolved to. A snapshot restored from history is therefore distinguishable from
   one taken from the current tree, and neither can be passed off as the other.
4. Snapshots live in `data/`, are git-ignored, and are written atomically so a half-written corpus artifact
   cannot be mistaken for a whole one. After acquisition, training reads the snapshot and needs no history
   and no network.
5. History is read through a `RevisionReader` seam. `GitRevisionReader` is the only implementation, is
   read-only, takes fixed argv with no shell, refuses option-shaped revision names, and refuses an archive
   path that would write outside the staging tree.

## Reasoning

The drift guard and reproducibility are the same property seen from two times. Checked now, it means "the
tree is not your corpus"; checked later, it means "the corpus was these bytes, and here is where they came
from". Making the second case run the first case's code is what stops this from becoming a parallel,
weaker verification path — which is the usual way reproducibility machinery rots.

Recording the resolved commit rather than only the short name matters: `HEAD~1` is a position, a commit
object name is an identity. And keeping the acquisition visible in the artifact answers the question a later
reader actually has — *is this the historical corpus or the documentation of today?* — without asking them to
trust a command history.

The alternative that would have been simpler — commit `data/experiments/E2605/*-snapshot.json` — was
rejected for the reason ADR-0020 already gives: generated bytes that duplicate tracked bytes. What is
committed instead is the recipe (the manifest, the revision in the record) and the verification (the tests),
which is the part that has to be reviewable.

## Trade-offs

- Reproduction now depends on the **clone being complete**. A `--depth 1` checkout does not contain the
  revision and says so, naming the remedy. This is the first place the project has needed more than one
  commit of history.
- Git history becomes part of the reproducibility contract. That raises the cost of the rule the repository
  already follows — never rewrite pushed history — because a force-push would delete the input of a
  measured result.
- Restoring runs `git archive` in a subprocess, so corpus acquisition needs Git where pure training does
  not. The training path itself keeps no such requirement: it reads the snapshot.
- The snapshot file is large relative to a manifest (the corpus text, as JSON). It is a research artifact
  under `data/`, not a distributable, and that is what ADR-0020 is for.

## Consequences

- `chakaso.training.snapshot.restore_corpus_snapshot` is the documented acquisition API, and `write_snapshot`
  / `read_snapshot` gain a file checksum, per-document digests and an acquisition block. ADR-0035's overwrite
  guard extends to corpus artifacts: a re-run may not replace a recorded run's snapshot unasked.
- Every experiment that trains on tracked text names the revision its corpus came from in its record, the
  same way it names its commit.
- Experiment records stay immutable. A reproduction is a new record with its own measurements, linked to the
  original, not an edit to it.
- The reproduction commands for E2605 are documented in `docs/training.md`, and E2605's own record gains a
  dated addendum pointing at the reproduction rather than changing any number in it.
