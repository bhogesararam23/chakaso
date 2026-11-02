# ADR-0046: a checkpoint records the tokenizer's frequency floor as an optional field, and says "unrecorded" when it is not

- Status: Accepted (2026-11-02; implemented in the same change)
- Date: 2026-11-01
- Related: [ADR-0035](ADR-0035-transformer-checkpoint-format.md) (the format), [ADR-0044](ADR-0044-a-vocabulary-a-record-cites-is-committed-beside-it.md)
  (committed vocabularies), K-030 item 6.

This record was written first (status Proposed, 2026-11-01) so the format decision was made on purpose and not slipped into a
patch. It was accepted on 2026-11-02 when the project brief assigned the task (T14) to be resolved, and implemented test-first in
the commit that changes its status. The *Context* below describes the state before the change.

## Context

`SubwordVocabulary` carries `min_frequency`, the floor the merge search stopped at when the vocabulary was **fitted**.
Applying the vocabulary (encoding, decoding, token counts) uses the merge table only; the floor is provenance.

A checkpoint's metadata document stores the merge table, its fingerprint and size, but not the floor. `load_artifact`
rebuilds the vocabulary with `SubwordVocabulary(merges=…)`, so `min_frequency` takes the dataclass default, **2**,
whatever the fit used. Reproduced on 2026-11-01 with a throwaway artifact: a vocabulary fitted at 3 loads back reporting 2,
its fingerprint is unchanged, and its file-level `checksum()` (which does include the floor) no longer equals the original
file's.

What this does and does not break today:

* Encoding and generation are unaffected. The fingerprint, which is what checkpoint identity and every record cite, covers the
  merge table only and is unchanged.
* A loaded artifact states a fact that may be false (`min_frequency == 2`), and a vocabulary file re-serialised from a
  loaded artifact has a different checksum from the file it came from. Neither affects a recorded number; both are the kind of
  silent wrongness the project's rules prohibit.
* The plan-versus-stored-tokenizer check (`_check_vocabulary_matches_plan`) compares the floor, but only for tokenizer **files**,
  never for checkpoints, so it is not misled.

## The constraint that makes this a decision

[ADR-0035](ADR-0035-transformer-checkpoint-format.md) fixes the checksum over the metadata document, and the released
`chakaso-base@0.1.0` verifies against it. Two facts about the current code make a compatible change possible:

1. **Verification hashes the stored metadata string verbatim** (`nn_checkpoint.load_artifact` updates the digest with
   `metadata.encode("utf-8")`, not with a re-serialisation). An old file therefore verifies whatever the reader's idea of the
   document is.
2. **The loader reads named keys** out of the parsed metadata and ignores others, so an old reader opens a file that has an extra
   key.

The format is "versioned, not migrated": a different `schema_version` is refused. A bump would refuse every existing artifact
in the new reader or every new artifact in the old one, which is exactly the compatibility break this issue must avoid.

## Options considered

- **A. Optional top-level key `tokenizer_min_frequency`, written only when known, no schema bump.** New artifacts record the
  floor. Old artifacts are untouched and still verify. A loaded old artifact reports the floor as *unrecorded* rather than 2.
  An artifact that is loaded and re-saved without a known floor emits no key, so its metadata document is byte-identical to the
  original and its checksum is preserved.
- **B. Put the floor in `run_metadata`.** Free-form, needs no format change. But `run_metadata` is caller-supplied, so absence
  cannot be distinguished from "the caller forgot", and the loader could not rely on it.
- **C. Bump the schema.** Refuses old artifacts or new ones; rejected for the reason above.
- **D. Recompute the floor from the merge table.** The floor is a stopping rule, not a function of the merges (the last merge's
  count is not stored), so it is not recoverable. Rejected.
- **E. Do nothing and document the field as "always 2 on load".** Cheapest; keeps a field that states something false.
  Acceptable only as the interim, which is the present state.

## Proposed decision

Option A, in four parts:

1. `ModelArtifact` gains `tokenizer_min_frequency: int | None = None`.
2. `metadata_document()` includes `"tokenizer_min_frequency"` **only when it is not `None`**, so every artifact that does not
   know its floor serialises exactly as before.
3. `load_artifact` reads the key with `recorded.get(…)`. Present → used, and the rebuilt vocabulary is constructed with it.
   Absent → the artifact reports `None`; the vocabulary's own `min_frequency` field stays at its default but the artifact's
   accessor, not the vocabulary's field, is what a caller is told to trust.
4. Writers that know the floor (the experiment pipeline, which holds the fitted vocabulary) pass it.

Tests the implementation must carry, written first: (a) a vocabulary fitted at 3 round-trips as 3 through a new artifact;
(b) an artifact written by the **current** code, kept as a fixture, still loads, verifies, and reports the floor as unrecorded;
(c) re-saving that loaded artifact yields a byte-identical metadata document and the same checksum; (d) a reader that ignores
the key (simulated by deleting it from the parsed document before reading named keys) still opens a new artifact.

## What this does not decide

* It does not recover the floor for already-released artifacts. For `chakaso-base@0.1.0` the value is a fact of the training
  plan that produced it (`tokenizer_min_frequency` in its protocol), and a model card may cite it from there.
* It does not make the floor part of tokenizer **identity**. The fingerprint stays over the merge table, because two tokenizers
  with the same merges tokenise identically whatever floor produced them.
* It does not touch `SubwordVocabulary.checksum()`, which already includes the floor.

## Consequences (as implemented)

* `ModelArtifact.tokenizer_min_frequency: int | None`, validated (a positive integer or `None`); `metadata_document()` emits the key only
  when it is not `None`; `load_artifact` reads it with `.get`, refuses a stored value that is not a positive integer, and rebuilds
  the vocabulary with it when present.
* `nn_training` writes the floor of the vocabulary the run was handed, in both the usable and the resumable artifact. That is as
  trustworthy as the vocabulary object: one built from a tokenizer file or a fit carries its true floor (the experiment pipeline
  already checks it against the plan); one rebuilt from an old checkpoint carries the dataclass default, which is why a caller must
  not feed a loaded old artifact's vocabulary to a new run as if its floor were known.
* `TrainedModel.describe()` reports `tokenizer_min_frequency`, `None` for an unrecorded artifact.
* Tests (five more than the four listed, all in `tests/test_training_nn_checkpoint.py` and `tests/test_training_nn_training.py`): a floor
  of 3 round-trips as 3; an artifact in the pre-field shape loads, verifies and reports the floor unrecorded; re-saving it reproduces
  its metadata document and checksum byte for byte; the new key is purely additive; invalid values are refused on construction and
  on load; both training artifact kinds carry the run's floor. The pre-field shape was built by a test helper that was checked once
  against a file written by the unmodified code (identical metadata string and checksum); no binary fixture is committed, because
  checkpoint bytes stay out of Git.
* No artifact on disk or in a release changes. `chakaso-base@0.1.0` still reports the floor as unrecorded; its training plan records 3.
