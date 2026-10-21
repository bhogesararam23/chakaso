# ADR-0035: A Transformer checkpoint is one checksummed file that carries its tokenizer inline

- Status: Accepted
- Date: 2026-10-21
- Relates to: [ADR-0002](ADR-0002-language-model-boundary.md),
  [ADR-0004](ADR-0004-apache-2.0-license.md), [ADR-0020](ADR-0020-experiment-artifacts.md),
  [ADR-0030](ADR-0030-minimal-optional-training-dependency.md),
  [ADR-0033](ADR-0033-model-centred-product-contract.md),
  [ADR-0034](ADR-0034-optional-pytorch-model-extra.md)
- Follows up on: ADR-0034's Consequences, which deferred exactly this decision to a record of its own.

## Context

[ADR-0034](ADR-0034-optional-pytorch-model-extra.md) admitted `torch` for the model path and required that
a checkpoint be written and read with `weights_only=True`, carry content checksums, and refuse an
incompatible tokenizer, architecture or configuration rather than half-loading. It left one question open:
whether a Transformer artifact stores its tokenizer's merge table **inline** or references it by
fingerprint.

The project already decided this once, for the character checkpoint. P25 changed schema/2 to embed the
vocabulary's characters, "so a checkpoint can decode on its own — a fingerprint alone identifies a
vocabulary but cannot turn identifiers back into text, which an inference path requires". That reasoning
applies with more force to a subword tokenizer, whose merge table *is* the decoding function.

Meanwhile the artifact format for a Transformer has properties the JSON schema/2 format cannot supply: it
stores float32 tensors, it needs a real serialization format, and `torch`'s native container is a pickle
format unless loading is restricted.

## Problem

Define the artifact format for a trained Transformer: what it contains, how it is made safe to read, and
how it refuses to be loaded as something it is not.

## Options considered

- **Option 1 — numpy arrays in the existing JSON schema/2.** Human-readable and already tested, but the
  schema is bound to the character MLP's five named arrays and its `TrainingConfig`; a Transformer's state
  is a different set of tensors. JSON float round-tripping is exact but slow and bloated at float32 scale.
- **Option 2 — `torch.save` of a state dict, metadata in a sidecar JSON.** Conventional and compact. Costs:
  two files that can be separated or paired wrongly, and no integrity claim over the tensor bytes.
- **Option 3 — `torch.save` of one payload: tensors plus a canonical metadata block plus a checksum over
  both, read with `weights_only=True`.**
- **Option 4 — Safetensors.** The purpose-built safe format, but it brings a new dependency of its own for
  a project that has just admitted one framework, and it does not carry arbitrary metadata in the form the
  record needs without another library.

## Decision

**Adopt Option 3, with the tokenizer stored inline.**

1. **One file, one integrity claim.** `chakaso-transformer-checkpoint/1` holds: a canonical JSON metadata
   block (schema, architecture, tokenizer fingerprint + size + merge table, corpus fingerprint, step, run
   metadata, environment, sorted tensor names), the tensors, an optional optimizer block, and a SHA-256
   over the metadata bytes concatenated with every tensor's name, shape, dtype and raw bytes. An altered
   tensor and an edited metadata block produce the same typed failure: `CorruptedCheckpointError`.
2. **The loader is restricted, always.** `torch.load(..., map_location="cpu", weights_only=True)`, and no
   other load call exists in the package. A file that tries to reconstruct a Python object fails at that
   boundary and is reported as unreadable rather than executed. This is a security rule, not a style rule:
   a checkpoint from a directory the caller does not control must be inert.
3. **The tokenizer travels with the weights.** The merge table is inline and, on load, is re-hashed against
   the recorded `tokenizer_fingerprint`; its size must equal the architecture's `vocab_size`. A checkpoint
   paired with a different tokenizer is refused — the failure that would otherwise surface as a model
   producing fluent nonsense.
4. **The architecture is rebuilt from the declared configuration, and the tensors are checked against it.**
   A missing tensor, an unexpected one, or a shape disagreement is `IncompatibleCheckpointError` naming the
   offending names. A configuration field this code does not implement is refused rather than ignored, so
   an artifact cannot quietly mean something else.
5. **Two artifact kinds, distinguished in the metadata: `usable` and `resumable`.** A usable artifact holds
   the validation-selected parameters and no optimizer state. A resumable one, written by the training loop
   at each evaluation, holds the *current* parameters plus the moments plus the step. `resume_state_from`
   refuses a usable artifact, because resuming from selected weights with fresh moments is a different
   schedule that looks like the same one.
6. **Overwrite is opt-in.** `save_artifact` refuses an existing path unless the caller passes
   `allow_overwrite`; the interruption writer, which rewrites its own file by design, does.
7. **No artifact bytes in Git.** They live under `data/`, which is ignored, and a record cites the
   fingerprint. [ADR-0004](ADR-0004-apache-2.0-license.md)'s reservation of weights and datasets for their
   own release policy is untouched by this format decision.

## Reasoning

- **A fingerprint is an identity, not a decoder.** Rule 3 follows the P25 precedent for the same reason: an
  inference path must be able to turn identifiers back into text without locating a second file that is
  assumed to match. A merge table is a few hundred pairs; the cost is negligible against the tensors.
- **One file keeps the integrity claim honest.** A sidecar can be swapped independently of the tensors it
  describes. Since the checksum already covers metadata and bytes together, splitting them would weaken the
  only guarantee the format makes.
- **`weights_only=True` makes the restricted loader the default rather than a discipline.** Requiring it is
  cheap once the floor is `torch>=2.6`, which is why ADR-0034 set that floor.

## Trade-offs

- **Larger than a tensor-only file.** The metadata repeats the merge table and stores a JSON string; at this
  scale it is kilobytes against megabytes.
- **`torch.save` container, not a language-neutral one.** Safetensors or a numpy-in-JSON route would be
  readable without `torch`; the project has already accepted `torch` as the model path's dependency, and
  adding a second serialization library to avoid it buys a portability nobody has asked for. If weights are
  ever *released*, the export format becomes a distribution decision with its own record.
- **The format is versioned, not migrated.** Schema 1 refuses anything else; there is no older artifact to
  migrate from, so a migration path would be speculative code.
- **Tied embeddings are stored twice** — torch's own `state_dict` convention — so `parameter_count` is
  reported from the declared architecture rather than by summing stored tensors, and the record says which
  it is reading.

## Consequences

- `chakaso.training.nn_checkpoint` owns the format, and `chakaso.training.nn_training` writes resumable
  artifacts through it. Both import `torch`, and both are on ADR-0034's allowlist.
- An inference adapter over this format becomes possible, which is what
  [ADR-0033](ADR-0033-model-centred-product-contract.md)'s Python-API rule requires; it must not fall back
  to the development double when the extra or the file is missing.
- An experiment record cites the artifact's checksum, not its bytes.
- Creates no compatibility shim for the JSON schema/2 character checkpoint: the two artifact kinds are
  distinct formats with distinct loaders, and a trained character model stays what it is — a measured toy.
