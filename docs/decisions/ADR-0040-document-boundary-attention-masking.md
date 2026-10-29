# ADR-0040: document-boundary attention masking is an opt-in training option, off by default

**Status:** Accepted
**Date:** 2026-10-29
**Related:** [ADR-0034](ADR-0034-optional-pytorch-model-extra.md) (the decoder stays inside the `[nn]` extra),
[ADR-0035](ADR-0035-transformer-checkpoint-format.md) (checkpoint and resume identity), E2605–E2611 (the
measured protocols that keep full causal attention).

## Context

Packing concatenates one side of a document split into fixed-width blocks and terminates every document with
a separator (`chakaso.training.packing`). The decoder's attention is causal over the whole block, so a token
in one document can attend to every earlier token in the block, including tokens from unrelated documents
that happen to precede it.

That is a deliberate design in many language-model training pipelines, and every measured result in this
repository used it. It has two costs:

1. **A training/inference mismatch.** At inference a prompt usually begins at the start of a document, not
   after an arbitrary prefix of unrelated text. Training on concatenated prefixes teaches the model to use
   context that a real caller will not supply.
2. **Capacity spent on spurious dependencies.** A small model trained on many short documents spends some of
   its attention budget learning across boundaries that carry no task signal. As the corpus grows and a
   packed block holds more documents, the fraction of cross-boundary positions rises.

OLMo 2/3 and SmolLM3 document intra-document attention masking as part of their training recipes; it is not a
frontier-only technique. But adopting it silently would make every earlier protocol unreproducible by name:
E2605–E2611 trained with full causal attention, and a report's `document_masking` flag did not exist when
their checkpoints were written.

## Problem

Should attention be confined to document boundaries in a packed batch, and if so, should that become the
default for every future run?

## Decision

1. **Document membership is carried by the packed example, not inferred.** `pack_units(...,
   track_documents=True)` records a document index for every input position. The default is `False` because
   the bookkeeping is per-token and only a masking run consumes it.
2. **The decoder can mask across document boundaries.** `ChakasoDecoder.forward(..., segment_ids=...)`
   intersects causality, key validity and same-document membership into one additive mask. A query may attend
   only to keys from its own document. The option is refused with a key/value cache: a cached past would need
   its own segment identifiers, and a single prompt is one document anyway.
3. **The switch is opt-in at the experiment and CLI layers.** `ExperimentPlan.document_masking` and
   `chakaso experiment model --document-masking` are off by default. A masked run requires packed examples
   with segment identifiers, and the training engine refuses a masked run over untracked examples rather than
   silently training with full attention.
4. **The transformer family refuses the option for now.** Its encoder path does not expose a per-batch
   attention mask here. A user who wants masking builds the decoder family; extending the transformer is a
   separate change with its own tests.
5. **The setting is part of resume identity.** A resumable checkpoint records it. A checkpoint written before
   the field existed is read as `False`, which is what it actually was; a checkpoint with masking on refuses
   to resume a run with it off, and vice versa.
6. **No benefit is claimed until a controlled experiment measures one.** The next experiment is a
   masked/unmasked decoder pair at the same corpus, split, seed, token budget, architecture and schedule,
   scoring held-out bits per character. Until that record exists, this is an implemented option, not a
   recommended default.

## Reasoning

- **The default protects the measured history.** Changing the default would silently make E2605–E2611's
  reports describe a protocol the code no longer runs. The project's rule is that a reproduction is a new
  record with its own deviations, not an edit to the numbers it reproduces.
- **The mechanism is testable without a training run.** A test changes a token in document A and asserts that
  document B's logits are unchanged with the mask and changed without it. That is a property of the mask, not
  a claim about quality.
- **The cost is bounded.** The mask is `batch × 1 × context × context` per layer; at the measured 128–256
  token contexts it is tens of megabytes, not a new memory regime. It is the per-token segment bookkeeping
  that scales with the corpus, which is why it is opt-in at packing time.
- **The literature is consistent with the hypothesis, not evidence for it.** OLMo and SmolLM report using
  intra-document masking; none of them isolates its effect at Chakaso's scale. This project has to measure
  its own.

## Consequences

- `PackedExample` gains an optional `segment_ids` field; `pack_units` gains `track_documents`; the decoder
  gains `segment_ids`; the training configuration and experiment plan gain `document_masking`; the CLI gains
  `--document-masking`. The default path is byte-identical to the measured one.
- A masked run's report states the flag in its budget block, so a reader can tell which protocol produced
  it.
- The planned experiment is a paired comparison against an unmasked arm at the same budget. If masking does
  not improve held-out loss, the option stays available and the default stays off; if it does, a later ADR
  proposes making it the default and declares a new protocol version rather than editing old records.

## Rejected alternatives

1. **Mask by default now.** It would change what every earlier protocol means without a measurement.
2. **Rely on the document separator alone.** The separator teaches the model where a document ends; it does
   not stop attention from reading across the boundary.
3. **Mask at inference only.** The mismatch is created during training; truncating at inference does not
   remove the spurious dependency the model learned.
4. **Implement it for both families in one change.** The transformer encoder path needs a per-batch mask
   whose shape and device behaviour differ from the decoder's, and bundling it would make the tested
   property harder to isolate. The decoder is the release path; the transformer can follow with its own
   tests.

## Conditions to revisit

1. The planned masked/unmasked experiment finds a material held-out difference in either direction.
2. The corpus mixture grows enough that most packed blocks contain several documents, making the
   cross-boundary fraction large enough to matter.
3. The transformer family needs the option for a controlled architecture comparison, at which point its
   encoder path gets a tested per-batch mask.
