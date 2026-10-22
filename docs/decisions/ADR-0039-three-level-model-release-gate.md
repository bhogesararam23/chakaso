# ADR-0039: a model release is gated at three levels, and the level is awarded on evidence in the model card

**Status:** Accepted
**Date:** 2026-10-22
**Amends:** [ADR-0033](ADR-0033-model-centred-product-contract.md) (the model is the product, so the model
gets a release gate rather than a changelog), and it is the gate that
[ADR-0037](ADR-0037-reference-model-sized-by-measured-throughput.md)'s reference model must pass through.
**Related:** [ADR-0035](ADR-0035-transformer-checkpoint-format.md),
[ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md),
[ADR-0038](ADR-0038-local-http-serving-adapter-from-the-standard-library.md), E2608, `MODEL_CARD.md`.

## Context

The repository now ships a real artifact: a 3,413,760-parameter decoder trained from random initialization on
14,490,005 characters of licensed public-domain prose, loadable through the Python API, verifiable against its
own contract, serveable over local HTTP, and recorded in a preregistered experiment. It is also, on the same
evidence, a model whose generations repeat phrases and answer nothing.

That combination is where research projects get dishonest. The pressure is not to lie; it is to let a set of
achieved sub-goals imply a goal that was not met. "The contract verifies, the pipeline reproduces, CI is green,
the loss beats every baseline" is a true sentence about `chakaso-base@0.1.0`, and a reader can easily finish it
as "so it is a working language model". It is not, and the project's own non-negotiables say a capability claim
needs a status label and a measurement behind it.

A release needed a rule that forces the behaviour claim to be settled separately from the plumbing claims — in
writing, before anyone reads the plumbing as a capability.

## Problem

Define what it takes for this project to release a model, such that:

- integration work cannot be mistaken for capability;
- a level can be lowered or withheld without re-litigating every other fact;
- the level a model earned travels with the model rather than living in a roadmap;
- passing a lower level never makes a higher one automatic.

## Decision

Three levels, defined by the evidence that closes them, and assessed in the model card — not in a roadmap, not
in a changelog, and not in conversation.

**Level 1 — Artifact contract.** A named, versioned artifact exists whose provenance it states itself:
architecture, tokenizer, corpus identity and licence, budget, measured held-out metrics, and non-empty
`known_limitations`. Loading verifies; substituting is refused.

Closed by: `chakaso model verify` succeeding on the release directory; the checkpoint's SHA-256 matching the
contract; the contract refusing to build when a report's family and its artifact's schema contradict each other
or when a required statement is absent; a load of the release directory producing the same model as a load of
the file it names.

**Level 2 — Reproducible integration.** A developer can reproduce the artifact from a pinned input and use it
through a stable interface without a pretrained dependency anywhere on the path.

Closed by: an experiment record that preregisters its corpus fingerprint, split, tokenizer, budget and primary
comparison *before* the runs, and states the exact commands; a corpus snapshot verified by digest rather than
read from the working tree ([ADR-0036](ADR-0036-corpus-snapshot-recovery-from-history.md)); the vertical slice
running train → save → load → generate → stream → cancel → score → release → serve from a clean install; the
model path verified on the runners, not only on a maintainer's machine; and random initialization — weights that
originate in this repository's own seeded training, with no hosted API, pretrained weight, or third-party model
appearing as the solution.

**Level 3 — Useful behaviour.** The model does something a caller would want done, measured as doing it.

Closed by: a measured result on a workload that is not this repository's own writing or a self-authored
fixture, where the *model* produced the text and the harness only scored it — with failures reported, and with
the distinction the project insists on: a retrieval benchmark, a reference oracle or a citation checker passing
is evidence about the plumbing, never about language-model competence. Until then a grounded answer is not
answered, a instruction is not followed, and neither is claimed.

**Rules about the levels.**

1. **Levels do not imply each other.** Passing 1 and 2 says nothing about 3. The card states each verdict
   separately, and a headline number may not be used to imply a level.
2. **Level 3 is not awarded for effort.** The brief for the foundation release anticipated the temptation
   directly: a complete, well-engineered, reproducible toy earns Levels 1 and 2 and is described as what it is.
3. **A level can be withdrawn.** If later evidence contradicts a verdict, the card changes and the change is
   recorded as a deviation, not as a quiet edit — the same rule that makes a reproduction a new record rather
   than an edit to the numbers it reproduces.
4. **The default stays the double until Level 3.** A model at Level 2 is loadable, serveable and measurable; it
   is not wired into the runtime as the answer to a grounded question (ADR-0033's opt-in rule), because
   deployment is how an unfixed degeneration becomes somebody else's product.

## Assessment of `chakaso-base@0.1.0`

Level 1 **met**, Level 2 **met**, Level 3 **not met and not claimed**. The evidence for each verdict, including
the measurements that withhold the third, is in [`MODEL_CARD.md`](../../MODEL_CARD.md); the generation statistics
that decide Level 3 are mean bigram repetition 0.5538 over five greedy continuations, zero tail loops, zero
periodic runs of three or more, every sample truncated at its bound, and no sample anyone would publish.

That is the first release this project can gate at Level 2, and it is also the first time the project has
measured non-cyclic output from its own model — both facts belong in the same sentence, which is the sentence
the card writes.

## Consequences

- `MODEL_CARD.md` becomes the normative statement of what a released model is, and the levels are rows in it
  with evidence attached. A release without a card is not a release.
- The gate is checkable: Level 1 and 2 have commands behind them (`model verify`, the vertical slice, the
  pinned-snapshot re-run), so a card that claims them can be tested rather than trusted.
- Level 3 stays open, and roadmap Phase 5 (a trained model in front of a workload) is the work that could close
  it. Nothing in Levels 1–2 shortcuts that.
- Application-layer milestones can no longer be read as model milestones: the gate separates them by
  construction, which is the discipline ADR-0033 asked for and this record makes operational.

## Conditions to revisit this decision

- If a released model ever reaches Level 3, the default-model question reopens: whether a Level-3 model should
  answer turns by default, and with which refusals.
- If a fourth axis turns out to be needed — safety or bias behaviour, say — it becomes a fourth level with its
  own evidence requirement, not a footnote under Level 3.
- If the project ships more than one artifact, the card becomes per-release (carried inside the release
  directory beside `model.json`) rather than one file at the repository root.
