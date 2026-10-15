# ADR-0028: An answer persists the historical evaluation it reached; the live view stays recomputed

- Status: Accepted
- Date: 2026-10-15
- Relates to: [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0009](ADR-0009-unresolved-references-are-recorded.md), [ADR-0014](ADR-0014-claim-representation.md), [ADR-0015](ADR-0015-grounding-boundary.md), [ADR-0017](ADR-0017-answer-identity.md), [ADR-0018](ADR-0018-answer-persistence.md), [ADR-0021](ADR-0021-sqlite-answer-store.md)

## Context

ADR-0021 deliberately refused to persist an answer's `evaluation` object. An
`AnswerEvaluation` is a *view*: it is computed from the record's claims and evidence by
whichever grounding evaluator runs at read time. Freezing that view into durable history
would capture a possibly-stale judgement of an ephemeral computation, and worse, it would
let a record silently claim *what today's evaluator thinks* while presenting it as what the
original turn decided. The record therefore persisted only the durable substance — the text,
the model, the claims, the evidence it was given, the sources it cited, and the correction
link — and dropped the evaluation.

That was the right call about the *live view*, but it left a real auditability gap. A
completed turn genuinely reaches evaluation outcomes — which citations resolved and which
were fabricated, what structural-grounding status each claim was assigned, which evaluator
(name and version) produced them, and which references in the answer text did not resolve.
Those are not a recomputable view; they are **facts about the past**. Losing them means a
stored answer cannot explain how it was judged when it was written, and (per K-006) the rate
at which a model invents an evidence reference — a headline measurement in
`docs/evaluation.md` — could never be computed from anything the repository keeps.

## Problem

How can a persisted answer preserve what its turn actually evaluated to — enough to be
audited after a restart and to make the invented-reference rate measurable — without either
freezing a recomputable evaluator snapshot or blurring the line between "what the turn
decided then" and "what a current evaluator decides now"?

## Options considered

- **Persist the whole `AnswerEvaluation` object.** Rejected: it freezes a recomputable view,
  couples durable history to an evaluator's in-memory shape, and makes a later evaluator's
  opinion indistinguishable from the original turn's — the exact confusion ADR-0021 avoided.
- **Keep persisting nothing and recompute on every read.** Rejected: it cannot answer "what
  was decided at the time," cannot distinguish an original outcome from a reassessment, and
  leaves the invented-reference rate unpersisted (K-006).
- **Persist a separate, immutable historical snapshot of the outcomes the turn reached, while
  the live view stays recomputed and unpersisted.** Chosen.

## Decision

`AnswerRecord` gains a persisted, typed, immutable `evaluation_provenance:
EvaluationProvenance | None` field (`chakaso.answers.provenance`), captured from the turn's
own evaluation and citation resolution at write time. It records, as separate dimensions that
are never merged into one score:

- the **citation-structure** outcome: claims, citations made, and how many were valid,
  unknown, irrelevant, and uncited-of-required;
- the **per-claim structural-grounding** status the turn assigned, and which claims were left
  not-evaluated;
- the **evaluator identity and version** (`grounding_evaluator`, `grounding_evaluator_version`)
  that produced the grounding outcome, so a change of judge is visible in provenance and an
  unknown third-party evaluator name stays exactly as recorded;
- the **evidence references found in the answer text that did not resolve** —
  `unresolved_reference_ids` and `malformed_reference_ids` — the K-006 condition, now a
  durable fact.

The live `AnswerEvaluation` field remains **non-persisted and recomputed** (ADR-0021 is
preserved, not reversed). Derived values (`evidence_coverage`, `is_grounded`, citation
precision/recall) are computed from the stored raw counts and statuses, never stored as a
single opaque number, and `is_grounded` is labelled the structural definition it is — "every
claim cited supplied evidence" — not a truth verdict (ADR-0014, ADR-0015).

The snapshot is written by the `ConversationManager` as it already computes the turn's
evaluation, so no new evaluator runs and the development double gains no fabricated
capability: the evaluator named in a default record is the honest `structural-0.1`.

The stored representation is **backward compatible without a schema change**: the snapshot
lives inside the answer's JSON `document` (the SQLite table and `PRAGMA user_version` are
unchanged). A document that predates this field simply lacks it and decodes to `None`; a
present-but-malformed snapshot is rejected as corruption. `None` is the honest "no provenance
was captured for this answer," never an empty snapshot that would read as "evaluated and found
lacking."

## Reasoning

- **A historical fact and a recomputable view are different objects; conflating them is the
  failure mode.** Keeping two fields — one persisted, one not — makes "what was decided then"
  and "what is decided now" simultaneously expressible and impossible to confuse.
- **Persistence follows the append-only model.** A reassessment writes a new answer or a new
  correction record with its own snapshot; it never rewrites the original's `evaluation_provenance`,
  so the original outcome stays exactly what the first turn recorded.
- **The minimal sufficient set, not a duplicate.** Everything the snapshot stores is what the
  turn decided and is not otherwise derivable from the durable record after a restart; claim
  texts, citations and evidence references already live on the record and are not copied in.

## Trade-offs

- The snapshot is a superset of stored bytes on the `document`; it is small (counts, statuses
  and reference strings) and deliberately does not copy evidence bodies. A full immutable
  evidence snapshot is explicitly not taken (see `docs/retrieval.md` and K-001) — a reference
  that outlives the in-memory corpus records that the evidence is no longer retrievable rather
  than duplicating it.
- Recomputing an evaluation can now *differ* from the recorded snapshot. That divergence is the
  point, but it means inspection must label which it is showing; the CLI labels the snapshot
  "historical, not recomputed" and reports no live view for a reopened record.

## Consequences

- `chakaso answers show`/`list` can distinguish a recorded historical outcome from a
  recomputation, name the evaluator, and expose unresolved/malformed references; `--json` keeps
  the exact stored values while the human view renders them terminal-safe.
- K-006 is advanced, not fully closed: the invented-reference condition is now persisted on every
  durable answer, so the rate becomes computable from history once the durable store is selected.
- A real semantic evaluator, when it exists, records its own name and version into the same
  snapshot through the existing `GroundingResult`, with no change to the record or codec.
- The stored-document format gained a field; the SQLite schema version did not change, and
  forward migration is not required because absence decodes to `None`.
