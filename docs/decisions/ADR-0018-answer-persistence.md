# ADR-0018: Answer persistence is an append-only store boundary, in-memory for now

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0007](ADR-0007-content-derived-identifiers.md), [ADR-0008](ADR-0008-transactional-turns.md), [ADR-0014](ADR-0014-claim-representation.md), [ADR-0017](ADR-0017-answer-identity.md)

## Context

The correction path needs a real history. `docs/correction.md` requires that an earlier answer
stays in the record when it is superseded, because the evidence that correction works at all is
"here is what was said, here is what changed it, here is the new answer" — none of which survives
if a correction overwrites the answer it corrects. The correction *foundation* already produces an
immutable `CorrectionRecord`, but there is no first-class `AnswerRecord` and nowhere to keep the
lineage of answers across turns. ADR-0017 established the identity that lets a revised answer be a
distinct, addressable object; this decides where those objects live and what the store guarantees.

The whole system is otherwise local-first and un-persisted (ADR-0001): conversation state lives in
memory, the retrieval corpus is in-memory, nothing is written to disk. Introducing a database now
would be the opposite mistake — committing to a storage shape before a workload has shown what it
needs.

## Problem

How should answers be persisted so that history is append-only and its integrity is enforced,
while keeping the conversation manager, retrieval, model and correction engine free of the storage
mechanism — and without adopting a database prematurely?

## Options considered

- **Persist answers inside conversation state.** Rejected: a turn is the model-input-facing record;
  an answer record is evaluation- and provenance-facing, referencing claims, evidence and a
  correction lineage. Folding them together would put the shape of correction back into the shape
  of conversation (the very coupling the conversation state deliberately avoided) and would make a
  turn carry fields a model request never uses.
- **A durable backend now (SQLite, files).** Rejected: premature. The schema of an answer store is
  exactly the thing that should not be frozen before the correction and evaluation loops have
  exercised it, and ADR-0001's local-first runtime does not require persistence.
- **A store protocol with an in-memory implementation and enforced append-only invariants.** Chosen.

## Decision

`chakaso.answers` defines an `AnswerStore` protocol — `save`, `get`, `list`, `history` — with one
implementation today, `InMemoryAnswerStore`. The store, not a caller, enforces the history
invariants on `save`:

- **no overwrite**: saving a second answer under an identifier already present is an
  `AnswerStoreError`, never a silent update;
- **no dangling correction**: an answer whose `correction_of` names an answer not in the store is
  refused;
- **no cross-conversation correction**: a correction may only reach an answer in the same
  conversation;
- **acyclic by construction**: because every `correction_of` must already be saved, a correction
  always points backwards in time, so the chain cannot cycle; `history(answer_id)` walks it
  deterministically, oldest first, and raises rather than fabricating a chain for an unknown
  answer.

The conversation manager may *coordinate* persistence — build the `AnswerRecord` and hand it to a
store — but does not implement storage, keeping the boundary separate (ADR-0002's discipline
applied to persistence rather than to the model).

Persistence participates in the turn's atomicity (ADR-0008 extended to the record): every fallible
step — generation, citation resolution, building the new conversation and the record, and saving it
to the store — happens before the live conversation is swapped. A store failure therefore aborts
the turn and leaves the conversation unchanged, rather than leaving a recorded answer with no
conversation, or a conversation with no answer.

## Reasoning

- **History is the research data.** Overwriting an answer destroys the only evidence that a
  correction was warranted; refusing the overwrite is the difference between a system that can be
  audited and one that merely claims to have been right all along.
- **The boundary is what makes a backend cheap later.** Writing answer persistence, reassessment and
  (later) experiment records against `AnswerStore` means swapping in a durable store is an
  implementation change. Doing it before a real workload would only fix a schema by guesswork.
- **Letting the store own the invariants** keeps them enforced everywhere they matter, instead of
  at each call site — an integrity rule that lives in callers is one `git grep` away from being
  forgotten.

## Trade-offs

- In-memory answers are lost on process exit, exactly like conversation state. This is stated, not
  hidden: nothing here persists across a run, and `docs/agent/KNOWN_ISSUES.md` keeps the persistence
  gap (K-007's durable `AnswerRecord`) open until a real backend is warranted.
- `history` returning oldest-first assumes the acyclic-by-construction property; the store enforces
  it at `save`, so a caller never sees a chain that loops. The defensive cycle break in `history`
  can never trigger through the public API and is there only so a hand-corrupted store fails safe.

## Consequences

- Answers become first-class: the conversation manager produces an `AnswerRecord` per assistant
  turn, and reassessment can read a stored prior answer rather than being handed one ad hoc.
- Correction becomes a *lineage*: `correction_of` plus the store turns the existing
  `CorrectionRecord` decision into a traversable chain, and the answer-history immutability and
  chain integrity are now things a test can regress.
- A durable backend is a drop-in `AnswerStore` implementation later; no current module assumes the
  in-memory one beyond the protocol.
