# ADR-0017: Answer identity is a per-event identifier, not a content hash

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0007](ADR-0007-content-derived-identifiers.md), [ADR-0003](ADR-0003-evidence-identifier-ownership.md)

## Context

Evidence identifiers are content-derived: the same source bytes deduplicate to one
identifier, and a changed page yields a distinguishable one (ADR-0007). That property is
load-bearing — it is why an answer can be re-examined against the exact bytes it used and why
retrieval does not double-count a document.

An answer is a different kind of thing. It is a runtime event: the model produced this reply,
at this turn, in this conversation, given this evidence. The system now needs to record answers
(`AnswerRecord`) so that corrections can form a traceable history. That raises a question the
evidence rule does not answer: should an answer's identifier also be derived from its content?

Deriving it from content is the obvious move, because everything else here does it, and it
makes an identifier reproducible. But it is wrong for a history of corrections.

## Problem

What identifier scheme lets a corrected answer be a distinct, addressable record even when it
says exactly what the answer it supersedes said, without breaking the reproducibility the rest
of the system relies on?

## Options considered

- **Content-derived (`hash(conversation, turn, text)`).** Rejected: it collapses two identical
  answers into one identifier. A correction that re-affirms or repeats wording would collide
  with the answer it replaces, so "the first saying" and "the second, after correction" would be
  indistinguishable — the exact history the feature exists to preserve. It also cannot be
  computed before the answer exists, which is a usability trap for a record that must be created
  as part of producing the turn.
- **A content-derived id over an added nonce/sequence.** Rejected: that is a random identifier
  wearing a content hash's clothes — the nonce makes it non-reproducible anyway, so it buys the
  collision-freedom of randomness while inheriting none of the deduplication that justifies a
  hash.
- **A typed, validated random identifier per answer event.** Chosen.

## Decision

`chakaso.answers` defines `AnswerId` as the same validated shape as every identifier in the
system — `ans_` plus a 64-bit hexadecimal digest — so it is recognisable in a log and checked
before any lookup (the `Identifier` base, reused from `chakaso.core`). `new_answer_id()` returns
a random token per event, following the existing precedent of `new_conversation_id()`: a
conversation, like an answer, has no content identity — two sessions that open with the same
question are different conversations.

Content-derived identity stays exactly where ADR-0007 put it: on evidence. An `AnswerRecord`
references evidence, claims and evaluation by their own identifiers; only the answer itself is
identified by the event.

## Reasoning

- **A correction must be distinguishable from what it corrects.** That is the whole point of an
  append-only answer history, and it fails under content identity the moment a revised answer
  repeats any of the original wording.
- **Reproducibility comes from the transcript, not the name.** An answer can be replayed and
  audited from its recorded text, model, evidence and evaluation; the identifier is a handle, not
  a proof. Deterministic *ordering* of history is provided by the store's insertion order, not by
  sorting on a content hash.
- **Reuse the validated shape.** Random does not mean unvalidated; `AnswerId` is a frozen,
  ordered `Identifier`, so a malformed reference from untrusted text fails at construction like a
  source or chunk identifier does.

## Trade-offs

- A random answer identifier is not reproducible across runs, so an `AnswerId` cannot be used to
  align two experiments. Experiments align on content (answer text, claims, evidence identifiers)
  and on the deterministic benchmark/evaluator versions, not on per-event ids. This is stated so
  no one reaches for an answer id as a join key.
- The store, not the identifier, is responsible for detecting a duplicated save; it does so by
  `AnswerId` presence, which a random id makes unambiguous.

## Consequences

- `AnswerId` is generated at record-creation time and never derived from a field of the record.
- `AnswerStore` (next) keys on `AnswerId` and treats a second save under an existing id as an
  error rather than an update — history is appended to, never overwritten.
- `correction_of` references another `AnswerId`; because ids are unique per event, a correction
  chain is a genuine graph, and self-reference is detectable rather than silently deduplicated.
