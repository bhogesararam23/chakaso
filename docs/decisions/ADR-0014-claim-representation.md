# ADR-0014: A claim is a first-class, content-identified unit; status is evaluation, not truth

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0007](ADR-0007-content-derived-identifiers.md)

## Context

Grounding, citation validation and correction are all defined over *claims*, not whole
messages. `docs/correction.md` requires correction to act on a claim ("the population
figure is wrong but the trend is right" must be expressible); `docs/transparency.md`
requires the system to distinguish what evidence supports from what it infers. Neither is
possible until a claim is a thing the code can hold.

There is no trained model to produce claims — the only model is a development double — so
this decision is about the *representation*, not about automatic extraction. Getting the
data model right now means extraction, validation and correction can be built against a
stable unit instead of inventing one each.

## Problem

What is a claim, how is it identified, and what does its "status" mean — without implying
that the system can determine truth?

## Options considered

- **Treat the whole answer as the unit.** Rejected: it cannot express a partially
  supported answer, which is the exact case correction must handle.
- **Free-text claims with a running counter id.** Rejected: a counter id is not
  reproducible across runs and makes a citation to "claim 2" ambiguous once the answer is
  re-decomposed.
- **An immutable claim with a content-derived identifier and an evaluation status that is
  explicitly *not* a truth judgement.** Chosen.

## Decision

`chakaso.claims.Claim` is a frozen record: `answer_id`, `position`, `text`,
`cited_evidence_ids` (existing `ChunkId`s, never a new evidence type), and a `status`. Its
`claim_id` is derived from `(answer_id, position, text)` exactly as chunk identifiers are
derived from content (ADR-0007), so the same claim in the same answer always has the same
id and setting a status — which must not change what a citation points at — leaves the id
untouched.

`ClaimStatus` is a closed set — `supported`, `unsupported`, `contradicted`, `uncertain`,
`not_evaluated` — and `not_evaluated` is the default. **Status describes the relationship
between a claim and the evidence that was supplied; it never asserts that the claim is
true in the world**, and the two are kept distinct in documentation and in the evaluator
that consumes them. `with_status` returns a copy, so a claim's identity and citations are
stable and history is not mutated.

## Reasoning

- **A stable, derived id is what makes a claim citable and correctable.** Grounding
  results and correction records point at claims by id; that only works if the id is
  reproducible and independent of evaluation state.
- **`not_evaluated` as the default prevents an unexamined claim from reading as checked.**
  The dangerous failure is a system that quietly promotes the unevaluated to the supported.
- **Reusing `ChunkId` for citations keeps a single evidence identity** (ADR-0003): a
  claim cites the same chunk identifiers retrieval produced and the evidence pack contains,
  so citation validation is a check against the pack, not a new scheme.

## Trade-offs

- **Deriving the id from text means re-worded claims are different claims.** That is
  correct — a different assertion — but it means a correction that rewrites a claim's text
  produces a new claim id, which the correction record must link explicitly rather than
  assume. This is handled by the correction record (a later ADR), not hidden here.
- **`contradicted` and `uncertain` cannot be justified by the structural evaluator alone.**
  They exist because the *model* of correction needs the states; the structural grounding
  layer will only assign the ones it can actually support, and stronger semantic judgment
  is a documented future boundary, not a present claim.

## Consequences

- Extraction, citation validation, grounding and answer evaluation are all defined over
  `Claim`.
- `chakaso.claims.model` owns `Claim`, `ClaimId` and `derive_claim_id`; nothing else
  constructs a claim id by hand.
- The separation "supported by supplied evidence" versus "true" must survive every
  report and metric name that follows.
