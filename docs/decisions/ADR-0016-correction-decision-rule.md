# ADR-0016: The correction decision is an explicit, evidence-driven rule

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0014](ADR-0014-claim-representation.md), [ADR-0015](ADR-0015-grounding-boundary.md)

## Context

`docs/correction.md` makes correction architectural rather than aspirational: when better
evidence arrives, an earlier answer is *eligible* to be replaced, and the reason a system fails
to do so is that a next-token predictor prefers agreeing with the fluent answer already in its
context. The instruction "if you were wrong, say so" reduces the frequency and bounds nothing.
The alternative is a separate path with its own inputs and its own recorded outcome.

Claims (ADR-0014) and the grounding boundary (ADR-0015) already exist. Reassessment re-runs
grounding over a prior answer's claims against a pack the caller composes, then must turn the
per-claim before/after statuses into a decision. That decision is load-bearing in two opposite
directions: a rule that corrects too readily is *agreeable* and fails the "unnecessary revision"
metric, while a rule that defends the earlier answer fails the whole premise. Both failures look
like success from the outside, so the rule has to be written down, not felt.

There is no language model, so producing the *prose* of a revised answer is impossible today.
That constrains what correction can honestly be: a decision and a record, not a rewrite.

## Problem

What rule turns "the evidence now supports these statuses" into a per-claim disposition and an
answer-level decision, without either defending the earlier answer for its own sake or inventing
a change the evidence does not warrant?

## Options considered

- **Correct whenever any claim's status differs from before.** Rejected: it revises on noise —
  a claim moving from `not_evaluated` to `supported` is a reason to keep it, not to touch it, and
  a system that revises whenever challenged is not corrigible, it is agreeable.
- **A confidence threshold on "how wrong" the answer is.** Rejected: no calibrated evaluator
  exists, so a threshold would be a fabricated number (ADR-0001, `docs/evaluation.md`).
- **Let the model decide within one generation call.** Rejected outright: that is precisely the
  failure correction is meant to prevent — the earlier answer reads confident in the context and
  the model tends to defend it.
- **A stated precedence over evidence statuses, contradiction-only-triggers-correction, and a
  refusal to pick a conflict's winner.** Chosen.

## Decision

`chakaso.correction` decides in two explicit layers.

**Per claim** (`decide_claim`), from `prior_status` and the status the supplied evidence supports
`now`:

- `contradicted` → **correct** — the only route to a correction;
- `unsupported` or `uncertain`, where the claim previously read as `supported` → **qualify** —
  soften, do not delete; support weakened without anyone having been flatly wrong;
- anything else → **retain** — re-confirmation and silence are not reasons to change.

Nothing in this map inspects which answer came first, how old a source is, or any source
ranking. Correction inherits ADR-0015's refusal to decide a contest on recency or authority.

**Per answer** (`decide_answer`), a fixed precedence:

1. **correct** if any claim is contradicted;
2. **needs_review** if a conflict was recorded that was *not* adjudicated into a contradicted
   status, or a required claim was left uncertain — the engine will not pick a winner, so a human
   decides rather than the code guessing;
3. **qualify** if any claim lost support it had;
4. **abstain** if the answer had required claims and none is supported by the reassessment
   evidence — withholding beats asserting something the evidence no longer grounds;
5. **retain** otherwise. An answer with no claims retains; it is not reported as an abstention.

A status is always "supported / unsupported / contradicted / uncertain *by the supplied
evidence*" (ADR-0014/0015), never "true." `contradicted` and `uncertain` can only enter through a
judgement a caller supplies, because structural matching cannot notice disagreement.

## Reasoning

- **Contradiction is the only definite wrongness.** Losing support is not the same as being
  opposed; qualifying rather than deleting a weakened claim is what `docs/correction.md` asks for
  ("qualifying a claim whose evidence turns out to be weaker ... does not require anyone to have
  been wrong").
- **`needs_review` and `abstain` are the conservative exits.** Where the engine cannot settle a
  conflict or ground the answer at all, it declines to fabricate a confident revision — the exact
  failure mode the transparency rules forbid.
- **Deciding without rewriting is the honest subset today.** The record captures what changed and
  why; the revised wording is deferred to a model that does not exist, so correction claims a
  decision, not a finished answer.

## Trade-offs

- Structural grounding can only ever produce `retain`/`qualify` (never a correction), because a
  `contradicted` status requires a supplied judgement. A fully automatic correction loop needs a
  semantic evaluator that does not exist; the boundary for one is in place (ADR-0015).
- The answer-level `abstain` condition depends on the pack the caller passes in. Reassessing
  against new evidence alone can make a still-valid claim look unsupported; the contract is that
  the caller composes the pack (standing evidence plus the new evidence), and the engine reports
  only what *that* pack supports. That is stated, not hidden.

## Consequences

- `Reassessment` carries the per-claim `ClaimReassessment`s, the answer-level `CorrectionDecision`,
  the contradictions it was given, and the evaluator name — so a report shows *how* a decision was
  reached, not only what it was.
- The next unit records a reassessment as an append-only correction (a record referencing the
  answer it supersedes) and measures the rule over labelled cases; neither produces a model-quality
  result, and no measured correction outcome exists yet.
- Nothing here wires into `ConversationManager` or the CLI as an automatic loop; reassessment is
  invoked by a caller on evidence it supplies.
