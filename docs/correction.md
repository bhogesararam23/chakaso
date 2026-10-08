# Correction

**Correction over consistency** is the principle that separates Chakaso from a
system that defends its transcript. When better evidence arrives, an earlier
answer is eligible to be replaced, and the replacement is recorded.

## Why this needs to be architectural

A language model asked to reconsider its own earlier answer tends to agree that it
was right. The previous answer is in its context, it reads as fluent and
confident, and nothing in a next-token predictor prefers the discomfort of being
wrong. Prompt instructions ("if you were wrong, say so") reduce the frequency of
this and do not bound it.

Correction is therefore treated as a separate path with its own inputs and its own
recorded outcome, rather than as something hoped for inside one generation call.

## The contract

```text
reassess(previous_answer, new_evidence, conversation_state)
    -> compare the previous answer's claims against the new evidence
    -> classify each claim: supported | unsupported | contradicted
    -> decide per claim: retain | qualify | correct
    -> produce a revised answer
    -> record what changed and why
```

The signature is a contract, not an implementation. What it obliges:

**Inputs are explicit.** Reassessment takes the previous answer, new evidence, and
conversation state as arguments. It does not re-read a chat log and guess what was
claimed.

**Claims are the unit.** Correction happens at the level of a claim, not a whole
message. "The population figure is wrong but the trend is right" must be
expressible; a system that can only accept or reject an entire answer will accept
it.

**The outcome is one of three, per claim.** Retaining a claim is a decision, not a
default. The same applies to qualifying a claim whose evidence turns out to be
weaker than it appeared, which is a form of correction that does not require
anyone to have been wrong.

**The revision is recorded.** The new `AnswerRecord` references the record it
replaced. History is appended to, never rewritten: the earlier answer stays in the
record with a note that it was superseded. Silently mutating an earlier answer
destroys the evidence needed to evaluate whether correction works.

## What must not happen

- Defending a previous answer because it was produced first.
- Restating a contradicted claim in softer language while keeping its substance.
- Claiming to have been corrected when the evidence did not require it — a system
  that agrees with every challenge is not corrigible, it is agreeable, and it is
  worse because it looks like the behaviour being asked for.
- Discarding the previous answer from the record.
- Correcting in a way the user cannot see. A revision that is not announced leaves
  the user holding a belief the system no longer holds.

## What a correction should sound like

A correction is a normal sentence, not a ceremony. It states what changed and
what changed it:

> I said the deadline was in March. The specification I retrieved now says the
> submission window closes on 14 April, and the March date I used came from a
> news article that predates it. The April date is the one the specification
> supports.

Contrast with two failure modes that are easier to generate and worse:

> You're right, I apologize for the confusion. The deadline is in April.

This asserts agreement without establishing what the correct answer is or why it
changed. It is indistinguishable from capitulation.

> As I mentioned earlier, the deadline is in March, though there are reports of an
> April date.

This preserves a contradicted claim by mentioning the contradiction without
resolving it.

## Relationship to transparency

A correction is the most important thing Chakaso can be transparent about,
because it is the moment the system's earlier output is shown to be inadequate.
The correction notice uses the same categories as
[`transparency.md`](transparency.md): what was observed, what was inferred, what
was assumed, and which of those changed.

## Metrics

The behaviour is measured, not asserted
([`evaluation.md`](evaluation.md)):

| Metric | Question it answers |
| --- | --- |
| Correction Success Rate | When revision was required, did the system reach the correct revised conclusion? |
| Unjustified Persistence Rate | How often did it keep defending an unsupported earlier answer? |
| Unnecessary Revision Rate | How often did it revise an answer the evidence still supported? |

The third metric exists because the first two can be gamed by a system that
revises whenever challenged. Both directions are failures.

## Status

The vocabulary this contract consumes now exists. A claim is a first-class, immutable record
with a content-derived identifier and an evaluation status (`chakaso.claims`, ADR-0014);
structural citation validation and a grounding boundary are in place (`chakaso.citation`,
`chakaso.grounding`, ADR-0015); and a claim can already be classified **supported** or
**unsupported** by the supplied evidence (with **contradicted** and **uncertain** available
only when a caller supplies that judgement). `evaluate_answer` keeps these as separate
dimensions.

What does **not** exist is the reassessment path itself: nothing takes a previous answer plus
new evidence, compares the claims, decides retain / qualify / correct, produces a revised
answer, or records the change. The `AnswerRecord` type that would carry `correction_of` does
not exist, and no correction loop runs. The status of this area is **Planned**, and the design
above is the specification to build against. The claim and grounding primitives are its
inputs, not its implementation.
