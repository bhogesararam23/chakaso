# ADR-0029: Reassessment is triggered by an explicit identifier and composed at the application root

- Status: Accepted
- Date: 2026-10-15

## Context

The correction package could already *perform* a reassessment: `reassess_and_record` takes a stored
answer's `AnswerId` and new evidence, re-grounds its claims, and returns a decision plus a `CorrectionRecord`.
The retrieval-aware runtime could already conduct a grounded conversation turn and record each answer.
What was missing was the connection between them: nothing in a conversation *initiated* a reassessment.
The literature and `docs/correction.md` describe the intended behaviour as a follow-up that arrives with
better evidence causing an earlier answer to be reconsidered.

The tempting shape — watch every turn and decide, automatically, whether it warrants revising a prior
answer — cannot be honoured. Deciding *when* to reassess, and resolving a vague reference like "my
previous answer about the deadline" to one specific stored answer, is a language-understanding task. There
is no language model (the only implementation is a deterministic development double), and the query
planner is a documented keyword heuristic with no notion of an answer identifier. Building an autonomous
loop on top of that would be a fabricated capability: the appearance of self-correction with no machinery
that could justify it.

## Problem

How can a follow-up in a real conversation initiate reassessment of a stored answer, while staying
honest about the fact that the system cannot decide on its own when to revise, cannot resolve a vague
reference, and cannot generate a revised answer?

## Options considered

- **An autonomous trigger.** A model or heuristic inspects every turn and decides whether to reassess.
  Rejected: it requires the very inference that does not exist, and a wrong trigger either silently
  rewrites history or silently ignores a genuine correction. It is the fake autonomous loop the roadmap
  explicitly refuses.
- **A planner cue.** Extend `DeterministicQueryPlanner` to emit a "reassess" intent alongside its
  retrieval modes. Rejected: the planner decides *whether and how to retrieve*, on conversation state it
  has; it has no access to answer identity and adding a second, unrelated decision to it would blur two
  boundaries (ADR-0022). Intent detection would no longer be independently testable from retrieval.
- **An explicit, identifier-addressed trigger composed at the application root.** A follow-up that
  carries a well-formed `ans_…` reference names its own target; a deterministic trigger recognises that
  reference, and the composition root calls the existing reassessment boundary. Chosen.

## Decision

Reassessment is initiated only by an explicit reference, and the intent to reassess is separated from the
act of reassessing:

1. **A deterministic trigger detects intent, nothing else.** `chakaso.correction.trigger` classifies a
   message as naming zero, one, or several distinct answer identifiers. It reads no store, runs no
   evaluator, retrieves no evidence and mutates nothing. It fires on an explicit, validated `ans_…`
   reference and never infers one from a vague phrase.
2. **Ambiguity is refused, not guessed.** Two or more distinct identifiers are an ambiguous request and
   raise a typed error naming what was seen; the runtime does not pick one. A message naming no identifier
   is an ordinary conversational turn. This is ADR-0003's discipline — the caller owns the identifier, the
   system does not invent or infer it — applied to *intent*.
3. **The composition root wires trigger to boundary.** `chakaso.app.Application` composes the trigger
   with the existing `reassess_and_record`. Evidence is the answer's standing evidence (the corpus chunks
   still present for the sources it named) merged with what the retrievers return for the follow-up
   through a read-only `gather_evidence` seam — no model call, no recording, no state change.
4. **The original answer is never rewritten.** A reassessment produces a distinct, content-identified
   `CorrectionRecord` naming the target it supersedes. The `AnswerRecord` and its historical
   `evaluation_provenance` are unchanged (ADR-0018, ADR-0028). No revised prose is generated, because
   there is no model; the record's summary is a structural tally.
5. **Correction records are surfaced, not auto-persisted.** This decision does not change that a
   `CorrectionRecord` is returned and displayed rather than written to the answer store; automatic
   persistence of correction history remains out of scope.

## Reasoning

A named identifier is a deliberate act by a caller, not a guess by the system, so the trigger's firing is
as reproducible and inspectable as the planner's rules — and, unlike a heuristic that "senses" a
correction is needed, it can never be wrong about *which* answer it means in a way the user did not
choose. Keeping detection separate from execution means each is testable alone, which is what lets the
composition be trusted without an autonomous loop.

## Trade-offs

Reassessment requires the user (or a calling program) to know and supply the answer identifier; it is not
something that happens unbidden. The runtime cannot tell that a follow-up *implicitly* contradicts an
earlier answer. That limitation is accepted precisely because closing it would require a model that does
not exist, and pretending otherwise would corrupt the honesty the project is built on.

## Rejected alternatives

The autonomous trigger and the planner-cue alternatives are rejected above; both would put a decision the
system cannot make into a place that implies it can. A third rejected option was to fold trigger detection
into the CLI: it belongs behind the same boundary the library exposes, so a non-CLI front end can compose
the identical behaviour.

## Consequences

- `chakaso.chat` routes a follow-up that names a stored answer into reassessment and reports the
  decision honestly, and now surfaces each recorded answer's `ans_…` identifier so a later turn can name
  it. The feature is reachable from the supported runtime interface, not only callable in a test.
- A new typed error, `AmbiguousReassessmentTargetError`, joins the correction path's contract.
- The `LanguageModel` boundary, the answer store, and the reassessment boundary are unchanged; this is
  composition, not new surface area.
- A real model that could resolve references and generate revised prose would replace only the *evidence
  of intent* and the *output wording*, not this identifier-addressed decision rule.
