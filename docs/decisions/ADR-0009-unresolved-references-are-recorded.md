# ADR-0009: A reference the model was not given is recorded, not fatal

- Status: Accepted
- Date: 2026-10-07
- Interprets: [ADR-0003](ADR-0003-evidence-identifier-ownership.md)

## Context

[ADR-0003](ADR-0003-evidence-identifier-ownership.md) establishes that the
retrieval layer owns source identity, that generated text may only reference
identifiers it was supplied, and that any identifier in generated output which was
not supplied "is treated as an error condition, not as a source. It must never be
resolved to a URL, and it must be recorded, because fabricated identifiers are a
metric (unsupported claim rate), not a cosmetic defect."

"Recorded" and "an error condition" can be read two ways: as a condition to be
noted and passed on, or as a reason to reject the turn. Implementing the manager
forced the question, because a turn now has somewhere to report it.

## Problem

When a model references an identifier it was not given, should the turn fail, or
should the answer be returned with the condition recorded?

## Options considered

- **Fail the turn.** Raise, discard the generated text, leave the conversation
  unchanged (as [ADR-0008](ADR-0008-transactional-turns.md) requires for failures).
- **Record the condition and return the answer.** Never resolve the identifier to a
  source, and report it to the caller alongside the answer.
- **Drop the reference silently** and present the rest of the answer as if the
  citation had not been written.

## Decision

The reference is recorded, not fatal. The manager never resolves an identifier it
was not given, and it reports the outcome to the caller:

- `resolve_citations` returns resolved citations, unknown references and malformed
  references as three separate results;
- the turn never records a cited source that was not in the supplied evidence;
- the reply carries the resolution, so a caller cannot render a fabricated citation
  as a source, and a caller that ignores the resolution has ignored a field that is
  named for exactly what went wrong.

## Reasoning

- **The metric is the point.** The project's headline measurements are citation
  precision and unsupported claim rate. Those are rates over turns. Failing a turn
  destroys the observation that would measure it, and a system that fails on the
  behaviour it is trying to measure cannot report the behaviour at all.
- **The enforcement is unchanged either way.** What ADR-0003 protects against is a
  fabricated reference becoming a rendered URL. That is prevented by never
  resolving an unsupplied identifier, which happens whether the turn fails or not.
  Failing adds no safety; it only removes the observation.
- **The answer may still be useful.** The parts of an answer that are not the
  fabricated citation can be correct and relevant. Discarding them serves the
  system's tidiness rather than the user.
- **Silence is the one unacceptable option.** Dropping the reference and presenting
  the rest would leave a reader believing the answer was checked. A recorded
  condition at least travels with the answer.

## Trade-offs

- **A defective answer is returned.** A consumer has to look at the resolution to
  know. Mitigated by naming the field for the condition and by keeping the answer's
  provenance in the same object, but a caller that ignores it will treat a
  fabricated citation as an ordinary answer.
- **Nothing persists the rate yet.** There is no evaluation harness and no store, so
  the condition is observable only in-process, in the reply. Recorded as a known gap
  rather than solved by adding a field to conversation state that nothing reads.
- **`is_clean` is a convenience that invites a boolean.** Reducing three outcomes to
  one flag would lose the distinction between a model that invented a source and one
  that fumbled the format. The flag exists for callers that only need "was anything
  wrong", and the three results remain available.

## Rejected alternatives

**Failing the turn** was rejected because it trades a measurable behaviour for a
tidier state, and because it makes the manager's failure semantics do double duty:
"the model was unreachable" and "the model invented a source" are different facts
about a run, and only the first is a failure of the turn.

**Dropping the reference silently** was rejected because it is the only option that
makes the system's output less honest than the model's.

## Consequences

- `Reply.citation_resolution` carries the outcome of every reference in the answer,
  and `Reply.has_unresolved_references` is a convenience over it.
- The assistant turn records the evidence it was given and the sources it actually
  cited, and can never record a cited source outside that evidence — the state type
  rejects it.
- Measuring the rate at which this happens requires the evaluation layer. Until it
  exists, this is a known gap in `docs/agent/KNOWN_ISSUES.md`, not a solved problem.
- A caller that renders answers to users is responsible for not presenting an
  unresolved reference as a source. The manager enforces this by never resolving
  one, but it cannot stop a caller from inventing a rendering of its own.
