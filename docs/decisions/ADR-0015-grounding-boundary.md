# ADR-0015: Grounding is a boundary; structural evaluation never claims semantic support

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0003](ADR-0003-evidence-identifier-ownership.md), [ADR-0014](ADR-0014-claim-representation.md)

## Context

The project's central promise is that an answer can be traced to the evidence that
supports it. "Supported" is a load-bearing word: a system that marks a claim supported
because it *mentioned a source near it* is worse than one that abstains, because it looks
verified and is not. `docs/transparency.md` requires the system to distinguish observed
evidence from inference; `docs/correction.md` requires claims to be classified
supported / unsupported / contradicted.

There is no trained model and no semantic entailment checker. What can be computed today
is structural: does a claim cite a chunk that was actually supplied, and does it cite the
chunk a benchmark expected? That is not the same as "the source proves the claim," and
the whole risk is a codebase that quietly treats the first as the second.

Contradiction adds a second trap: given "X in 2024" and "X in 2025," a system can either
refuse to choose, or pick one by a rule. If it picks, the rule must be explicit and owned
by the case, not a hidden global preference for newer or for some domain.

## Problem

How should grounding be represented and evaluated so that structural capability is not
overstated as understanding, and conflict is recorded without an arbitrary winner?

## Options considered

- **A single boolean "is grounded" for a whole answer.** Rejected: it cannot express a
  partially supported answer and it hides which claim failed — the opposite of the point.
- **A confidence score per claim.** Rejected: no calibrated evaluator exists, and a number
  would be fabricated (`docs/evaluation.md` forbids it; ADR-0001's honesty).
- **One `GroundingEvaluator` boundary with a structural implementation today and a manual
  judgement path for the states structural matching cannot decide.** Chosen.

## Decision

`chakaso.grounding` defines a `GroundingEvaluator` protocol returning a `GroundingResult`:
a `ClaimStatus` per claim, plus any `Contradiction`s and the name/version of the evaluator
that produced it.

- `StructuralGroundingEvaluator` assigns only `supported` (a valid citation exists),
  `unsupported` (a claim that required evidence has none that is valid) and
  `not_evaluated` (nothing decided). It **cannot** emit `contradicted` or `uncertain`.
- `ManualGroundingEvaluator` carries statuses and contradictions a caller supplied — the
  stand-in for a human label or a future model, and the only way those stronger states
  appear. It exists so the *interface* to semantic grounding is real and tested now,
  without the semantic capability being claimed.
- A `Contradiction` records ≥2 disagreeing chunks for a claim and names no winner. A
  precedence is an explicit, caller-supplied rule; `most_recent_wins` is provided as a
  sample and returns "no winner" rather than guessing on a tie or missing dates. No
  precedence is applied by the engine automatically.
- `SourceAuthority` is metadata a case may attach, not a global ranking. The engine has no
  built-in "newer" or "authoritative domain wins" policy.

`supported` means "supported by the supplied evidence," is defined that way in every
docstring and metric, and is never equated with "true."

## Reasoning

- **A boundary now, capability later.** Writing answer evaluation and correction against
  `GroundingEvaluator` means swapping structural for a semantic evaluator is an
  implementation change, not a refactor — the same discipline that let retrieval be built
  behind the `Retriever` protocol.
- **Naming the evaluator in the result** keeps a `supported` from a structural check
  distinguishable in a report from one a human or model asserted, so a number carries how
  it was obtained.
- **Refusing to pick a contradiction winner** is the conservative default `docs/correction.md`
  demands ("do not defend A merely because it came first," but equally do not silently
  discard it) — correction, not a scorer, is where a conflict is resolved.

## Trade-offs

- Structural grounding calls a claim with a *valid but wrong-source* citation unsupported
  only when a benchmark supplies expected evidence; without that expectation it can at
  best call the citation present. This is inherent to not understanding text and is stated,
  not hidden.
- A manual evaluator can be fed an optimistic judgement. That is a data-quality risk for
  whoever curates the benchmark, mitigated by recording the evaluator name so a report
  shows what produced each status.

## Consequences

- Answer evaluation (next) consumes `GroundingResult` and keeps grounding status a
  separate dimension from citation validity and retrieval quality — no single merged
  "quality score."
- Correction classifies prior claims against new evidence; the contradiction and status
  vocabulary here is what it operates on.
- A `SemanticGroundingEvaluator` is an explicitly named future implementation of the same
  protocol; nothing in this ADR claims it exists.
