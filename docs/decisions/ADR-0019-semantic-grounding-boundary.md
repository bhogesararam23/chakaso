# ADR-0019: Semantic grounding is a boundary with only a fixture implementation today

- Status: Accepted
- Date: 2026-10-08
- Relates to: [ADR-0014](ADR-0014-claim-representation.md), [ADR-0015](ADR-0015-grounding-boundary.md), [ADR-0016](ADR-0016-correction-decision-rule.md)

## Context

ADR-0015 drew the line: the automatic grounding evaluator is *structural* — it decides only that a
claim cited supplied evidence, never that the evidence proves the claim. A real system eventually
needs the stronger judgement ("does this source actually support this claim?") for grounding to
mean what `docs/transparency.md` promises and for correction to detect a contradiction without a
human hand-asserting it.

There is no trained semantic judge, no model, and `docs/evaluation.md` forbids a fabricated
confidence number. But the *shape* of the judgement can be fixed now, and building answer
evaluation and correction against that shape is cheaper and less error-prone than retrofitting it
once a judge exists. The danger to design against is subtle: the moment a `SemanticEvaluator`
symbol exists, someone will feed it a keyword-overlap heuristic and call the output
"understanding." The decision must make that impossible to do by accident.

## Problem

How should the semantic grounding capability be specified so its *interface* is real and testable
today, while nothing in the repository can plausibly claim the semantic capability exists?

## Options considered

- **Skip it until a model exists.** Rejected: answer evaluation and correction are written now;
  without a boundary they would hard-code structural matching, and swapping in a judge later would
  be the exact cross-cutting rewrite ADR-0002/0015 exist to avoid.
- **Extend `GroundingResult` with a numeric "support score."** Rejected: no calibrated evaluator, so
  the number is fabricated (`docs/evaluation.md`, ADR-0001's honesty).
- **Build a keyword-overlap evaluator as the "semantic" one.** Rejected outright: overlap is
  structural in disguise, and shipping it under the word "semantic" is precisely the confusion the
  feature must prevent.
- **A narrow per-claim judge protocol, an adapter onto the existing grounding boundary, and one
  obviously-a-fixture implementation.** Chosen.

## Decision

`chakaso.grounding` defines:

- `SemanticJudge` — the per-claim boundary `judge(claim, evidence) -> SemanticEvaluation`, returning
  one of `supported / contradicted / uncertain / not_evaluated`. A judgement always means "as
  established by the evidence *as judged*", never "true" (ADR-0014).
- `SemanticGroundingEvaluator` — an adapter that turns a `SemanticJudge` into the existing
  `GroundingEvaluator` (answer-level → a `GroundingResult`), so `evaluate_answer` and `reassess`
  accept it with **no signature change**: choosing structural or semantic grounding is injecting a
  different object, not editing callers (ADR-0015).
- `FixtureSemanticJudge` — the only implementation. It returns relations a *caller supplied* and is
  named `fixture-semantic-0.1` so a report shows plainly that no semantic capability produced the
  number.

There is **no numeric confidence** anywhere on this boundary. Where a judge names two or more
conflicting chunks, the adapter records a `Contradiction` but names no winner — resolving a
conflict stays correction's job under an explicit precedence (ADR-0015/0016), not the evaluator's.
Every result carries the evaluator's name/version, so a run records *which* judge decided.

## Reasoning

- **Write against the boundary, not the capability.** Answer evaluation and correction become
  correct-by-construction for a future judge the moment the interface and a conforming (fixture)
  implementation exist and are tested.
- **The fixture is named to be un-mistakable.** The risk is not that a fixture exists — benchmarks
  need them — but that its output is read as understanding. Baking "fixture" into the identifier
  that lands in every report makes that misreading a visible choice, not an accident.
- **Judgement, not score.** Four discrete states mirror the claim-status vocabulary the rest of the
  system uses; inventing a float would both fabricate calibration and fork the status language.

## Trade-offs

- The boundary's honest emptiness means, today, semantic grounding adds no *automatic* capability
  over the manual evaluator — it reorganises the same caller-supplied judgement into a per-claim,
  evidence-shaped interface. That is worth it precisely because it is the interface a real judge
  will implement, and reorganising it now costs nothing the system already had.
- A `FixtureSemanticJudge` can be fed an optimistic relation, exactly like `ManualGroundingEvaluator`.
  Mitigated as in ADR-0015: the evaluator name is recorded, so a report shows a `fixture-semantic-*`
  produced the status and no claim of intelligence is available to hide behind.

## Consequences

- `evaluate_answer` and `reassess` can now be run with a semantic evaluator injected; the answer
  still reports grounding as a separate dimension, now attributed to whichever evaluator ran
  (structural or fixture-semantic), never merged with citation or retrieval.
- A future trained Chakaso model implements `SemanticJudge`; nothing in this ADR claims one exists,
  and external judge APIs stay out of the runtime (ADR-0001).
- The correction benchmark can label cases with a *semantic* expected relation through the same
  fixture boundary, keeping the benchmark honest about what is fixture versus measured.
