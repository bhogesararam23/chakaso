# ADR-0027: The semantic-evaluation contract carries evaluator version and an explicit error outcome; no real evaluator is simulated

- Status: Accepted
- Date: 2026-10-15
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0014](ADR-0014-claim-representation.md), [ADR-0015](ADR-0015-grounding-boundary.md), [ADR-0016](ADR-0016-correction-decision-rule.md), [ADR-0019](ADR-0019-semantic-grounding-boundary.md)

## Context

ADR-0019 established the `SemanticJudge` boundary with a single, explicitly-named fixture
implementation, so that answer evaluation and correction could be written against semantic grounding
before any real evaluator existed. The boundary was sound but the contract left three things
unrepresented that Phase-4-grade semantic evaluation needs: an evaluator **version** distinct from
its name (so a change of evaluator is visible in provenance), an explicit **error/abstention**
outcome for a judge that cannot decide, and an **honest capability report** so nothing downstream —
or a reader — mistakes the fixture for a real judge.

A tempting shortcut is to ship a "semantic evaluator" that is really token-overlap or a keyword
heuristic, or to reach for an NLI model. The first is exactly the fabrication the project forbids
(§2.1: never present a development double or token overlap as semantic understanding). The second
would make a heavyweight, weights-downloading, GPU-leaning dependency part of an offline-first,
CPU-only, dependency-free runtime, which ADR-0001 and the packaging rules refuse without a measured
need. There is, today, no dependency-free, offline, CPU-feasible real semantic evaluator that can be
built and tested honestly in this environment.

## Problem

How should the semantic-evaluation boundary be strengthened so it is a credible seam for a future
real evaluator — carrying identity, version, judgement, rationale, structured conflict and an
indeterminate outcome, with failures that cannot be misread as success — without inventing a
capability that does not exist?

## Options considered

- **Ship a heuristic "semantic judge" (overlap/keywords) and label it semantic.** Rejected: it
  fabricates intelligence that is not there (§2.1), and would make benchmark numbers describe string
  matching while claiming understanding.
- **Add a heavyweight local NLI/embedding dependency now.** Rejected: it breaks the dependency-free,
  offline, CPU-only guarantees (ADR-0001) with no measured need, and pulls model weights onto every
  install path.
- **Harden the contract, add capability detection, keep only the fixture, and document the missing
  real evaluator precisely.** Chosen.

## Decision

`chakaso.grounding.semantic` is extended additively:

- `SemanticEvaluation` gains an `evaluator_version` field, and judges may declare a `version`; the
  `SemanticGroundingEvaluator` records both name and version on the `GroundingResult`, so changing
  the evaluator identity or version is reflected in evaluation provenance rather than hidden behind
  a single generic name.
- A dedicated `SemanticJudgeError` marks an evaluator that fails or abstains on a claim. The adapter
  catches it per claim and records `NOT_EVALUATED` — never `SUPPORTED` — and one failing claim does
  not abort the whole answer. A structural result still needs no evaluator at all, so grounding stays
  usable without any semantic judge wired.
- `describe_semantic_capability()` reports honestly that no real evaluator is available and that the
  only judge is `FixtureSemanticJudge`, which replays caller-supplied relations. `chakaso
  capabilities` surfaces this so a run states plainly that its grounding is structural.

A versioned synthetic semantic-evaluation dataset and a deterministic scorer are delivered alongside
(see the follow-up research record) so a real evaluator can be measured when one exists; the fixture
is scored only to prove the harness, and its numbers are labelled structure-replay, never semantic
capability.

## Reasoning

- **The value of a boundary is that failures are honest at it.** Carrying version and an explicit
  error outcome means a future real evaluator is comparable and auditable, and a broken one degrades
  to "undecided", not to false support — the correction-integrity property ADR-0016 depends on.
- **Naming the missing capability is more useful than faking it.** A capability report and a precise
  limitation stop the fixture reading as understanding, which is the single most dangerous confusion
  this boundary invites.
- **Staying dependency-free keeps the guarantees real.** An NLI model can be added later behind the
  same `SemanticJudge` boundary as an optional adapter when there is a measured, offline-feasible
  implementation — a configuration change, not a redesign.

## Trade-offs

- There is still no semantic judgement the system can make on its own; grounded answers are checked
  only structurally. This is stated in `CURRENT_STATE.md` and the capability report, not glossed.
- The dataset scores a fixture as structure-replay; it is a development instrument, and its
  "accuracy" measures the harness, not understanding.

## Consequences

- A real semantic evaluator plugs into the existing `SemanticJudge` boundary (identity, version,
  rationale, error, conflict) without touching `evaluate_answer`, `reassess` or the correction rule.
- Correction composition keeps its guarantees (tested): an evaluator exception is never a support
  judgment, an unresolved judgment never becomes a confirmed contradiction, and structural grounding
  stays usable with no evaluator wired.
- The semantic capability stays explicitly Experimental/absent; closing it needs a genuine, offline,
  dependency-feasible evaluator and a measured evaluation, which is a separate milestone.
