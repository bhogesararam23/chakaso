# ADR-0033: Chakaso is a language-model project; the Python API is its primary interface

- Status: Accepted
- Date: 2026-10-21
- Relates to: [ADR-0001](ADR-0001-local-first-runtime.md), [ADR-0002](ADR-0002-language-model-boundary.md),
  [ADR-0005](ADR-0005-src-layout.md), [ADR-0025](ADR-0025-retrieval-aware-runtime.md),
  [ADR-0026](ADR-0026-application-composition-root.md), [ADR-0030](ADR-0030-minimal-optional-training-dependency.md),
  [ADR-0031](ADR-0031-grounded-answer-evaluation-workload.md)

## Context

[`../internal/engineering/model-centred-audit.md`](../internal/engineering/model-centred-audit.md) measured
the repository's shape rather than its prose: 131 modules and 20,697 lines under `src/chakaso/`, 93 test
modules, 32 decision records of which exactly one concerns training, 11 top-level CLI commands, and two
implementations behind `LanguageModel` — a development double and an adapter over a measured
984-parameter character model that generates incoherent text.

The founding goal in [`../../AGENTS.md`](../../AGENTS.md) is to train Chakaso's own model from random
initialization. The structure above is not that goal's shape. It is the shape of a project that builds the
system a model would live in, and defers the model. That deferral was deliberate once —
[`../roadmap.md`](../roadmap.md) states the rule "build the system before scaling the model", and
[ADR-0030](ADR-0030-minimal-optional-training-dependency.md) accepted a dependency sized for a toy on the
strength of it. E2601 and E2602 then showed the learning loop works, and E2603/E2604 measured retrieval and
the harness — while the model stayed a toy and the application kept growing.

Two facts make the deferral no longer self-justifying. First, the application work now *cannot* be
finished by building more of the application: P4, P5 and P10's remaining exit criteria all need a model
that generates (the audit's D3). Second, a model experiment that is not the centre of the project gets
sized by whatever the application leaves over, which is how the training layer ended up numpy-only,
character-level and out of CI (K-009).

**No application interface to a trained checkpoint exists.** Loading one today means constructing a
`ModelConfig` or `TrainedCharacterModel(checkpoint_path=…)` by hand; the documented entry point in the
README is `chakaso chat`, a terminal program.

## Problem

State, in a form that can be enforced by a reviewer or a test, which artifact is Chakaso's product and
which interfaces are primary — such that the next model step is not again gated on application work.

## Options considered

- **Option 1 — Keep the current ordering.** The system is the product; the model remains an experiment
  behind `[train]`. Costs: P4/P5/P10 stay permanently blocked, and every further ADR is written about a
  component no model has ever exercised.
- **Option 2 — Declare the model the product; a Python library API the primary interface; the CLI and the
  conversation system optional consumers of it.** Training, tokenization, checkpointing, inference and
  model evaluation become the centre of the roadmap; application subsystems stay in the repository, tested
  and useful, but stop being the default next unit of work.
- **Option 3 — Split the repository** into a model project and a system project, so the priorities cannot
  drift back into each other.
- **Option 4 — Make a graphical or hosted interface the primary surface** to make the model easy to try.

## Decision

**Adopt Option 2.** Concretely, and enforceably:

1. **The product is a trained language model plus the reproducibility machinery around it.** A change
   whose stated value is "another command" or "another application subsystem" is not the next unit of work
   while the model is a toy.
2. **The primary interface is the Python library.** A developer must be able to take a trained checkpoint
   and use it from Python — load it, generate from it — with no terminal invocation, no configuration file,
   and no import of the conversation runtime. This is delivered as a small, typed, documented entry point
   in `chakaso.models` that *is* a `LanguageModel` rather than a parallel model abstraction
   ([ADR-0002](ADR-0002-language-model-boundary.md) is unchanged).
3. **The CLI is a secondary developer utility.** Its handlers stay thin and delegate to the same Python
   services; no core model behaviour may be implemented only inside a CLI handler; useful commands are
   preserved rather than removed for symmetry. Deleting the CLI is explicitly *not* part of this decision.
4. **Dependency direction is normative:** corpus → tokenizer → model → training → checkpoint → inference
   API → model evaluation, with the application layer (retrieval, conversation, correction, persistence)
   composable *on top* and never required beneath. Model code must not import CLI code; training must not
   require an interactive session; inference must not require the training loop alive; application
   subsystems must not require the model layer to be importable.
5. **Model evaluation is a research discipline of its own**, distinct from harness evaluation. A model
   number must state its corpus, tokenizer, split, seed, parameter count and baseline; training loss and
   held-out loss are never conflated; a reference oracle, a retrieval benchmark or a citation checker is
   never evidence of language-model competence.
6. **Honesty is unchanged and still binding.** A character-level toy is not a capable LLM; no run,
   loss curve, perplexity, benchmark, parameter count or hardware number may be invented; implementing a
   mechanism is not the same as measuring it, and measuring it on a toy is not the same as usefulness.
7. **Web frontends, hosted APIs, desktop apps and agent frameworks are out of scope** unless a written
   technical justification exists that does not displace core model work.

## Reasoning

- **Only one of the two directions can be finished by adding code.** Application milestones P4/P5/P10 are
  blocked on a model that does not exist; the model milestone is blocked only on decisions and compute.
  Continuing Option 1 spends engineering on the half that is waiting.
- **A library interface is the smallest thing that can be tested.** `generate(messages, params)` behind a
  protocol already exists; a checkpoint loader is a thin, typed surface over it. A terminal or a frontend
  cannot be asserted about as precisely.
- **The existing boundaries make this cheap.** The audit found no coupling to undo — nothing in
  `chakaso.training` imports application code, and the hygiene tests already enforce the adapter-import
  rule structurally. Reordering priorities is therefore a documentation, API and dependency decision, not
  a rewrite, which is the reason Option 3 (split) is unnecessary.
- **Preserving the application is not a concession.** Retrieval, evidence identity, citation, correction
  and durable answers are tested assets and are the *only* reason a grounded model could later be measured
  honestly. Making them optional consumers costs them nothing.

## Trade-offs

- The project stops compounding its most visible surface. `chakaso chat` will not get new capabilities
  while the model path is built, and the roadmap's ordering rule is inverted, so previously-sequenced
  application units (P29 calibration, P32 runtime consumption of the sufficiency decision) are deferred by
  this decision rather than completed.
- A second numerical dependency becomes a maintenance and installation burden that the base install must
  not inherit; the isolation rules of [ADR-0030](ADR-0030-minimal-optional-training-dependency.md) carry
  forward and are re-decided, not assumed, in ADR-0034.
- "Primary interface = Python library" makes the project less demo-able to a visitor who only reads the
  README, which is precisely the audience a fluent terminal misleads.

## Rejected alternatives

- **Option 1 (no change).** Rejected because its own exit criteria are unreachable inside it: it keeps
  scheduling model work behind application work that needs the model.
- **Option 3 (split the repository).** Rejected: the coupling problem it would solve does not exist, and it
  would sever the one thing that makes a future grounded model measurable — the evaluation and evidence
  layers already in this repository.
- **Option 4 (frontend or hosted service first).** Rejected outright by rule 7: an interface in front of a
  toy model advertises a capability that does not exist, and ADR-0001 already forbids the hosted
  dependency that would make it convenient.

## Consequences

- `AGENTS.md`, `README.md`, [`../architecture.md`](../architecture.md) and
  [`../roadmap.md`](../roadmap.md) are realigned to this contract; the README leads with the model and the
  Python API, and the CLI is documented as an optional interface.
- The next unit of work must be a model-path unit. This makes P30 (corpus manifest, subword tokenizer,
  sequence packing) and P31 (Transformer, training, checkpoint, inference API, model metrics) the active
  arc, and defers P29 and P32 explicitly.
- The dependency decision that gates P31 must be re-made on measured evidence before its code: ADR-0034.
- A `load_model` entry point becomes public API and is therefore subject to the same compatibility
  discipline as the boundary: it needs tests that do not go through the CLI, and it must distinguish a
  trained Chakaso model, an experimental model, a development double and any external adapter.
- Model-versus-harness evaluation must stay distinguishable in documentation and naming; where a metric
  measures the harness rather than a model, the record says so.
