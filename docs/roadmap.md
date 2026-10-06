# Roadmap

The roadmap's rule: build the system before scaling the model, and require each
phase to produce a working artifact plus evidence that the next phase is
justified. Phases are not dates. A phase is done when its exit criteria are met,
and the criteria are written so that a phase cannot be declared done by assertion.

## Phases

| Phase | Milestone | Exit criteria | Status |
| --- | --- | --- | --- |
| P0 | Specification | Architecture, interfaces, constraints and evaluation plan written down | Done |
| P1 | Repository foundation | Package builds, tests and static checks pass in CI, decisions recorded | In progress |
| P2 | Local conversational shell | Model interface with a local implementation; conversation state; CLI that holds a conversation | In progress |
| P3 | Local retrieval | Documents can be ingested, indexed and retrieved as evidence with stable identifiers | Not started |
| P4 | Web-grounded answers | Permitted public retrieval produces source-linked answers with verified citations | Not started |
| P5 | Correction loop | Challenge → reassessment → correction works on fixed tests | Not started |
| P6 | Evaluation harness | Automated grounding, citation, retrieval and conversation benchmarks | Not started |
| P7 | Training pipeline | Tokenizer, data, configuration and checkpoint pipeline runs end to end | Not started |
| P8 | Chakaso-0 | Tiny model trains from random initialisation and learns a controlled corpus | Not started |
| P9 | Chakaso Base | Small model reaches useful language-model baselines | Not started |
| P10 | Chakaso Grounded | Instruction, grounding and correction behaviour improve on internal benchmarks | Not started |
| P11 | Domain selection | A defensible niche is chosen using explicit criteria | Not started |
| P12 | Domain adaptation | Specialized dataset and benchmark established | Not started |
| P13 | Research release | Model, code, evaluation and limitations published together | Not started |

P0 and P1 are deliberately boring and deliberately first. Most of what makes a
research project unreproducible is decided in the first week.

## P1 detail: what "foundation" means

The current phase is complete when all of the following hold:

- The package installs from a clean checkout and the tests pass.
- Formatting, linting, type checking and tests all run in CI without secrets and
  without network access to a model provider.
- Configuration is externalized, typed and validated, with clear failures on bad
  input.
- The language-model boundary exists with a documented contract and contract
  tests, and no business logic imports a concrete implementation.
- Source and evidence records exist with stable identifiers and content hashes.
- Documentation states what is implemented, what is planned and what does not
  exist, and the agent context matches reality.
- Every decision of consequence has a record explaining it.

## P2 detail: the first thing that behaves like a system

P2 is the first phase where Chakaso does something. Its scope is deliberately a
shell: a conversation can be held, state persists across turns, and a local model
answers through the boundary. It does not retrieve, does not cite, and does not
correct.

The reason to build the shell before retrieval is that the shell defines the state
that retrieval and correction will attach to. Building retrieval first would mean
retrofitting conversation state around evidence handling, which is backwards.

## First implementation sprint

The next concrete work, in order:

1. Configuration system with typed, validated configuration.
2. Core primitives: identifiers, content hashing, errors.
3. `LanguageModel` interface, capability declaration, registry.
4. One local model adapter, chosen for being runnable on CPU.
5. Conversation state and the Conversation Manager.
6. Source and evidence records with URL normalization.
7. Citation resolution with rejection of unknown identifiers.
8. Local document ingestion and a lexical retrieval baseline.
9. A retrieval interface for public web access that does not require a commercial
   search API.
10. Answer generation with source-identifier citations.
11. Reassessment entry point.
12. Unit and integration tests for all of the above.
13. A hand-built evaluation set of modest size, with unacceptable-evidence fields.
14. A CLI that holds a conversation.

Steps 1–3 and 6–7 are the current foundation work. Steps 4–5, 8, and 9–14 are
subsequent units.

## First model-training sprint

After the evaluation harness exists:

1. Collect a small corpus whose licence permits use.
2. Train a tokenizer and inspect its segmentations by hand.
3. Implement the tiny decoder-only Transformer and verify tensor shapes with
   tests.
4. Overfit a tiny dataset on purpose, to prove the training loop can learn at all.
5. Train a small experiment with a held-out split.
6. Log loss, throughput, memory and checkpoints.
7. Compare against a trivial baseline (a uniform or unigram predictor), because a
   loss curve with no baseline is not evidence of anything.

Scaling happens after reproducibility, not before.

## Domain-selection gate

The domain decision is deliberately deferred. When it is made, candidate domains
are scored against explicit criteria:

- availability of usable data,
- difficulty for general-purpose models,
- potential for proprietary or hard-to-replicate data,
- whether outcomes are measurable,
- access to practitioners or domain data,
- commercial value,
- research value,
- competitive landscape,
- whether the work can start cheaply.

A niche is not selected because it sounds impressive. The gate exists so that the
choice can be defended later.

## Long-term target

```text
                     Chakaso System
                           |
          +----------------+----------------+
          |                                 |
    Chakaso Model                  Domain Intelligence
          |                                 |
   Base + Instruct                  Retrieval + Tools
   + Grounding                      + Data + Memory
          |                                 |
          +----------------+----------------+
                           |
                    Evaluation Core
                           |
                     Feedback Loop
                           |
                    New Training Data
```

The loop at the bottom is the part that matters. The system's own measured
failures are the source of the next training data, which is why evaluation is
built before the model is scaled.

## Release discipline

- Every model release has a model card.
- Every dataset release has a data card and a licence note.
- Every benchmark result records configuration, dataset and version information.
- Every correction-behaviour change has regression results.
- Limitations are published rather than omitted.
- Prototype benchmark performance is not presented as general capability.
