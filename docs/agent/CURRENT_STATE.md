# Current state

Last updated: 2026-10-07, at commit `docs: add the contribution guide`.

This file is the authority on what exists. If it disagrees with any other
document, this file is right and the other document is a defect.

A private planning note is never evidence that something is implemented.

## What exists

| Thing | Where | Notes |
| --- | --- | --- |
| Repository conventions | `.gitignore`, `.gitattributes`, `.editorconfig` | Private docx pack is excluded by `.gitignore` |
| License | `LICENSE` | Apache-2.0, rationale in ADR-0004 |
| Public documentation | `README.md`, `docs/*.md`, `docs/research/` | Design specifications with status labels |
| Decision records | `docs/decisions/` | ADR-0001 to ADR-0004 |
| Contribution guide | `CONTRIBUTING.md` | |
| Agent contract | `AGENTS.md`, `docs/agent/` | This directory |
| Python package | **does not exist** | Not yet created |
| Tests | **do not exist** | Not yet created |
| CI | **does not exist** | Not yet created |
| Configuration | **does not exist** | Specified in `docs/architecture.md` and ADR-0006 |

## What works

Nothing can be run. There is no package to install, no CLI, no test suite and no
CI. The repository is currently a specification with conventions.

This is the accurate state, and the README says the same thing.

## What is tested

Nothing. There is no test suite yet, so no behaviour in this repository has been
verified by anything other than reading.

## What is experimental

Nothing. There is no code to be experimental.

## What is not implemented

Everything in the system, stated as a list so that its length is visible:

- Python package, CLI, configuration loading
- Language-model interface, registry, and every model adapter
- Conversation state and the conversation manager
- Query planner
- Retrieval: fetch, parse, chunk, rank, index, embeddings
- Evidence pack assembly and citation resolution
- Grounding and citation validation
- Reassessment and the correction loop
- Evaluation harness, benchmarks and metrics
- Tokenizer, dataset pipeline, model code, training loop
- Any model weights, of any size

## Decisions made

| Record | Decision |
| --- | --- |
| ADR-0001 | No required commercial inference or search dependency at runtime |
| ADR-0002 | Application code depends on a language-model boundary, not a model |
| ADR-0003 | The retrieval layer owns evidence identifiers; the model only references them |
| ADR-0004 | Apache-2.0 for code; weights and datasets need their own policy before release |

Decisions referenced but not yet written: ADR-0005 (src layout) and ADR-0006
(TOML configuration). Both are made and both are cited from documents in this
repository; the records are written as part of the package-metadata unit.

Two process facts are recorded outside the ADR series because they are repository
history rather than architecture:

- The repository history was restarted from an empty tree; the previous head is
  preserved as tag `archive/bootstrap-attempt-1`
  ([`../internal/engineering/repository-history.md`](../internal/engineering/repository-history.md)).
- The private planning pack lives in `docs/*.docx`, is git-ignored, and is never
  committed.

## Known gaps and risks

See [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Next step

The next unit is the Python project foundation: `pyproject.toml` with the
`[dev]` extra, a `src/chakaso/` package with a version and a minimal CLI, a smoke
test, and CI running format, lint, types and tests. ADR-0005 and ADR-0006 are
written in that unit, since the layout and configuration-format decisions are what
the metadata encodes.

After that: typed configuration, then core primitives, then the language-model
boundary, then source/evidence/conversation records. See
[`ACTIVE_TASK.md`](ACTIVE_TASK.md).
