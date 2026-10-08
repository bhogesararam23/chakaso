# Architecture decision records

Chakaso is a long-running research project. Most of its cost is in decisions
that were made early and then forgotten: why the model boundary looks the way it
does, why retrieval owns source identity, why a dependency was accepted. A
decision record is the cheapest way to keep that reasoning attached to the code.

## What belongs here

Write a record when a decision:

- constrains future code (an interface, a boundary, a storage format, a
  dependency, a version baseline), or
- is expensive to reverse, or
- has plausible alternatives that a future contributor would otherwise
  re-litigate.

Do not write a record for a rename, a bug fix, or a formatting choice. Do not
write a record for a decision nobody made.

## What does not belong here

Records describe decisions, not status. Current implementation status lives in
[`../agent/CURRENT_STATE.md`](../agent/CURRENT_STATE.md). A record that says
"Accepted" means the project is committed to the direction, not that the code
exists.

## Numbering and status

Records are numbered `ADR-NNNN` sequentially and never renumbered. The first
three records formalize decisions that were fixed before this repository
existed, so they are dated with the rest of the founding set.

A record's status is exactly one of:

| Status | Meaning |
| --- | --- |
| Proposed | Written down, not yet binding. |
| Accepted | Binding. The code is expected to follow it. |
| Rejected | Considered and deliberately not adopted. Kept so it is not re-proposed. |
| Superseded | Replaced by a later record, which must be linked. |
| Deprecated | Still binding on existing code, but no longer recommended for new code. |

A record is not edited after acceptance except to correct an error or to update
its status line. Changing a decision means writing a new record that supersedes
the old one.

## Format

```markdown
# ADR-NNNN: Title

- Status: Accepted
- Date: YYYY-MM-DD
- Supersedes: (optional) ADR-NNNN
- Superseded by: (optional) ADR-NNNN

## Context

The situation that produced the decision.

## Problem

The specific question that had to be answered.

## Options considered

- **Option** — what it means, what it costs.

## Decision

What was chosen, stated so it can be enforced.

## Reasoning

Why this option won.

## Trade-offs

What is given up. Every real decision gives something up.

## Rejected alternatives

Why the other options lost.

## Consequences

What this forces on the rest of the project, including follow-up decisions it
creates.
```

## Index

| Record | Title | Status |
| --- | --- | --- |
| [ADR-0001](ADR-0001-local-first-runtime.md) | Local-first runtime with no commercial inference or search dependency | Accepted |
| [ADR-0002](ADR-0002-language-model-boundary.md) | The application depends on a language-model boundary, not a model | Accepted |
| [ADR-0003](ADR-0003-evidence-identifier-ownership.md) | The retrieval layer owns evidence identifiers; the model only references them | Accepted |
| [ADR-0004](ADR-0004-apache-2.0-license.md) | Apache-2.0 as the project license | Accepted |
| [ADR-0005](ADR-0005-src-layout.md) | src-layout Python package | Accepted |
| [ADR-0006](ADR-0006-toml-configuration.md) | TOML configuration parsed with the standard library | Accepted |
| [ADR-0007](ADR-0007-content-derived-identifiers.md) | Evidence identifiers are derived from content, not assigned at random | Accepted |
| [ADR-0008](ADR-0008-transactional-turns.md) | A turn either completes or the conversation is unchanged | Accepted |
| [ADR-0009](ADR-0009-unresolved-references-are-recorded.md) | A reference the model was not given is recorded, not fatal | Accepted |
| [ADR-0010](ADR-0010-section-bounded-chunks.md) | Chunks are section-bounded and do not overlap | Accepted |
| [ADR-0011](ADR-0011-typed-source-references.md) | Source identity is a typed reference, not a URL | Accepted |
| [ADR-0012](ADR-0012-optional-bounded-fetcher.md) | The fetcher is an optional, explicitly-bounded adapter | Accepted |
| [ADR-0013](ADR-0013-benchmark-identity-and-versioning.md) | Benchmark cases have a stable identity and a content-derived version | Accepted |
| [ADR-0014](ADR-0014-claim-representation.md) | A claim is a first-class, content-identified unit; status is evaluation, not truth | Accepted |
