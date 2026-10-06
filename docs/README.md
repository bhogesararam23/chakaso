# Chakaso documentation

This directory holds three kinds of material that must not be confused with each
other.

| Kind | Location | Audience | Rule |
| --- | --- | --- | --- |
| **Public** | `docs/*.md`, `docs/decisions/`, `docs/research/` | Anyone | Written to be accurate and complete on its own. Says what is not implemented. |
| **Internal engineering** | `docs/internal/` | Contributors and agents | Working notes: experiment records, artifact policy, environment notes. May reference half-finished thinking, but must not contradict the public docs. |
| **Private** | not in this repository | The project author | Planning and specification material kept outside version control. It is source material, never repository content. |

If a private note and a public document disagree, the public document is wrong
and should be fixed. A private note is not evidence that something is
implemented.

## How to read the status labels

Documentation in this repository labels capability rather than promising it.

| Label | Meaning |
| --- | --- |
| **Implemented** | Code exists in this repository and is covered by tests that pass. |
| **Experimental** | Code exists and runs, but the approach is not settled. |
| **Planned** | Designed in enough detail to build; not built. |
| **Research** | An open question with no committed design. |
| **Unknown** | Not yet determined. |

"Planned" never means "mostly done". If a document describes a mechanism and does
not label it, assume it is Planned and verify against
[`agent/CURRENT_STATE.md`](agent/CURRENT_STATE.md).

## Public documents

| Document | Contents |
| --- | --- |
| [`getting-started.md`](getting-started.md) | Setup, tests, and what "running Chakaso" means today |
| [`architecture.md`](architecture.md) | Components, responsibilities, boundaries, data flow |
| [`transparency.md`](transparency.md) | Observed evidence vs model knowledge vs inference; what is exposed |
| [`retrieval.md`](retrieval.md) | Retrieval pipeline, source records, evidence chunks, citation resolution |
| [`correction.md`](correction.md) | Reassessment and the correction contract |
| [`training.md`](training.md) | Tokenizer, dataset pipeline, model stages |
| [`evaluation.md`](evaluation.md) | Evaluation layers, metrics and regression gates |
| [`roadmap.md`](roadmap.md) | Milestones and their exit criteria |
| [`research/`](research/) | Research log, literature, hypotheses, open questions |
| [`decisions/`](decisions/) | Architecture decision records |

## Internal documents

| Document | Contents |
| --- | --- |
| [`internal/engineering/repository-history.md`](internal/engineering/repository-history.md) | How this repository's history was started, and what it replaced |
| [`internal/engineering/artifact-policy.md`](internal/engineering/artifact-policy.md) | Where weights, datasets and generated outputs live, and what is never committed |
| [`internal/engineering/tooling.md`](internal/engineering/tooling.md) | The chosen formatter, linter, type checker and test runner, and why |
| [`internal/experiments/README.md`](internal/experiments/README.md) | The experiment record format, and the fact that no experiment has been run |
| [`internal/model-development/`](internal/model-development/) | Model development notes |

## Agent context

[`agent/`](agent/) holds durable context for coding agents so that a repository
can be understood without the conversation that built it. It is written to be
current, and it is a defect when it is not.
