# AGENTS.md

Instructions for coding agents working in this repository. Humans are welcome to
read it too; it is the same contract.

## What this repository is

Chakaso is a research project building a local-first conversational system that
grounds answers in retrieved evidence, cites sources by identifier, and corrects
itself when better evidence arrives. It intends to train its own language model
from random initialization. At the time of writing, none of the system exists —
see [`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md), which is the
authority on what is real.

## Read order

1. [`docs/agent/CURRENT_STATE.md`](docs/agent/CURRENT_STATE.md) — what exists right now.
2. [`docs/agent/ACTIVE_TASK.md`](docs/agent/ACTIVE_TASK.md) — the current unit of work and what follows it.
3. [`docs/agent/CONVENTIONS.md`](docs/agent/CONVENTIONS.md) — how code and docs are written here.
4. [`docs/architecture.md`](docs/architecture.md) — components and boundaries.
5. [`docs/decisions/`](docs/decisions/) — why the architecture is shaped this way. Read before proposing a change to any boundary.

## Non-negotiables

**Never commit the private documentation pack.** `docs/*.docx` is the author's
private planning material. It is git-ignored, must stay where it is, and must not
be deleted, moved, copied into the repository, or summarised wholesale into a
tracked file. `tests/test_repository_hygiene.py` fails if any of it becomes
tracked. If you find it staged, unstage it and say so.

**Do not describe planned work as done.** Every capability claim in documentation
carries one of: Implemented, Experimental, Planned, Research, Unknown. Definitions
are in [`docs/README.md`](docs/README.md). A change that implements something
updates that status in the same commit.

**Do not invent numbers, datasets, citations or experiments.** If it was not
measured, it is not written down. There are currently no measured results of any
kind in this repository, and it should stay obvious that this is true.

**The runtime must not require a commercial inference or search API.**
[ADR-0001](docs/decisions/ADR-0001-local-first-runtime.md). External providers may
only exist as optional adapters behind an existing interface.

**Application code must not import a concrete model implementation.**
[ADR-0002](docs/decisions/ADR-0002-language-model-boundary.md). Configuration maps
to a concrete model in exactly one place.

**The model may not produce source identity.**
[ADR-0003](docs/decisions/ADR-0003-evidence-identifier-ownership.md). Identifiers
come from the retrieval layer; an identifier the model was not given is an error
to record, never a URL to render.

**Do not create empty scaffolding.** No placeholder packages, no TODO-only
modules, no directories created because a roadmap mentions them. A module appears
when there is a real reason for it, with tests and a stated responsibility. A
directory listing future work belongs in [`docs/roadmap.md`](docs/roadmap.md).

**Do not weaken or delete a test to get CI green.** If it fails, the code or the
expectation is wrong.

**A changed interface needs a decision record or an update to the relevant
specification, in the same commit.**

## Commands

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

python -m pytest                   # tests
python -m ruff check .             # lint
python -m ruff format --check .    # format check
python -m mypy src                 # types
```

All four must pass before a commit is pushed. CI runs exactly these.

## Repository map

```text
src/chakaso/        package (src layout — you must install it or set PYTHONPATH)
tests/              unit, contract and repository-hygiene tests
configs/            versioned configuration, TOML
docs/               public documentation
docs/decisions/     architecture decision records
docs/research/      research log, hypotheses, literature, open questions
docs/agent/         this directory's context: current state, conventions, active task
docs/internal/      engineering notes that are not for external readers
experiments/        experiment records
```

The `src` layout is deliberate: it makes an uninstalled import fail rather than
silently testing the working tree
([ADR-0005](docs/decisions/ADR-0005-src-layout.md)).

## Working method

Inspect → understand → decide → implement one coherent unit → test → review →
document → commit → push → verify → next unit.

One commit is one coherent change that can be understood on its own. Types are
`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`, `research:`. Do
not split a single change into several commits, and do not bundle unrelated
changes into one.

Before every commit:

- the change is meaningful and the code is actually needed,
- tests cover the behaviour and pass,
- formatting, lint and types pass,
- documentation matches the code, including status labels,
- a decision record exists if a decision was made,
- `docs/agent/CURRENT_STATE.md` is updated if what exists has changed,
- `CHANGELOG.md` is updated if the change is notable for a reader,
- no private material and no secrets are staged.

## Traps

- **`docs/*.docx` is not repository content.** It is the private pack. Ignore it,
  and never assume it is absent because it is not in `git status`.
- **`git status` will not show the private pack** because it is ignored. Check
  tracked files instead if you need to be sure: `git ls-files | grep docx`.
- **The package is not importable without installing it** when using the
  repository as the working directory. That is intentional.
- **No GPU is available in CI.** Tests must be CPU-only and offline. A test that
  needs the network must be marked and skipped by default.
- **The research log is not a place for opinions.** Entry types are defined in
  [`docs/research/README.md`](docs/research/README.md).
