# ADR-0005: src-layout Python package

- Status: Accepted
- Date: 2026-10-07

## Context

The pre-repository engineering handbook sketched the package as a `chakaso/`
directory at the repository root, alongside `training/`, `tests/`, `configs/` and
`docs/`. That is a flat layout.

Python packaging practice has converged on an alternative: keep the importable
package under `src/`, so the repository root contains no importable package.

The difference looks cosmetic and is not. With a flat layout, `python -c "import
chakaso"` from the repository root imports the working tree whether or not the
package is installed, and `pytest` run from the root does the same. With a
`src` layout, that import fails until the package is installed, so tests exercise
the installed distribution — including its metadata, its declared dependencies and
its entry points.

Chakaso also intends to grow component areas (`retrieval/`, `training/`,
`evaluation/`) inside the package. The handbook placed `training/` at the
repository root, outside the package.

## Problem

Should the importable package live at the repository root or under `src/`, and
should training code be part of the package?

## Options considered

- **Flat layout, as sketched in the handbook** — `chakaso/` at the root, with
  `training/` beside it.
- **src layout** — `src/chakaso/` with everything importable inside it, and
  `tests/`, `configs/`, `docs/` at the root.
- **Flat layout with a namespace package** — avoid the question by not having a
  top-level package at all.

## Decision

The package lives at `src/chakaso/`. Everything importable belongs to that single
distribution, including model-training code when it exists. Non-importable
material — tests, configuration files, documentation, experiment records — stays
at the repository root.

## Reasoning

- **Tests must exercise the installed package.** The failure this prevents is
  specific and common: a module or data file that is missing from the built
  distribution passes every test locally and fails on install. A `src` layout makes
  that class of bug impossible to miss, because the working tree is not importable.
- **Declared dependencies are actually used.** With a flat layout, a missing
  dependency can be masked by a stray module in the working directory. With `src`,
  the import graph is the installed one.
- **It keeps the root clean.** `src/` separates "the thing being shipped" from
  "the things used to build and describe it", which matters more as the repository
  accumulates `configs/`, `experiments/` and documentation.
- **A single distribution, not two.** The handbook's root-level `training/`
  suggested separating the system from the model work. That separation is real, but
  a second top-level directory is the wrong mechanism for it: it would need its own
  packaging, its own test discovery and its own import path. The separation is
  better expressed as a module boundary inside one package, which the project
  already needs anyway, and which keeps `chakaso.training` importable and testable
  in the same way as everything else.
- **Heavy training dependencies stay optional.** A separate distribution was
  considered as a way to keep inference-only installs small, but that is solvable
  with optional dependency extras, which do not require a layout decision.

## Trade-offs

- **An uninstalled import fails.** `python -c "import chakaso"` from the repository
  root raises `ModuleNotFoundError` until the package is installed or `PYTHONPATH`
  includes `src`. This surprises newcomers and is documented in
  [`../../docs/getting-started.md`](../../docs/getting-started.md) and
  [`../../AGENTS.md`](../../AGENTS.md), because an agent that does not know it will
  conclude the package is broken.
- **Editable installs are one extra step** before running anything.
- **Deviation from the planning sketch.** The handbook drew a flat layout. The
  layout change is recorded here rather than silently applied, and the handbook's
  intent — a system/model separation — is preserved as a module boundary.

## Rejected alternatives

**Flat layout as sketched** was rejected on the installed-package argument. Its
advantage is that `import chakaso` works immediately, which is a convenience that
costs correctness in exactly the area this project cannot afford it: reproducible
evaluation depends on the code that ran being the code that was installed.

**Namespace packages** were rejected because they add import machinery to solve a
problem the project does not have, and they interact badly with tooling that
assumes a single distribution root.

## Consequences

- `pyproject.toml` sets `package-dir = {"" = "src"}` and finds packages under
  `src`.
- Test running requires an install (`pip install -e ".[dev]"`) or an explicit
  `PYTHONPATH=src`. Both are documented.
- Model-training code becomes `chakaso.training`, which means the boundary between
  system and model work is enforced by the module graph rather than by directory
  convention.
- `python -m chakaso` and the `chakaso` console script both work after install, and
  the entry point is part of the installed metadata, so a broken entry point is
  caught by installation rather than by reading.
