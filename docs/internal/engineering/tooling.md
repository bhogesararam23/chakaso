# Tooling

Why these tools and not others. A tool that is replaced should have its reasoning
replaced here too.

## Runtime and language

**Python 3.11 or later.** The floor is set by `tomllib`, in the standard library
from 3.11, which removes the need for a YAML or TOML dependency for configuration
([ADR-0006](../../decisions/ADR-0006-toml-configuration.md)). Nothing in the
project requires a feature newer than 3.11, so the floor is not raised without a
reason.

**Runtime dependencies: none, for now.** The foundation — configuration,
identifiers, records, the model boundary — is expressible with the standard
library. `dataclasses`, `enum`, `hashlib`, `urllib`, `typing` and `tomllib` cover
it. Heavy dependencies (PyTorch, tokenizers, an embedding stack, a vector index)
arrive with the module that needs them, and stay optional so the test environment
remains installable on a laptop with no GPU.

## Test runner: pytest

Chosen for readable small tests and a fixture model that scales to functional
suites. The alternatives considered were `unittest` (no dependency, but verbose
and poor at parametrisation) and `nose2` (unmaintained). pytest is the least
surprising choice for a Python research repository and is what a contributor
already knows.

Test selection is by marker where a test needs something unusual:

- `@pytest.mark.network` — requires network access; deselected by default so the
  suite runs offline.
- `@pytest.mark.slow` — takes more than a second or two.

`pyproject.toml` sets `addopts = "-m 'not network'"` so the default run never
reaches the network. A test that needs the network must be marked; CI does not
have model-provider access and never will.

## Formatter and linter: ruff

One dependency replacing `black`, `flake8`, `isort`, `pyupgrade` and several
plugins. It is fast enough to run on every commit, which is the property that
matters: a slow linter gets skipped.

Rule selection is explicit in `pyproject.toml` rather than the defaults, so that
enabling a rule is a visible change. Rules that fire on stylistic preference rather
than defects are not enabled.

## Type checker: mypy

Strictness is deliberate here, more than in a typical application, because this
project's product *is* its interfaces. `SourceRecord`, `EvidenceChunk`,
`LanguageModel` and the configuration types are the things other components depend
on, and a type error at one of those boundaries is a defect that tests written
against the same wrong assumption will not catch.

Configuration: strict mode with `disallow_untyped_defs`,
`disallow_incomplete_defs`, `warn_unused_ignores` and `no_implicit_optional`.
Third-party modules without stubs are handled per-module rather than by disabling
checking globally.

## Build backend: setuptools

`pyproject.toml` only, no `setup.py`. Setuptools was chosen over hatchling and
flit for being boring and universally understood. Nothing in the project needs a
build feature that would justify a less common backend, and a src layout means the
build configuration is short.

## Configuration format: TOML via `tomllib`

See [ADR-0006](../../decisions/ADR-0006-toml-configuration.md). The short version:
`tomllib` is in the standard library, TOML has real types (a duration is an
integer, not a string that resembles one), and it is read-only, which prevents
code from writing configuration back over the versioned files.

YAML was the obvious alternative and would have required PyYAML for typed
configuration. That is not an unreasonable dependency; it just is not needed.

## Continuous integration: GitHub Actions

The only realistic option for a repository hosted on GitHub, and it needs no
external service or secret. The workflow runs the same four commands a contributor
runs locally — `pytest`, `ruff check`, `ruff format --check`, `mypy src` — on a
matrix of Python versions so the declared floor is actually tested.

No model provider, no API key and no network-dependent test appears in CI. A CI
configuration that cannot run on a fork is not a check; it is a privilege.

## Deliberately not used

| Tool | Why not |
| --- | --- |
| Poetry / PDM | Solve dependency-management problems this project does not have. `pip` plus a venv plus a `[dev]` extra is sufficient and adds no lockfile to maintain. |
| tox / nox | CI already runs the matrix. Adding a local orchestrator duplicates the workflow file in a second place that drifts. |
| pre-commit | Useful, but it hides what CI runs behind another config file. The four commands are documented and short enough to run directly. Reconsider if contributors start pushing lint failures. |
| Black | ruff format does the same job without a second dependency. |
| A documentation generator | The documentation is prose about decisions and open questions, which no generator produces. |
