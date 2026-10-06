# Conventions

Operational rules for working in this repository. [`../../CONTRIBUTING.md`](../../CONTRIBUTING.md)
covers what to contribute; this covers how.

## Tooling

| Concern | Tool | Reason |
| --- | --- | --- |
| Test runner | pytest | Small readable tests, scales to functional suites |
| Formatter | ruff format | One tool for formatting and linting; fast; no plugin management |
| Linter | ruff check | Replaces flake8/isort/pyupgrade with one dependency |
| Type checker | mypy | Static boundary checking matters when interfaces are the product |
| Build backend | setuptools | Boring and universally understood; no build plugin needed |
| Configuration format | TOML, parsed with `tomllib` | Standard library since 3.11 ([ADR-0006](../decisions/ADR-0006-toml-configuration.md)) |

Do not add a second tool for a job one of these already does.

## Python

- Target Python 3.11+. `tomllib` and modern typing syntax are available; do not
  use syntax that requires a later version without updating the floor in
  `pyproject.toml` and CI.
- Type annotations on every public function and on all data crossing a boundary.
- `from __future__ import annotations` at the top of modules that use modern
  annotation syntax, so annotations stay unevaluated.
- Prefer `dataclasses` (with `frozen=True` where the value is a record) over
  hand-written `__init__`. Prefer `Protocol` over inheritance for boundaries that
  implementations should be free to satisfy structurally.
- No mutable default arguments. No mutable module-level state that a test could
  leak into.
- Explicit default values and keyword-only arguments for anything with more than
  two parameters, so call sites stay readable.

## Errors

- Define specific exception types in the module that raises them, deriving from a
  package-level base where callers need to catch a family.
- Error messages name the offending value, and say what was expected.
- Never a bare `except Exception` without a stated reason, and never a silent one.
- Do not convert a specific failure into a generic one. If a model does not
  support an operation, say that — do not raise "generation failed".

## Comments

Comments explain why something exists, what invariant it protects, or what
trade-off was accepted. They do not restate the code. Non-obvious research
assumptions and compatibility constraints belong in a comment; nothing else does.

## Configuration

- No magic constants in component code. A value that could reasonably differ
  between runs is configuration.
- Configuration is loaded explicitly and validated on load; a bad value fails at
  startup with a message naming the field and the file.
- Configuration files are versioned in `configs/`. Machine-local overrides go in
  `configs/local.toml`, which is git-ignored.
- Environment variables do not silently override file values. An override is
  requested explicitly or it does not happen.

## Tests

- `tests/` mirrors the package structure: `tests/test_<module>.py` for
  `src/chakaso/<module>.py`.
- Test names state the behaviour, not the method: `test_rejects_unknown_source_id`
  rather than `test_resolve_citation`.
- One behaviour per test. If a test needs two unrelated assertions, it is two
  tests.
- Tests must not depend on execution order, wall-clock time, network access, a GPU,
  or the developer's locale. Deterministic randomness: use an explicit seed.
- Failure paths are tested as carefully as success paths. A validator with no test
  for the input it rejects has no evidence that it rejects anything.
- Fixtures live in `tests/conftest.py` and are named for the thing they provide.
- Repository-hygiene tests (`tests/test_repository_hygiene.py`) are real tests and
  belong in CI. They enforce rules that are otherwise easy to break by accident —
  such as the private documentation pack never becoming tracked.

## Documentation

- Every capability claim carries a status: Implemented, Experimental, Planned,
  Research, Unknown. A sentence that implies a capability without a status is a
  defect.
- Do not duplicate a specification across documents. Link to it. Duplicated text
  goes stale in one place and stays correct in the other.
- Write directly and technically. No hype words. No claims about maturity.
- Where something is unknown, write that it is unknown.
- Update documentation in the same commit as the code it describes.

## Decision records

Write one when a decision constrains future code, is expensive to reverse, or has
plausible alternatives a future contributor would re-litigate. Format and status
vocabulary are in [`../decisions/README.md`](../decisions/README.md). Never edit an
accepted record's decision; supersede it.

## Commits

Types: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`, `ci:`, `research:`.

The subject line completes the sentence "this commit will…". The body explains why,
what was rejected, and what it costs. If a commit needs no body, the change is
probably too small to need its own commit — unless it is genuinely atomic.

## Branches

Bootstrap-scale changes may go directly to `main`. Anything that changes a
documented interface, adds a dependency, or spans more than one component goes on
a branch and through a pull request, so that CI runs before it lands.
