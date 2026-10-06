# ADR-0006: TOML configuration parsed with the standard library

- Status: Accepted
- Date: 2026-10-07

## Context

Chakaso needs versioned configuration files: model selection, retrieval settings,
paths, limits and later training hyperparameters. The pre-repository handbook
sketched `configs/default.yaml`, `development.yaml`, `evaluation.yaml` and
`training/*.yaml`, which presumes a YAML parser.

The project's dependency rule is that a dependency must do something the standard
library cannot. Since Python 3.11, `tomllib` parses TOML in the standard library.

## Problem

Which configuration format, and which parser?

## Options considered

- **YAML with PyYAML** — matches the handbook sketch, widely familiar,
  human-friendly.
- **TOML with `tomllib`** — standard library, no dependency, typed values,
  read-only.
- **JSON with `json`** — standard library, but no comments and unpleasant to edit
  by hand.
- **Python modules as configuration** — no format decision, but configuration
  becomes executable code and cannot be validated before it runs.
- **Environment variables plus a small parser** — no files, but no versioning
  either, which is the requirement.

## Decision

Configuration is TOML, parsed with `tomllib`. Configuration files live in
`configs/` and are versioned. Machine-local overrides use `configs/local.toml`,
which is git-ignored.

`tomllib` is read-only: nothing in the application writes configuration back over
a versioned file.

## Reasoning

- **No dependency.** PyYAML is a reasonable dependency, but it is not needed for
  this. A configuration system is not the place to spend the project's first
  dependency.
- **TOML has real types.** A limit is an integer, a timeout is an integer number of
  seconds (or an explicit string with units), and a flag is a boolean. YAML's
  willingness to turn `no` into `False` and `1.0` into a float where a version
  string was intended is a genuine source of configuration bugs, and the failure
  appears at runtime rather than at load.
- **Read-only is a feature.** Configuration that the code can rewrite is
  configuration that silently accumulates machine-specific state. The versioned
  files should only ever change through a commit.
- **TOML is already in the repository.** `pyproject.toml` is TOML. Contributors are
  already reading and editing it, so the format adds no new syntax to learn.
- **Explicit failure is achievable.** `tomllib` raises on malformed input, and
  typed validation on top of it produces messages that name the field and the file
  rather than a generic parse error.

## Trade-offs

- **Deeply nested configuration is uglier in TOML** than in YAML. The project's
  configuration is flat to shallowly nested, so this is a cost that is not paid.
  If a future configuration genuinely needs four levels of nesting, that is a
  signal the configuration is doing too much.
- **No anchors or aliases.** YAML can factor out repeated blocks. TOML cannot, so
  repetition is handled by defaults in code, which is more explicit but also more
  verbose.
- **Multi-line strings are clumsy.** Prompts and templates will need them. TOML's
  `"""` blocks work but are not pleasant; prompts are likely to live as versioned
  files rather than as configuration strings, which is arguably correct anyway
  because they need versioning independent of configuration.
- **Deviation from the handbook's `.yaml` file names.** Recorded here rather than
  applied silently.

## Rejected alternatives

**YAML with PyYAML** was rejected for the dependency, and secondarily because its
scalar coercion is a real failure mode. The dependency is small and the format is
familiar, so this was a close decision; the deciding factor is that the project
gains nothing from YAML that TOML does not provide at this nesting depth.

**JSON** was rejected because hand-edited configuration needs comments, and JSON
comments are not standard.

**Python modules as configuration** was rejected because it makes configuration
executable. A malformed value would run arbitrary code before validation could
reject it, and static checks on configuration become impossible.

**Environment variables alone** was rejected because the project's requirement is
versioned configuration with reproducibility, which is exactly what environment
variables do not provide. Environment values will still be usable later as explicit
overrides, but never as the source of truth.

## Consequences

- The Python floor becomes 3.11, since `tomllib` does not exist earlier
  ([ADR-0005](ADR-0005-src-layout.md) records the packaging decision; the floor is
  declared in `pyproject.toml` and tested in CI).
- `configs/` holds TOML files. `configs/local.toml` is git-ignored.
- Configuration loading must validate types and report the field and file on
  failure; `tomllib` alone only guarantees syntactically valid TOML.
- Nothing writes configuration files. If a feature needs to persist state, that
  state does not belong in `configs/`.
- Prompts and templates are expected to become versioned files rather than
  configuration values, which is a follow-up decision rather than part of this
  one.
