# Getting started

## What you can actually run

Chakaso has no application yet. There is no chat interface, no retrieval, no
trained model. What exists is a Python package with typed configuration, core
primitives, the language-model interface and the source/evidence records, plus
the test suite that covers them.

This document describes how to install and verify that package. It will grow as
the system does, and it will not describe anything that does not exist.

## Requirements

- Python 3.11 or later. (Chakaso reads TOML configuration with the standard
  library `tomllib`, which arrived in 3.11. See
  [`decisions/ADR-0006`](decisions/ADR-0006-toml-configuration.md).)
- `git`
- No GPU. Nothing in the test suite requires one, and CI does not have one.

## Setup

```bash
git clone https://github.com/bhogesararam23/chakaso.git
cd chakaso

python -m venv .venv
```

Activate it:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install the package with its development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`.[dev]` installs the runtime dependencies plus the tools used by CI: `pytest`,
`ruff` and `mypy`. Nothing else is required, and no dependency is a commercial
inference or search provider.

## Verify the install

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m chakaso --version
```

All five should succeed. If `mypy` is slow on a first run, that is normal; it
caches under `.mypy_cache/`.

The test suite is designed to run offline. If a test needs the network, it is
marked and skipped by default.

## Running without installing

The package uses a `src/` layout, so `python -c "import chakaso"` from the
repository root fails until the package is installed or the source directory is
on the path. That is deliberate: it prevents tests from passing against a
half-installed tree. To work without installing:

```bash
# Linux / macOS
PYTHONPATH=src python -m pytest

# Windows PowerShell
$env:PYTHONPATH = "src"; python -m pytest
```

## Configuration

Configuration is TOML, parsed with the standard library and loaded explicitly. No
file is discovered by scanning, and no environment variable is substituted for a
file value: a caller passes the files it wants, in order.

```bash
# The built-in defaults, with no file read at all
python -m chakaso config show

# The repository's versioned defaults
python -m chakaso config show --config configs/default.toml

# Versioned defaults plus a machine-local override, later files winning
python -m chakaso config show --config configs/default.toml --config configs/local.toml
```

Every value is printed with the source it came from, so "temperature was 0.0" can
be distinguished from "temperature was 0.0 because a file that has since been
edited said so".

An unknown key is an error, not a warning. A typo in a configuration file produces
a run that does not use the settings its author believes it uses, and that is worse
than a startup failure. The error names the file, the key, and the keys that do
exist.

## Repository layout

```text
src/chakaso/           package source
  config/              typed configuration, loading and provenance
  core/                identifiers, hashing, error base
  models/              language-model boundary, capabilities, registry
  evidence/            source records, evidence chunks, packs, citation resolution
  conversation/        immutable conversation state
tests/                 unit, contract and repository-hygiene tests
configs/               versioned configuration files
docs/                  documentation (see docs/README.md)
```

Only modules that exist appear here. `docs/architecture.md` describes the modules
that are planned and their status; a directory is created when it contains
something real, not in anticipation of it. Experiment records will live at
`experiments/<experiment-id>/record.md`, and that directory appears with the first
experiment.

## Contributing

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md). The short version: every
behavioural change needs a test, claims in documentation must match measured
reality, and a change in architecture needs a decision record.
