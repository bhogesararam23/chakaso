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

Configuration is TOML, loaded explicitly rather than discovered by magic. Versioned
configuration lives in [`../configs/`](../configs/); machine-local overrides use
`configs/local.toml`, which is git-ignored.

```bash
python -m chakaso config show
```

This prints the resolved configuration and where each value came from. Environment
values are not silently substituted for file values; an override has to be asked
for.

## Repository layout

```text
src/chakaso/           package source
  config/              typed configuration and loading
  core/                identifiers, hashing, errors, shared primitives
  models/              language-model interface, registry, implementations
  conversation/        conversation and message state
  evidence/            source records, evidence chunks, citation resolution
tests/                 unit, contract and repository-hygiene tests
configs/               versioned configuration files
docs/                  documentation (see docs/README.md)
experiments/           experiment records
```

Directories appear when they contain something real. If a component in
[`architecture.md`](architecture.md) has no directory here, it is not implemented.

## Contributing

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md). The short version: every
behavioural change needs a test, claims in documentation must match measured
reality, and a change in architecture needs a decision record.
